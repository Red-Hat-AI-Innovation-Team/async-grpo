import argparse
from collections import Counter
from copy import deepcopy
from functools import partial
from hashlib import sha256
import json
import os
import random
import logging
import time
import asyncio
import ray
import logging
import atexit
import uuid
import torch
import numpy as np
from numba import njit
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.inputs import TokensPrompt
from transformers import AutoTokenizer

from vllm_registry import get_or_create_registry 
from verifier_pool import get_or_create_verifier_pool
from reward_registry import RewardType


import numpy as np
from numba import njit
logging.getLogger('numba').setLevel(logging.WARNING)

logging.basicConfig(level=logging.DEBUG)
logging.getLogger().setLevel(logging.INFO)



delimiter = '\n\n'
# special_phrases = ['', 'The answer is \\boxed{', 'Let\'s doublecheck!', 'Alternatively']
special_phrases = [
    'Wait!',
    # 'Let\'s try another method to solve the problem:',
    # 'Now, let\'s think again:',
    # 'Let\'s doublecheck the work so far.',
    # 'Alternatively',
    # 'Let\'s look at it from a different perspective:',
]

def get_indices_of_delimiter(response, delimiter):
    indices = []
    start = 0
    while True:
        index = response.find(delimiter, start)
        if index == -1:
            break
        indices.append(index)
        start = index + len(delimiter)
    return indices

def insert_phrase(response, delimiter, special_phrases, eos_str):
    """
    Modifies the response by finding all occurrences of the delimiter,
    choosing one occurrence at random, truncating the response at that point,
    and appending a random phrase from special_phrases.
    
    Parameters:
        response (str): The original string.
        delimiter (str): The delimiter to search for in response.
        special_phrases (list of str): A list of phrases to randomly append.
    
    Returns:
        str: The modified string.
    """
    chosen_index = None
    # Find all indices where the delimiter occurs
    indices = get_indices_of_delimiter(response, delimiter)
    
    # If we found any delimiters, choose one at random and truncate the response.
    if not indices:
        delimiter = '\n'
        indices = get_indices_of_delimiter(response, delimiter)

    if indices:
        chosen_index = random.choice(indices)
        # Option 1: If you want to discard the delimiter itself, use:
        truncated_response = response[:(chosen_index + len(delimiter))]
    else:
        # If no delimiter is found, just keep the full response.
        truncated_response = response.split(eos_str)[0] + delimiter
        print(f"\033[1;38;2;255;0;0mNo delimiter found in response\033[0m")
    

    # Append a random phrase from special_phrases.
    random_phrase = random.choice(special_phrases)

    delimiter_not_found = chosen_index is None
    
    return truncated_response + random_phrase, delimiter_not_found

async def rewrite_with_insert_phrase(sample, tokenizer):
    full_text = sample['sample_text']
    original_output = full_text.split(sample['input'])[-1]
    modified_output, delimiter_not_found = insert_phrase(original_output, delimiter, special_phrases, tokenizer.eos_token)
    sample['input'] = sample['input'] + modified_output
    sample['input_token_ids'] = tokenizer.encode(sample['input'])
    sample['delimiter_not_found'] = delimiter_not_found
    return sample

@njit
def normalize_rewards(rewards):
    """
    Normalize rewards within each group to compute advantages.

    Parameters:
        rewards : np.ndarray (1D)
            Array of rewards for each sample of shape (n_samples,).

    Returns:
        np.ndarray (1D)
            Normalized rewards of shape (n_samples,).
    """
    mean = np.mean(rewards)
    std = np.std(rewards) + 1e-4
    return (rewards - mean) / std

class BaseVLLMWorker:
    """
    Base vLLM worker class encapsulating common logic.
    Inference requires an input_data dict and additional **kwargs to update SamplingParams.
    Subclasses must implement:
      - get_engine_args(...) to provide engine configuration.
    Generation workers allow dynamic generation parameters (temperature, n, max tokens),
    while logprob workers enforce fixed parameters for single token generation.
    """
    def __init__(self, model_path: str, worker_id: str,
                 tensor_parallel_size: int = 1, max_num_seqs: int = 16, overhead_seqs: int = 8):
        self.overhead_seqs = overhead_seqs
        self.counter = 0
        self.model_path = model_path
        self.worker_id = worker_id
        print(f"Initializing {self.__class__.__name__} with model path: {model_path}")
        self.engine_args = self.get_engine_args(model_path, tensor_parallel_size, max_num_seqs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.llm = AsyncLLMEngine.from_engine_args(self.engine_args, start_engine_loop=True)
        self.registry = get_or_create_registry("generation_vllm_registry")
        self.setup_registration()
    
    def get_engine_args(self, model_path: str, tensor_parallel_size: int, max_num_seqs: int) -> AsyncEngineArgs:
        raise NotImplementedError("Subclasses must implement get_engine_args method.")
    
    def setup_registration(self):
        try:
            '''double dereference since the returned value is an object storage reference from ray.put'''
            last_weights = ray.get(ray.get(self.registry.get_last_weights.remote()))
            if last_weights is not None:
                self.update_weights(last_weights)
            ray.get(self.registry.register.remote(service_id=self.worker_id, max_load=self.engine_args.max_num_seqs+self.overhead_seqs))
            print(f"Worker {self.worker_id} registered.")
        except Exception as e:
            print(f"Error during registration for worker {self.worker_id}: {e}")
    
    def update_weights(self, new_state_dict: dict):
        llm_model = self.llm.engine.model_executor.driver_worker.model_runner.model
        llm_model.load_weights(new_state_dict.items())
        print(f"vLLM weights updated successfully on service {self.worker_id}.")
        return True
    
    async def inference(self, sample: dict, **kwargs) -> list[dict]:
        """
        Base inference:
          - Input: input_data must include 'input_token_ids'.
          - **kwargs can include extra parameters.
          - Constructs a SamplingParams object and then performs inference.
        """
        try:
            input_ids = sample['input_token_ids']
            sampling_params = SamplingParams(
                **kwargs
            )
            request_id = sha256((str(sampling_params) + str(input_ids) + str(self.counter))
                                .encode('utf-8')).hexdigest()
            self.counter += 1
            print(f"[{self.__class__.__name__} {self.worker_id}] Sampling params: {sampling_params}")
            generator = self.llm.generate(
                prompt=TokensPrompt(prompt_token_ids=input_ids),
                sampling_params=sampling_params,
                request_id=request_id
            )
            result = None
            async for out in generator:
                result = out
            return result
        except Exception as e:
            print(f"\033[38;5;196m\033[1mError during inference for worker {self.worker_id}: {e}\033[0m")
            import traceback
            traceback.print_exc()
            raise e
        
def majority_vote(samples: list[dict]) -> list[dict]:
    """
    Get the majority vote answer from the samples.
    """
    maj_vote_answer = [sample.get('parsed_attempt', None) for sample in samples if sample.get('parsed_attempt', None) is not None]
    estimated_label, majority_count = Counter(maj_vote_answer).most_common(1)[0]
    for s in samples:
        s['maj_vote_answer'] = estimated_label
        s['maj_ratio'] = majority_count / len(samples)
    return samples


@ray.remote
class GenerationVLLMWorker(BaseVLLMWorker):
    def __init__(self, model_path: str, worker_id: str, tensor_parallel_size: int, max_num_seqs: int,
                 global_num_verifiers: int = 4, write_failed: bool = False, overhead_seqs: int = 8, enable_prefix_caching: bool = True,
                 reward_fns: list[RewardType] = [RewardType.MATHD, RewardType.SYMPY]):
        reward_enum_list = [RewardType(fn) for fn in reward_fns]
        # Initialize verifier pool with selected reward functions
        self.verifier_pool = get_or_create_verifier_pool(global_num_verifiers, write_failed, reward_enum_list)
        self.enable_prefix_caching = enable_prefix_caching
        super().__init__(model_path, worker_id, tensor_parallel_size, max_num_seqs, overhead_seqs)
    
    def get_engine_args(self, model_path: str, tensor_parallel_size: int, max_num_seqs: int) -> AsyncEngineArgs:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path)
        return AsyncEngineArgs(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.98,
            dtype=torch.bfloat16,
            enable_prefix_caching=self.enable_prefix_caching,
            max_num_seqs=max_num_seqs,
            max_model_len=config.max_position_embeddings,
            disable_log_requests=False,
            disable_log_stats=False,
        )
    
    def get_max_tokens(self, sample: dict, max_tokens=None) -> int:
        max_tokens = max_tokens if max_tokens is not None else self.engine_args.max_model_len
        max_tokens = max_tokens - len(sample['input_token_ids']) - 1
        if max_tokens <= 0:
            max_tokens = 1
            print(f"\033[1;38;2;255;165;0mMax tokens is less than 0 for sample: \033[0m {sample['input']}")
        return max_tokens
    
    def get_gen_kwargs(self, sample: dict, **kwargs) -> dict:
        return {
            "n": kwargs.get("n", 1),
            "max_tokens": self.get_max_tokens(sample, kwargs.get("max_tokens", self.engine_args.max_model_len)),
            "temperature": kwargs.get("temperature", 0.7),
            "include_stop_str_in_output": kwargs.get("include_stop_str_in_output", True),
            "spaces_between_special_tokens": False,
            "skip_special_tokens": False,
        }
    
    async def inference(self, sample: dict, **kwargs) -> list[dict]:
        insert_reasoning_phrases = kwargs.get("insert_reasoning_phrases", False)
        
        generation_kwargs = self.get_gen_kwargs(sample, **kwargs)

        request_out = await super().inference(sample, **generation_kwargs)
        if 'input_len' not in sample:
            sample['input_len'] = len(sample['input_token_ids'])
        
        samples = [deepcopy(sample) for _ in range(len(request_out.outputs))]
        
        sample_extract_answer_futures = []
        for sample, out in zip(samples, request_out.outputs):
            sample['modified_reward'] = None
            sample['delimiter_not_found'] = False
            sample['output_token_ids'] = list(out.token_ids)
            sample['output_len'] = len(sample['output_token_ids'])
            sample['sample_ids'] = sample['input_token_ids'] + sample['output_token_ids']
            sample['sample_text'] = self.tokenizer.decode(sample['sample_ids'])
            sample['sample_position_ids'] = list(range(len(sample['sample_ids'])))
            sample['truncated_sample'] = sample['sample_ids'][-1] != self.tokenizer.eos_token_id
            # Compute per-sample labels for logprobs
            labels = [-100] * len(sample['sample_ids'])
            if not sample['truncated_sample']:
                for i in range(sample['output_len']):
                    pos = sample['input_len'] + i
                    labels[pos] = sample['sample_ids'][pos]
            sample['labels'] = labels
            sample['num_non_masked_output_tokens'] = sum(1 for label in labels if label != -100)
            # Use the remote call because verifier_pool is now a ray actor
            sample_extract_answer_futures.append(
                self.verifier_pool.verify.remote(
                    sample,
                    max_gen_length=kwargs.get("max_tokens", self.engine_args.max_model_len),
                    reward_fns=[RewardType.TTRL_EXTRACT_ANSWER]
                )
            )
        logging.debug(f"\033[1;38;2;255;165;0mFirst sample before rewriting: \033[0m {samples[0]['sample_text']}")
        sample_extract_answer_results = await asyncio.gather(*sample_extract_answer_futures)
        samples = majority_vote(sample_extract_answer_results)
        sample_rewards_futures = []
        for sample, extract_answer_result in zip(samples, sample_extract_answer_results):
            sample['parsed_attempt'] = extract_answer_result['parsed_attempt']
            sample_rewards_futures.append(
                self.verifier_pool.verify.remote(
                    sample,
                    max_gen_length=kwargs.get("max_tokens", self.engine_args.max_model_len),
                    reward_fns=[RewardType.TTRL_REWARD]
                )
            )

        # if insert_reasoning_phrases:
        #     modified_samples = [deepcopy(sample) for sample in samples]
        #     modified_samples = await asyncio.gather(*[rewrite_with_insert_phrase(sample, self.tokenizer) for sample in modified_samples])
        #     logging.debug(f"\033[1;38;2;255;165;0mFirst sample after rewriting: \033[0m {modified_samples[0]['input']}")
            
        #     kwargs['n'] = 1
        #     modified_requests_out = await asyncio.gather(*[
        #         super().inference(s, **self.get_gen_kwargs(s, include_stop_str_in_output=False, **kwargs)) 
        #         for s in modified_samples
        #     ])
        #     modified_rewards_futures = []
        #     for modified_sample, sample, out in zip(modified_samples, samples, modified_requests_out):
        #         modified_sample['input'] = sample['input']  # original input
        #         modified_sample['sample_ids'] = modified_sample['input_token_ids'] + list(out.outputs[0].token_ids)
        #         modified_sample['input_token_ids'] = sample['input_token_ids']  # original input token ids
        #         modified_sample['output_token_ids'] = modified_sample['sample_ids'][len(modified_sample['input_token_ids']):]
        #         modified_sample['output_len'] = len(modified_sample['output_token_ids'])
        #         modified_sample['sample_text'] = self.tokenizer.decode(modified_sample['sample_ids'])
        #         modified_sample['sample_position_ids'] = list(range(len(modified_sample['sample_ids'])))
        #         # Compute per-sample labels for logprobs
        #         labels = [-100] * len(modified_sample['sample_ids'])
        #         if not modified_sample['truncated_sample']:
        #             for i in range(modified_sample['output_len']):
        #                 pos = modified_sample['input_len'] + i
        #                 labels[pos] = modified_sample['sample_ids'][pos]
        #         modified_sample['labels'] = labels
        #         modified_rewards_futures.append(
        #             self.verifier_pool.verify.remote(
        #                 modified_sample,
        #                 max_gen_length=kwargs.get("max_tokens", self.engine_args.max_model_len)
        #             )
        #         )
        #     logging.debug(f"\033[1;38;2;255;165;0mFirst sample after generating with rewritten input: \033[0m {modified_samples[0]['sample_text']}")
        #     modified_results = await asyncio.gather(*modified_rewards_futures)
        #     for s in modified_results:
        #         s['modified_reward'] = s['reward']
        #     final_samples = modified_results + (await asyncio.gather(*sample_rewards_futures))
        # else:
        final_samples = await asyncio.gather(*sample_rewards_futures)

        
        group_rewards = np.array([s['reward'] for s in final_samples])
        max_reward = np.max(group_rewards).item()
        group_advantages = normalize_rewards(group_rewards)
        for sample_, advantage in zip(final_samples, group_advantages):
            sample_['advantage'] = advantage.item()
            sample_['max_reward_in_group'] = max_reward
            sample_['advantage_is_zero'] = advantage == 0
        
        print(f"\033[38;5;201mWorker \033[0m {self.worker_id} \033[38;5;201mfinished inference with \033[0m {len(final_samples)} samples.")
        return final_samples


if __name__ == "__main__":
    ray.init(address="auto", namespace="test")
    import nest_asyncio
    nest_asyncio.apply()
    parser = argparse.ArgumentParser(description="Start vLLM generation worker service")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model weights")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of tensor parallel units")
    parser.add_argument("--max_num_seqs", type=int, default=16, help="Maximum number of sequences to generate")
    parser.add_argument("--global_num_verifiers", "--num_verifiers", dest="num_verifiers", type=int, default=4, help="Number of verifier workers")
    parser.add_argument("--write_failed", action="store_true", help="Write failed generation samples to file")
    parser.add_argument("--overhead_seqs", type=int, default=8, help="Number of extra sequences to send over the limit")
    parser.add_argument("--enable_prefix_caching", type=lambda x: x.lower()=="true", default=True, help="Enable prefix caching for the generation worker")
    parser.add_argument("--reward_fns", type=str, default="", help="Comma-separated reward function names (mathd,sympy,countdown)")
    args = parser.parse_args()

    # Prepare runtime_env for the worker (as seen in worker_dispatcher.py)
    runtime_env = {"env_vars": dict(os.environ)}
    # Remove CUDA device pinning from the environment
    runtime_env["env_vars"].pop("CUDA_VISIBLE_DEVICES", None)
    # Install vllm and verifier dependencies
    runtime_env["pip"] = [
        f"-r {os.path.join(os.path.dirname(__file__), 'requirements_vllm.txt')}",
        "math-verify[antlr4_13_2]"
    ]
    runtime_env["excludes"] = ["*.pyc", "__pycache__"]

    # Prepare reward_fns list
    reward_fns_list = args.reward_fns.split(',') if args.reward_fns else []
    # Spawn the GenerationVLLMWorker actor with dispatcher-style options
    service_id = f"generation_worker_{uuid.uuid4()}"
    worker = GenerationVLLMWorker.options(
        name=service_id,
        num_gpus=args.tensor_parallel_size,
        num_cpus=1,
        max_restarts=-1,
        runtime_env=runtime_env
    ).remote(
        model_path=args.model_path,
        worker_id=service_id,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.max_num_seqs,
        global_num_verifiers=args.num_verifiers,
        write_failed=args.write_failed,
        overhead_seqs=args.overhead_seqs,
        enable_prefix_caching=args.enable_prefix_caching,
        reward_fns=reward_fns_list
    )

    registry_name = "generation_vllm_registry"
    # Wait for the registry to be available.
    while True:
        try:
            ray.get_actor(registry_name)
            print(f"{registry_name} found.")
            break
        except Exception:
            print(f"{registry_name} not found, sleeping for 1 second...")
            time.sleep(1)

    async def main():
        # Define prefix token IDs and decode to text prompt
        prompt_text = "How would you compute the volume of a shape made of the union of two spheres of the same radius where the centers coincide with the surface of the other sphere?"
        messages = [
            # {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text}
        ]
        # input_ids = [
        #     100264, 882, 100266, 4438, 1053, 499, 12849, 279, 8286, 315,
        #     264, 6211, 1903, 315, 279, 11552, 315, 1403, 66818, 315,
        #     279, 1890, 10801, 1405, 279, 19169, 72359, 449, 279, 7479,
        #     315, 279, 1023, 26436, 30, 100265, 100264, 78191, 100266
        # ]
        # # Build tokenizer and prompt text
        # prompt_text = tokenizer.decode(input_ids)
        # Prepare sample for generation (with trailing token) and for registry
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False)
        input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        gen_ids = input_ids
        gen_text = tokenizer.decode(gen_ids)
        sample_gen = {'input': gen_text, 'input_token_ids': gen_ids}
        sample_reg = {'input': prompt_text, 'input_token_ids': input_ids}

        actor = ray.get_actor(service_id)
        result1 = await actor.inference.remote(sample=sample_gen, max_tokens=100)
        result2 = await actor.inference.remote(sample=sample_gen, max_tokens=100)

        registry = get_or_create_registry(registry_name)
        tasks = [registry.inference_balanced.remote(sample_reg) for _ in range(2)]
        results = await asyncio.gather(*tasks)
        return result1, result2, results
    
    results = asyncio.run(main())
    print(results)
    from IPython import embed
    embed()
    # Keep the worker process alive.
    while True:
        time.sleep(10000)
'''
Usage example:
  # Activate the Ray environment and launch one generation worker
  mamba activate ray
  CUDA_VISIBLE_DEVICES=0 python vllm_worker.py --model_path Qwen/Qwen2.5-1.5B-Instruct --tensor_parallel_size 1 --max_num_seqs 64 --global_num_verifiers 10 --overhead_seqs 8 --enable_prefix_caching true
'''

