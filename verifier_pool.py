from copy import deepcopy
import json
import logging
from pathlib import Path
import random
import shutil
import time
import uuid
import asyncio
from functools import partial

import ray
from filelock import FileLock, Timeout
from wrapt_timeout_decorator import timeout
from utils import patch_target_module
from functools import partial
from reward_registry import get_reward_adapter, RewardType
patch_target_module("math_verify.utils.timeout", partial(timeout, use_signals=False))

import numpy as np
logging.getLogger().setLevel(logging.DEBUG)


def cos_fn(t, T, eta_min, eta_max):
    """Basic cosine function component"""
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(t * np.pi / T))


def compute_cosine_reward(gen_length, max_length, format_quality, is_correct, 
                          r_c_0=6, r_c_L=5, r_w_0=-10, r_w_L=0, r_f_0=1.0, r_f_L=0.5,
                          exceed_penalty=-10):
    """
    Modification of the cosine reward function from this paper to include format quality.
    'Demystifying Long Chain-of-Thought Reasoning in LLMs' (Yeo et al., 2025)
    arXiv:2502.03373
      
    Parameters:
    gen_length: Generation length
    max_length: Maximum allowed length
    format_quality: 1=correct format, 0=incorrect format (None uses is_correct only)
    is_correct: 1=correct answer, 0=incorrect answer
    r_c_0/r_c_L: Rewards for correct at min/max length
    r_w_0/r_w_L: Rewards for wrong at min/max length
    r_f_0/r_f_L: Rewards for wrong but good format at min/max length
    exceed_penalty: Penalty for exceeding max length
    """
    # Check if generation length exceeds maximum length
    if gen_length >= max_length:
        return exceed_penalty
    
    if is_correct == 1:
        reward = cos_fn(gen_length, max_length, r_c_0, r_c_L)
    else:
        reward = cos_fn(gen_length, max_length, r_w_0, r_w_L)

    reward += cos_fn(gen_length, max_length, r_f_0, r_f_L) if format_quality == 1 else 0
    
    return reward.item()

@ray.remote
class VerifierWorker:
    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        print(f"Initializing VerifierWorker with id: {worker_id}")

    def verify_task(self, sample: dict, reward_fn_name: str, **kwargs) -> dict:
        """Generic verification entry point that applies the named reward adapter."""
        adapter = get_reward_adapter(reward_fn_name)
        out = adapter(sample, **kwargs)
        sample.update(**out)
        sample['reward_success'] = True
        return sample

@ray.remote
class VerifierPool:
    def __init__(self, global_num_verifiers: int, write_failed: bool = False, reward_fns: list[RewardType] = None, output_dir: str = None):
        self.node_id = ray.get_runtime_context().get_node_id()
        self.global_num_verifiers = global_num_verifiers
        self.write_failed = write_failed
        self.lock = asyncio.Lock()
        # Create an asyncio.Queue to hold available workers.
        self.verifier_queue = asyncio.Queue()
        # Default to both mathd and sympy if not provided
        self.reward_fns = reward_fns or [RewardType.MATHD, RewardType.SYMPY]
        for _ in range(global_num_verifiers):
            self.create_verifier_worker()
        self.outfile = Path(output_dir) / "failed_samples_verify.jsonl" if output_dir is not None else Path("failed_samples_verify.jsonl")
        self.outfile.unlink(missing_ok=True)
        Path(str(self.outfile) + '.lock').unlink(missing_ok=True)
        print(f"\033[38;5;196m\033[1m DEBUG: VerifierPool reward_fns: {self.reward_fns}\033[0m", flush=True)
        
    def create_verifier_worker(self):
        # Create a new worker instance.
        worker = VerifierWorker.options(
            num_cpus=1, 
            scheduling_strategy="SPREAD"
        ).remote(f"verifier_worker_{str(uuid.uuid4())}")
        self.verifier_queue.put_nowait(worker)
        return worker

    async def write_failed_sample(self, sample: dict):
        print("\033[38;5;196m\033[1m DEBUG: Failed to verify sample \033[0m", flush=True)
        if self.write_failed:
            try:
                with FileLock(f"{self.outfile}.lock", timeout=20):
                    with open(self.outfile, "a") as f:
                        f.write(json.dumps(sample) + "\n")
            except Timeout:
                print("Lock acquisition failed after 20 seconds", flush=True)
        return sample
    
    async def _verify_single(self, sample: dict, mode: str, **kwargs) -> dict:
        # Acquire a worker
        worker = await self.verifier_queue.get()
        try:
            # Dispatch to worker
            result_ref = worker.verify_task.remote(sample, mode, **kwargs)
            result = await asyncio.wait_for(result_ref, 30)
            # Return worker to pool
            self.verifier_queue.put_nowait(worker)
            return result
        except Exception as e:
            # Worker failed: kill and replace
            import traceback
            print(traceback.format_exc())
            ray.kill(worker)
            self.create_verifier_worker()
            # Return a default failure sample
            sample['reward'] = 0.0
            sample['reward_success'] = False
            return sample
    
    async def pick_verified_sample(self, results: list[dict]) -> dict:
        """Pick the best result by prioritizing reward, then success flag."""
        return max(
            results,
            key=lambda r: (
                r.get('reward', 0),
                r.get('reward_success', False),
            ),
        )

    async def verify(self, sample: dict, **kwargs) -> dict:
        """Verify using the configured reward functions list."""
        fn_list = kwargs.get('reward_fns', self.reward_fns)
        tasks = [
            asyncio.create_task(
                self._verify_single(deepcopy(sample), fn, **kwargs)
            ) for fn in fn_list
        ]
        results = await asyncio.gather(*tasks)
        # print(f'\033[38;5;196m\033[1m DEBUG: Results: {results[0]["reward"]}\033[0m', flush=True)
        if not any(r.get('reward_success', False) for r in results):
            await self.write_failed_sample(sample)
        return await self.pick_verified_sample(results)


def get_or_create_verifier_pool(global_num_verifiers: int, write_failed: bool = False, reward_fns: list[RewardType] = None, output_dir: str = None) -> VerifierPool:
    # For simplicity, always create a new instance. In a production setting, you might want to implement a singleton.
    try:
        return VerifierPool.options(name="verifier_pool").remote(global_num_verifiers, write_failed, reward_fns, output_dir)
    except Exception as e:
        return ray.get_actor("verifier_pool")
    
if __name__ == '__main__':
    import ray
    from reward_registry import RewardType
    from verifier_pool import get_or_create_verifier_pool

    # Initialize Ray
    ray.init(address="auto", namespace="test")

    # Create a verifier pool with one worker
    pool = get_or_create_verifier_pool(global_num_verifiers=1, write_failed=True, reward_fns=[RewardType.COUNTDOWN], output_dir=None)

    # Define a sample for countdown task
    with open("/new_data/experiments_rh/countdown_grpo_qwen3b/sample.json", "r") as f:
        sample = json.load(f)
    sample['sample_text'] = "<|im_start|>system\nYou are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer.<|im_end|>\n<|im_start|>user\nUsing the numbers [91, 65, 3], create an equation that equals 23. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think> We need to use the numbers 91, 65, and 3 only once to form an equation that equals 23. One way to approach this is to consider the operations and their effects on the numbers. The number 91 is quite large and 65 is also quite large, so subtracting or dividing might not be the best approach. The number 3 seems more manageable. A possible strategy is to think of a way to reduce the value of 91 and 65 to 23 using basic arithmetic. One way to do this is by combining them in a specific way after applying an operation to one of them. </think>\n<answer> (91 - 65) - 3 </answer><|im_end|>"
    sample['end_token'] = "<|im_end|>"

    # Invoke the countdown reward adapter via the verifier pool
    result = ray.get(pool.verify.remote(sample))

    # print('Debugger result:', result)
    from IPython import embed; embed()
    
