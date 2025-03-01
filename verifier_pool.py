from copy import deepcopy
import json
import logging
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
patch_target_module("math_verify.utils.timeout", partial(timeout, use_signals=False))
# Import verification functions from math_verify
from math_verify import verify, parse
from math_verify.parser import LatexExtractionConfig, NormalizationConfig
import re

def parse_last_boxed(generation: str) -> str:
    pattern = r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}'
    matches = re.findall(pattern, generation)
    if matches:
        generation = matches[-1]
        boxed = f"\\boxed{{{generation}}}"
        return generation, boxed
    return [], []

# def verify(generated, gt) -> bool:
#     generated, boxed = pre_parse(generated)
#     try:
#         score_1 = hf_verify(parse(generated), parse(gt))
#     except Exception as e:
#         score_1 = 0.0
#     try:
#         score_2 = hf_verify(parse(boxed), parse(gt))
#     except Exception as e:
#         score_2 = 0.0

#     if score_1 or score_2:
#         return 1.0
#     return 0.0


# def verify_sample(sample: dict) -> float:
#     parsing_func_ = partial(
#         parse,
#         extraction_config=[
#             LatexExtractionConfig(
#                 try_extract_without_anchor=False,
#                 boxed_match_priority=0, 
#                 normalization_config=NormalizationConfig(
#                     boxed="last"
#                 )
#             )
#         ],
#         fallback_mode="no_fallback",
#         extraction_mode="first_match",
#         parsing_timeout=1000,
#     )
#     attempt_answer, attempt_boxed = parse_last_boxed(sample['sample_text'])
#     gt_answer, gt_boxed = parse_last_boxed(r'\boxed{' + sample['gt_answer'] + '}')
#     # parsed_gt_answer = parsing_func_(r'\boxed{' + sample['gt_answer'] + '}')
#     # parsed_attempt = parsing_func_(sample['sample_text'])
#     try:
#         result_raw = float(verify(parse(attempt_answer), parse(gt_answer), timeout_seconds=10))
#     except Exception as e:
#         result_raw = 0.0
#     try:
#         result_boxed = float(verify(parse(attempt_boxed), parse(gt_boxed), timeout_seconds=10))
#     except Exception as e:
#         result_boxed = 0.0
#     sample['reward'] = max(result_raw, result_boxed)
#     return sample

def verify_sample_format(sample: dict) -> float:
    completion = sample['sample_text']
    try:
    # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
        completion = completion.split("Let me solve this step by step.\n")[1].split("<|im_end|>")[0]
        
        # Check if the format is correct
        regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"

        match = re.search(regex, completion, re.DOTALL) 
        # if the format is not correct, reward is 0
        if match is None or len(match.groups()) != 2:
            reward = 0.0
        else:
            reward = 1.0
    except Exception:
        reward = 0.0

    sample['reward_format'] = reward
    return sample


def verify_sample_equation(sample: dict) -> float:
    completion = sample['sample_text']
    gt = sample['gt_answer']
    numbers = sample['nums']
    try:
        # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
        completion = completion.split("Let me solve this step by step.\n")[1].split("<|im_end|>")[0]
        # Check if the format is correct
        match = re.search(r"<answer>(.*?)<\/answer>", completion)
        if match is None:
            reward = 0.0
        # Extract the "answer" part from the completion
        equation = match.group(1).strip()
        # Extract all numbers from the equation
        used_numbers = [int(n) for n in re.findall(r'\d+', equation)]
        
        # Check if all numbers are used exactly once
        if sorted(used_numbers) != sorted(numbers):
            reward = 0.0
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation):
           reward = 0.0        
        # Evaluate the equation with restricted globals and locals
        result = eval(equation, {"__builtins__": None}, {})
        # Check if the equation is correct and matches the ground truth
        if abs(float(result) - float(gt)) < 1e-5:
            reward = 1.0
        else:
            reward = 0.0
    except Exception:
        reward = 0.0

    sample['reward_equation'] = reward
    return sample


@ray.remote
class VerifierWorker:
    def __init__(self, worker_id: str, write_failed: bool = False):
        self.worker_id = worker_id
        print(f"Initializing VerifierWorker with id: {worker_id}")
    
    def verify_sample_format(self, sample: dict):
        return verify_sample_format(sample)
    
    def verify_sample_equation(self, sample: dict):
        return verify_sample_equation(sample)

@ray.remote
class VerifierPool:
    def __init__(self, global_num_verifiers: int, write_failed: bool = False):
        self.node_id = ray.get_runtime_context().get_node_id()
        self.global_num_verifiers = global_num_verifiers
        self.write_failed = write_failed
        self.verifier_pool = [None for _ in range(global_num_verifiers)]
        self.verifier_load = [0 for _ in range(global_num_verifiers)]
        self.lock = asyncio.Lock()
        self.create_verifier_tasks = [asyncio.create_task(self.create_verifier(i)) for i in range(global_num_verifiers)]
        shutil.rmtree(f"failed_samples_verify.jsonl", ignore_errors=True)
        shutil.rmtree(f"failed_samples_verify.jsonl.lock", ignore_errors=True)

    async def create_verifier(self, index: int):
        async with self.lock:
            self.verifier_pool[index] = VerifierWorker.options(
                num_cpus=1, 
                scheduling_strategy="SPREAD",
            ).remote(f"verifier_{index}_{str(uuid.uuid4())}", self.write_failed)
            self.verifier_load[index] = 0

    async def write_failed_sample(self, sample: dict):
        print("\033[38;5;196m\033[1m DEBUG: Failed to verify sample \033[0m", flush=True)
        if self.write_failed:
            try:
                with FileLock(f"failed_samples_verif.jsonl.lock", timeout=20):
                    with open(f"failed_samples_verify.jsonl", "a") as f:
                        f.write(json.dumps(sample) + "\n")
            except Timeout:
                print("Lock acquisition failed after 20 seconds", flush=True)
        return sample
    
    async def _verify_balanced(self, sample: dict, mode: str) -> dict:
        result = deepcopy(sample)
        result[f'reward_{mode}'] = 0.0
        for _ in range(2):
            try:
                async with self.lock:
                    min_index = min(range(len(self.verifier_load)), key=lambda i: self.verifier_load[i])
                    self.verifier_load[min_index] += 1
                if mode == 'format':
                    result_ref = self.verifier_pool[min_index].verify_sample_format.remote(sample)
                elif mode == 'equation':
                    result_ref = self.verifier_pool[min_index].verify_sample_equation.remote(sample)
                result =  await asyncio.wait_for(result_ref, 30)
                async with self.lock:
                    self.verifier_load[min_index] -= 1
                break
            except Exception as e:
                print(f"\033[1;38;5;196mCoroutine died in verify_balanced with mode: {mode}\033[0m", flush=True)
                await self.create_verifier(min_index)
                await self.write_failed_sample(result)
                await asyncio.sleep(random.uniform(0.1, 5))
        return result
 
    async def verify_balanced(self, sample: dict) -> dict:
        format_future = asyncio.create_task(self._verify_balanced(sample, 'format'))
        equation_future = asyncio.create_task(self._verify_balanced(sample, 'equation'))
        format_result = await format_future
        equation_result = await equation_future
        sample['reward_format'] = format_result['reward_format']
        sample['reward_equation'] = equation_result['reward_equation']
        sample['reward'] = format_result['reward_format'] + equation_result['reward_equation']
        return sample


def get_or_create_verifier_pool(global_num_verifiers: int, write_failed: bool = False) -> VerifierPool:
    # For simplicity, always create a new instance. In a production setting, you might want to implement a singleton.
    try:
        return VerifierPool.options(name="verifier_pool").remote(global_num_verifiers, write_failed) 
    except Exception as e:
        return ray.get_actor("verifier_pool")
