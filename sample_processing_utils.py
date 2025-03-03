import random
import numpy as np
import torch
import os
import logging
from numba import njit
from transformers import AutoTokenizer
import torch.distributed as dist
# Configure numba logging
logging.getLogger("numba").setLevel(logging.WARNING)

@njit
def get_output_logits_indices_numba(input_lens, output_lens):
    """
    Compute indices for extracting the output logits from a concatenated logits tensor.
    
    Parameters:
        input_lens (np.ndarray): 1D array of input lengths for each sample.
        output_lens (np.ndarray): 1D array of output lengths for each sample.
        
    For each sample, the logit corresponding to the first output token is at:
        global_index = cumulative_offset + (input_length - 1)
    and there are output_length logits (one per output token).
    
    Returns:
        np.ndarray: 1D array of indices that can be used to select the output logits.
    """
    n_samples = input_lens.shape[0]
    
    # First, compute total number of output indices.
    total_output = 0
    for i in range(n_samples):
        total_output += output_lens[i]
    
    out_indices = np.empty(total_output, dtype=np.int64)
    
    offset = 0    # Cumulative token offset for concatenated samples.
    pos = 0       # Position within out_indices array.
    
    for i in range(n_samples):
        L_in = input_lens[i]
        L_out = output_lens[i]
        # The first output logit's index for this sample is at offset + (L_in - 1)
        start_idx = offset + L_in - 1
        for j in range(L_out):
            out_indices[pos] = start_idx + j
            pos += 1
        # Update offset by the sum of input and output lengths for this sample.
        offset += L_in + L_out
    return out_indices

@njit
def broadcast_values(values, output_lens):
    """
    Broadcast per-sample scalar values to the token level.

    Parameters:
        values : np.ndarray (1D)
            An array of scalar values (e.g., rewards, weights, advantages) for each sample,
            with shape (n_samples,).
        output_lens : np.ndarray (1D of ints)
            An array representing the token counts (output lengths) for each sample,
            with shape (n_samples,).

    Returns:
        np.ndarray (2D)
            An array of broadcasted values, repeated to align with token positions,
            with shape (1, total_tokens), where total_tokens = sum(output_lens).
    """
    # Compute total number of tokens across all samples.
    total_tokens = 0
    num_samples = output_lens.shape[0]
    for i in range(num_samples):
        total_tokens += output_lens[i]
    
    # Initialize the broadcasted values array.
    # broadcasted = np.zeros((1, total_tokens))
    broadcasted = np.zeros(total_tokens)
    
    # Use a running offset to fill in the broadcasted values.
    offset = 0
    for i in range(num_samples):
        length_i = output_lens[i]
        for j in range(length_i):
            broadcasted[offset + j] = values[i]
        offset += length_i

    return broadcasted

def get_output_logits_indices(batched_questions, device):
    input_lens = np.array([s['input_len'] for s in batched_questions])
    output_lens = np.array([s['output_len'] for s in batched_questions])
    output_indices = get_output_logits_indices_numba(input_lens, output_lens)
    output_indices = torch.from_numpy(output_indices).to(device)
    return output_indices, output_lens

def get_input_for_logprobs(batched_questions, output_indices, device):
    batch_ids = torch.cat(
        [torch.tensor(s['sample_ids'], dtype=torch.long) for s in batched_questions]
    ).unsqueeze(0).to(device)
    batch_position_ids = torch.cat(
        [torch.tensor(s['sample_position_ids'], dtype=torch.long) for s in batched_questions]
    ).unsqueeze(0).to(device)
    labels = torch.ones_like(batch_ids) * -100
    labels[:, output_indices+1] = batch_ids[:, output_indices+1]
    return batch_ids, batch_position_ids, labels

def post_process_batch(batched_questions, device):
    output_indices, output_lens = get_output_logits_indices(batched_questions, device)
    if int(os.environ['RANK']) == 0:
        sample = random.choice(batched_questions)
        print(
            f"\033[1;96;40mDecoded Sample:\033[0m {sample['sample_text']}\n" +
            f"\033[1;96;40mReward:\033[0m {sample['reward']}\n" +
            f"\033[1;96;40mGround Truth Answer:\033[0m {sample['gt_answer']}\n" +
            (f"\033[1;96;40mParsed Ground Truth Answer:\033[0m {sample['parsed_gt_answer']}\n" if 'parsed_gt_answer' in sample else "") +
            (f"\033[1;96;40mParsed Attempt: {sample['parsed_attempt']}\033[0m\n" if 'parsed_attempt' in sample else f"\033[1;38;5;196mFailed verification\033[0m")
        )
    advantages = np.array([s['advantage'] for s in batched_questions])
    # print("\033[1;91;40mDEBUG using sample lens (not outputlens to broadcast)\033[0m")
    sample_lens = np.array([s['input_len'] +s['output_len'] for s in batched_questions])
    advantages = torch.from_numpy(broadcast_values(advantages, sample_lens)).to(device).to(torch.float32)
    output_lens_broadcasted = torch.from_numpy(broadcast_values(output_lens, sample_lens)).to(device).to(torch.float32)
    # if any(s['sample_logprobs'] is None for s in batched_questions)\
    #       or any(torch.tensor(s['sample_logprobs']).ndim == 0 for s in batched_questions):
    #     torch.distributed.breakpoint(dist.get_rank())
    reference_output_logprobs = torch.cat(
        [torch.tensor(s['sample_logprobs']) for s in batched_questions]
    ).to(device).to(torch.float32)

    batch_ids, batch_position_ids, labels = get_input_for_logprobs(batched_questions, output_indices, device)
    # output_mask = torch.zeros_like(batch_ids, dtype=torch.float32, device=device)
    # output_mask[:,output_indices+1] = 1
    # # Debug: Assert that the output mask correctly selects the output token ids
    # # Compute the output tokens selected by the mask from batch_ids.
    # # Note: batch_ids has shape [1, total_length] and output_mask is of the same shape.
    # output_tokens_from_mask = batch_ids[0][output_mask[0].bool()]
    # # Build the expected output tokens by concatenating the output portions from each sample.
    # expected_output_tokens = torch.cat([torch.tensor(s['output_token_ids'], device=device) for s in batched_questions]).cuda()

    # # Here we assume that each sample's output tokens are given by sample_ids[input_len:].
    # expected_output_tokens = []
    # for sample in batched_questions:
    #     sample_ids_tensor = torch.tensor(sample['sample_ids'], device=device)
    #     expected_output_tokens.append(sample_ids_tensor[sample['input_len']:])
    # expected_output_tokens = torch.cat(expected_output_tokens)

    # # Perform the assertion to ensure the mask correctly maps to the output tokens.
    # assert torch.equal(output_tokens_from_mask, expected_output_tokens), (
    #     f"Output mask mapping error: expected output tokens {expected_output_tokens} "
    #     f"but got {output_tokens_from_mask}"
    # )
    # torch.distributed.barrier()
    # torch.distributed.breakpoint()
    # torch.distributed.barrier()
    # batch_ids[:,-1000:]
    # torch.distributed.barrier()
    # torch.distributed.breakpoint()
    # torch.distributed.barrier()
    print(f"\033[1;32mYielding batch of length {output_lens.sum()} Rank: {os.environ['LOCAL_RANK']}\033[0m")
    return {
        "batch_ids": batch_ids.contiguous(),
        "batch_position_ids": batch_position_ids.contiguous(),
        "output_indices": output_indices.contiguous(),
        "advantages": advantages.contiguous(),
        "reference_output_logprobs": reference_output_logprobs.contiguous(),
        "output_lens_broadcasted": output_lens_broadcasted.contiguous(),
        "num_output_tokens": torch.tensor(output_lens.sum(), device=device, dtype=torch.float32),
        "num_samples": torch.tensor(len(batched_questions), device=device, dtype=torch.float32),
        # "output_mask": output_mask[:,1:].contiguous(),
        "reward_format": torch.tensor(sum([s['reward_format'] for s in batched_questions]), device=device, dtype=torch.float32),
        "reward_equation": torch.tensor(sum([s['reward_equation'] for s in batched_questions]), device=device, dtype=torch.float32),
        "samples": batched_questions,
        "labels": labels,
        "total_reward_rank": torch.tensor(sum([s['reward'] for s in batched_questions]), device=device, dtype=torch.float32),
    } 