import argparse
import asyncio
import os
from pathlib import Path
import time

import torch
import ray
from accelerate import Accelerator
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoTokenizer
from setup_model import setup_model, setup_training_components
from grpo_loss import compute_grpo_loss
from utils import init_distributed_environment, log_rank_0, setup_logger
from sample_processing_utils import post_process_batch


class JsonlDataset(Dataset):
    def __init__(self, path: str = "/new_data/aldo/v1_reasoning/math_simplerl_qwen_data_token_ids.jsonl"):
        # The fixed token sequence to be returned for every sample.
        # self.sequence = [
        #     100264, 882, 100266, 4438, 1053, 499, 12849, 279, 8286, 315,
        #     264, 6211, 1903, 315, 279, 11552, 315, 1403, 66818, 315,
        #     279, 1890, 10801, 1405, 279, 19169, 72359, 449, 279, 7479,
        #     315, 279, 1023, 26436, 30, 100265, 100264, 78191, 100266
        # ]
        # self.length = length
        self.dataset = load_dataset("json", data_files=path, split="train")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        # Ignore the index and return a fresh copy of the sequence tensor.
        return self.dataset[index]
    

class InfiniteDistributedSampler(Sampler):
    """
    An infinite data sampler that produces a new random permutation of dataset indices
    each epoch (or cycle) and splits the permutation among the different distributed ranks.
    This ensures that in a DDP setting each process gets a different subset of samples.
    """
    def __init__(self, data_source, seed=42):
        self.data_source = data_source
        self.seed = seed
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def __iter__(self):
        epoch = 0
        while True:
            # Use a seed that changes every epoch so that you get a new permutation each time.
            g = torch.Generator()
            g.manual_seed(self.seed + epoch)
            indices = torch.randperm(len(self.data_source), generator=g).tolist()
            
            # Drop any extra indices that don't divide evenly among ranks
            indices = indices[:(len(indices) - len(indices) % self.world_size)]
            
            # Each rank gets every world_size-th index starting from its rank.
            indices = indices[self.rank::self.world_size]
            yield from indices
            epoch += 1

    def __len__(self):
        return len(self.data_source) // self.world_size

def get_dataloader(global_batch_size: int, path: str = "/new_data/aldo/v1_reasoning/math_simplerl_qwen_data_token_ids.jsonl"):
    dataset = JsonlDataset(path=path)
    # Compute per-device local batch size based on the global batch size and world size.
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_batch_size = global_batch_size // world_size
    if rank < (global_batch_size % world_size):
        local_batch_size += 1

    sampler = InfiniteDistributedSampler(dataset)
    return DataLoader(dataset, batch_size=local_batch_size, sampler=sampler, num_workers=4, collate_fn=lambda batch: batch)

def update_vllm_worker_weights(model, accelerator, registry_actor_names=["reference_model_registry", "actor_model_registry"]):
    """
    Update the weights on all vLLM actors using the state dict obtained from the model.
    
    Args:
        model: The model whose weights should be updated on the remote actors.
        accelerator: The Accelerator instance (with accelerator.is_main_process).
        registry_actor_names: The names of the registries to update (the reference model and/or the actor models)
    
    Returns:
        The list of results from the update operations if on the main process; otherwise, None.
    """
    # Retrieve the state dict from the model.
    # log_rank_0(f"\033[1;32mStarting to update weights on {registry_actor_names}\033[0m")
    print(f"\033[1;32mStarting to update weights on {registry_actor_names} Rank: {accelerator.process_index}\033[0m")
    start = time.time()
    state_dict = accelerator.get_state_dict(model)
    
    # Only the main process performs the update.
    if accelerator.is_main_process:
        # Use ray.put to upload the state dict to the Ray object store.
        state_ref = ray.put(state_dict)
        tasks = []
        for registry_actor_name in registry_actor_names:
            # Get the registry actor which maintains the inference actors.
            registry = ray.get_actor(registry_actor_name)
            replica_handles = ray.get(registry.get_actors.remote())
            tasks.extend([handle.update_weights.remote(new_state_dict=state_ref)
                        for handle in replica_handles])
        ray.get(tasks)
        print(f"\033[1;32mUpdated weights on {registry_actor_names} in {time.time() - start:.2f} seconds\033[0m")

    torch.distributed.barrier()
    torch.cuda.empty_cache()

def save_model(args, model, accelerator, samples_seen):
    log_rank_0(f"Saving model at {samples_seen} samples")
    start = time.time()
    output_dir = Path(args.output_dir) / "hf_format" / f"samples_{samples_seen}"
    accelerator.save_model(model,
                            str(output_dir),
                            max_shard_size="20GB",
    )
    if accelerator.is_main_process:
        model.module.config.to_json_file(str(output_dir / "config.json"))
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        tokenizer.save_pretrained(output_dir)
        log_rank_0(f"\033[1;38;2;0;255;255mSaved model at\033[0m {samples_seen} samples in {time.time() - start:.2f} seconds")

async def remote_queue_batch_generator(global_rank: int, 
                                       device: torch.device,
                                       batcher_actor_name: str = "experience_batcher"):
    batcher_actor = ray.get_actor(batcher_actor_name)
    while True:
        batch = await batcher_actor.get_batch.remote(global_rank)
        if batch is None:
            break
        yield post_process_batch(batch, device)

def scale_model_gradients(model, total_samples_in_batch):
    """
    Scale gradients for every parameter in the model by world_size/total_samples_in_batch.
    It's necessary to scale by world_size because fsdp takes the mean of the gradients across the world_size.
    
    Args:
        model: The torch model whose gradients should be scaled.
        total_samples_in_batch: The number of samples in the batch.
    """
    scale_factor = 1.0 / total_samples_in_batch
    for param in model.parameters():
        if param.grad is not None:
            param.grad.mul_(scale_factor)

def take_gradient_step(model, optimizer, lr_scheduler, accelerator, total_samples_accumulated):
    """Scales gradients, applies clipping, and takes an optimization step."""
    scale_model_gradients(model, total_samples_accumulated)
    grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
    print(f"\033[1;38;2;255;165;0mGlobal Grad Norm:\033[0m {grad_norm} \033[1;38;2;255;165;0mRank:\033[0m {accelerator.process_index}")
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

async def train(args,
                policy_model, 
                optimizer,
                lr_scheduler,
                samples_per_question, 
                kl_coeff,
                accelerator: Accelerator,
                num_iterations,
                num_batches_per_ref_model_update,
                ):
    """
    Main training loop following Algorithm 1 from the paper.
    Simplified version since with μ=1, π_old = π_current during sampling.
    """
    # update_vllm_worker_weights(policy_model, accelerator, registry_actor_names=["generation_vllm_registry"])
    model.train()
    dataloader = iter(get_dataloader(args.batch_size, path=args.data_path))
    
    device = accelerator.device
    world_size = int(os.environ["WORLD_SIZE"])

    batcher_actor = ray.get_actor(args.experience_batcher_name, namespace="test")
    total_samples_accumulated = 0
    last_saved_samples = 0

    # actor_registry = ray.get_actor("generation_vllm_registry")
    # reference_registry = ray.get_actor("logprob_vllm_registry")
    
    # Outermost loop: Policy iteration
    for iteration in range(num_iterations):
        log_rank_0(f"Starting iteration {iteration + 1}/{num_iterations}")

        for step in range(num_batches_per_ref_model_update):
            batch = next(dataloader)
            await batcher_actor.generate_experience.remote(
                batch,
                samples_per_question,
                actor_registry="generation_vllm_registry",
                reference_registry="logprob_vllm_registry",
                temperature=args.temperature,
                timeout=1200 # 20 minutes per batch of questions or skipped. --> adjust depending on settings.
            )
            torch.distributed.barrier()
            if accelerator.is_main_process:
                ray.get(batcher_actor.start_creating_batches.remote())
            torch.distributed.barrier()
            samples_in_batch = 0
            async for minibatch in remote_queue_batch_generator(args.global_rank, 
                                                                device,
                                                                batcher_actor_name=args.experience_batcher_name):
                loss, loss_rank, pg_loss, kl_div = compute_grpo_loss(
                    policy_model,
                    minibatch,
                    kl_coeff,
                )

                # Multiply the loss by the number of GPUs to account for FSDP's mean reduction.
                # Gradient scaling divides by the total number of samples in the batch across all GPUs.
                loss *= int(os.environ["WORLD_SIZE"])

                total_num_output_tokens, total_num_samples, total_reward = map(
                    lambda x: x.item(),
                    accelerator.reduce(
                        torch.tensor([
                            minibatch["num_output_tokens"],
                            minibatch["num_samples"],
                            minibatch["total_reward_rank"]
                        ], device=accelerator.device),
                        reduction="sum"
                    )
                )
                
                print(
                    f"\033[1;38;2;255;165;0mActual Loss:\033[0m {loss.item()} \033[1;38;2;255;165;0mRank:\033[0m {accelerator.process_index}\n"
                    f"\033[1;38;2;255;165;0mLoss Rank:\033[0m {loss_rank} \033[1;38;2;255;165;0mRank:\033[0m {accelerator.process_index}\n"
                    f"\033[1;38;2;255;165;0mPG Loss:\033[0m {pg_loss} \033[1;38;2;255;165;0mRank:\033[0m {accelerator.process_index}\n"
                    f"\033[1;38;2;255;165;0mKL Div:\033[0m {kl_div} \033[1;38;2;255;165;0mRank:\033[0m {accelerator.process_index}\n"
                )
                if accelerator.is_main_process:
                    print(
                        f"\033[1;38;2;255;255;0mAverage Total Output Tokens:\033[0m {total_num_output_tokens/total_num_samples} \033[1;38;2;255;255;0mRank:\033[0m {accelerator.process_index}\n"
                        f"\033[1;38;2;255;255;0mAverage Reward:\033[0m {total_reward/total_num_samples} \033[1;38;2;255;255;0mRank:\033[0m {accelerator.process_index}\n"
                    )
                accelerator.backward(loss)

                total_samples_accumulated += total_num_samples
                samples_in_batch += total_num_samples

                torch.cuda.empty_cache()
            # Always take a gradient step before updating vLLM workers
            take_gradient_step(model, optimizer, lr_scheduler, accelerator, samples_in_batch)

            if total_samples_accumulated >= (args.min_samples_per_checkpoint + last_saved_samples):
                save_model(args, model, accelerator, total_samples_accumulated)
                last_saved_samples = total_samples_accumulated
            
            # torch.distributed.breakpoint()
            update_vllm_worker_weights(policy_model, accelerator, registry_actor_names=["generation_vllm_registry"])
        
        update_vllm_worker_weights(policy_model, accelerator, registry_actor_names=["logprob_vllm_registry"])
            


if __name__ == "__main__":
    setup_logger()
    parser = argparse.ArgumentParser()

    # Model and Tokenizer
    parser.add_argument(
        "--model_name_or_path",
        default="/dev/shm/qwen7b-math-base",
        # default="Qwen/Qwen2.5-Math-7B",
        # default="/dev/shm/phi-4",
        type=str,
        # required=True,
        help="Path to pre-trained model or identifier from huggingface.co/models."
    )

    # Training Parameters
    parser.add_argument(
        "--learning_rate",
        default=5e-7,
        type=float,
        # required=True,
        help="Learning rate for training."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128, #TODO: change to 32 for a real experiment
        help="Global batch size of questions per gradient step. The batch will be split among GPUs even if not divisible by the number of GPUs."
    )

    # Scheduler and Optimization
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup", # see transformers.trainer_utils.SchedulerType
        help="Type of learning rate scheduler to use."
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=40,
        help="Number of warmup steps for the scheduler."
    )

    parser.add_argument(
        "--fsdp_sharding_strategy",
        type=str,
        default="FULL_SHARD",
        choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"],
        help="Sharding strategy for Fully Sharded Data Parallel."
    )

    parser.add_argument(
        "--experience_batcher_name",
        type=str,
        default="experience_batcher",
        help="Name of the experience batcher actor."
    )

    parser.add_argument(
        "--max_tokens_per_gpu",
        type=int,
        default=26000,
        help="Maximum number of tokens per GPU."
    )

    # Added new argument for temperature with a default value of 1.0
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for generating experience."
    )

    parser.add_argument(
        "--data_path",
        type=str,
        # default="/new_data/aldo/v1_reasoning/grpobk/limo_data_cleaned_phi_4_format.jsonl",
        default="/new_data/aldo/v1_reasoning/math_simplerl_qwen_data_token_ids.jsonl",
        help="Path to the data file."
    )

    parser.add_argument(
        "--min_samples_per_checkpoint",
        type=int,
        default=64000,
        help="Minimum number of samples per checkpoint."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/new_data/experiments_rh/simple_r1_replica_v2",
        help="Output directory where model checkpoints and configuration files will be saved."
    )

    args = parser.parse_args()
    init_distributed_environment(args)

    model = setup_model(args)
    model, accelerator, optimizer, lr_scheduler = setup_training_components(args, model)

    asyncio.run(
        train(
            args,
            model, 
            optimizer,
            lr_scheduler,
            samples_per_question=64, 
            kl_coeff=0.04,
            accelerator=accelerator,
            num_iterations=100000,
            num_batches_per_ref_model_update=10,
        )
    )

'''
# torchrun --nproc_per_node=8 --master_port=12345 trainer_core.py
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=12345 trainer_core.py 2>&1 | tee train_qwen.log
'''