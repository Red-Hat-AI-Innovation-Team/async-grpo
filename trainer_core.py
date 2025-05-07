import argparse
import asyncio
import os
from pathlib import Path
import time
from enum import Enum
import json

import torch
import ray
from accelerate import Accelerator
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoTokenizer
from typer import Typer, Option

from setup_model import setup_model, setup_training_components
from grpo_loss import compute_grpo_loss
from utils import init_distributed_environment, log_rank_0, setup_logger, ArgsNamespace
from sample_processing_utils import post_process_batch
from batch_metrics import BatchMetrics
from async_structured_logger import AsyncStructuredLogger


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

def get_dataloader(global_batch_size: int, path: str = "/new_data/aldo/v1_reasoning/math_simplerl_qwen_data_token_ids.jsonl", sampler_seed: int = 37):
    dataset = JsonlDataset(path=path)
    # Compute per-device local batch size based on the global batch size and world size.
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_batch_size = global_batch_size // world_size
    if rank < (global_batch_size % world_size):
        local_batch_size += 1
    sampler = InfiniteDistributedSampler(dataset, seed=sampler_seed)
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
            tasks.append(registry.update_weights.remote(new_state_dict=state_ref))
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
                            safe_serialization=True,
    )
    if accelerator.is_main_process:
        model.module.config.to_json_file(str(output_dir / "config.json"))
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        tokenizer.save_pretrained(output_dir)
        log_rank_0(f"\033[1;38;2;0;255;255mSaved model at\033[0m {samples_seen} samples in {time.time() - start:.2f} seconds")

async def remote_queue_batch_generator(global_rank: int,
                                       device: torch.device,
                                       batcher_actor_name: str = "experience_batcher",
                                       constant_length_samples: int | None = None):
    batcher_actor = ray.get_actor(batcher_actor_name)
    while True:
        batch = await batcher_actor.get_batch.remote(global_rank)
        if batch is None:
            break
        yield post_process_batch(batch, device, constant_length_samples=constant_length_samples)

def scale_model_gradients(model, scale_factor):
    """
    Scale gradients for every parameter in the model by world_size/total_samples_in_batch.
    It's necessary to scale by world_size because fsdp takes the mean of the gradients across the world_size.
    
    Args:
        model: The torch model whose gradients should be scaled.
        total_samples_in_batch: The number of samples in the batch.
    """
    # the more samples per question, 
    # scale_factor = 1.0 / total_samples_in_batch
    for param in model.parameters():
        if param.grad is not None:
            param.grad.mul_(scale_factor)

def take_gradient_step(model, optimizer, lr_scheduler, accelerator, scale_factor):
    """Scales gradients, applies clipping, and takes an optimization step."""
    scale_model_gradients(model, scale_factor)
    grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
    print(f"\033[1;38;2;255;165;0mGlobal Grad Norm:\033[0m {grad_norm} \033[1;38;2;255;165;0mRank:\033[0m {accelerator.process_index}")
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    return grad_norm

async def train(args,
                policy_model, 
                optimizer,
                lr_scheduler,
                accelerator: Accelerator,
                ):
    """
    Main training loop following Algorithm 1 from the paper.
    Simplified version since with μ=1, π_old = π_current during sampling.
    """
    policy_model.train()
    dataloader = iter(get_dataloader(args.batch_size, path=args.data_path, sampler_seed=args.infinite_sampler_seed))
    
    device = accelerator.device
    world_size = int(os.environ["WORLD_SIZE"])

    batcher_actor = ray.get_actor(args.experience_batcher_name, namespace="test")

    if accelerator.is_main_process:
        metric_logger = AsyncStructuredLogger(os.path.join(args.output_dir, f"training_metrics.jsonl"))

    total_samples_accumulated = 0
    last_saved_samples = 0
    batch_totals = BatchMetrics()
    
    # Outermost loop: Policy iteration
    for iteration in range(args.num_iterations):
        log_rank_0(f"Starting iteration {iteration + 1}/{args.num_iterations}")

        for step in range(args.num_batches_per_ref_model_update):
            start_time = time.time()
            batch = next(dataloader)
            torch.distributed.barrier()
            await batcher_actor.generate_experience.remote(
                batch,
                args.samples_per_question,
                actor_registry="generation_vllm_registry",
                reference_registry="logprob_vllm_registry",
                temperature=args.temperature,
                max_tokens=args.max_generation_tokens,
                insert_reasoning_phrases=args.insert_reasoning_phrases,
                timeout=1200 # 20 minutes per batch of questions or skipped. --> adjust depending on settings.
            )
            torch.distributed.barrier()
            if accelerator.is_main_process:
                ray.get(batcher_actor.start_creating_batches.remote())
            torch.distributed.barrier()

            # Initialize a Metrics instance for accumulating minibatch metrics
            batch_totals.reset_batch()
            async for minibatch in remote_queue_batch_generator(args.global_rank,
                                                                device,
                                                                batcher_actor_name=args.experience_batcher_name,
                                                                constant_length_samples=args.constant_length_samples):
                loss, loss_metrics, pg_loss, kl_div, entropy = compute_grpo_loss(
                    policy_model,
                    minibatch,
                    args.kl_coeff,
                )
                # Multiply the loss by the number of GPUs to account for FSDP's mean reduction.
                # Gradient scaling divides by the total number of samples in the batch across all GPUs.
                loss *= int(os.environ["WORLD_SIZE"])
                loss /= int(4000) # scale the loss since we are using a sum instead of a mean and need to scale down the gradients.
                accelerator.backward(loss)
                torch.cuda.empty_cache()

                
                # Accumulate metrics in the Metrics instance
                batch_totals.accumulate_minibatch_metrics(
                    output_tokens = minibatch["num_output_tokens"],
                    non_masked_output_tokens = minibatch["num_output_tokens_non_masked"],
                    samples = minibatch["num_samples"],
                    reward = minibatch["total_reward_rank"],
                    modified_reward = minibatch["total_modified_reward"],
                    modified_samples = minibatch["num_modified_samples"],
                    delimiter_not_found = minibatch["delimiter_not_found"],
                    non_modified_reward = minibatch["total_non_modified_reward"],
                    max_reward_in_group = minibatch["max_reward_in_group"],
                    loss = loss_metrics,
                    pg_loss = pg_loss,
                    kl_div = kl_div,
                    entropy = entropy,
                    advantage_is_zero = minibatch["advantage_is_zero"],
                    truncated_sample = minibatch["truncated_sample"],
                )

            # End async for

            # Reduce minibatch metrics and accumulate into batch_metrics
            batch_totals.reduce_batch_metrics(accelerator)

            # Use accumulated metrics for gradient step and logging
            bm = batch_totals.totals
            batch_num_samples = bm["samples"]
            total_samples_accumulated += batch_num_samples
            # we multiply by 4000 and divide by the number of output tokens, which makes the loss a per-token loss.
            grad_norm = take_gradient_step(policy_model, optimizer, lr_scheduler, accelerator, int(4000)/bm['non_masked_output_tokens'])

            if accelerator.is_main_process:
                batch_time = time.time() - start_time
                metrics_to_log = {
                    "step": step,
                    "iteration": iteration,
                    "total_samples_accumulated": total_samples_accumulated,
                    "samples_in_batch": batch_num_samples,
                    "avg_reward": bm['reward']/batch_num_samples,
                    "avg_output_tokens": bm['output_tokens']/batch_num_samples,
                    "avg_loss": bm['loss']/bm['non_masked_output_tokens'],
                    "lr": lr_scheduler.get_last_lr()[0],
                    "avg_pg_loss": bm['pg_loss']/bm['non_masked_output_tokens'],
                    "avg_kl_div": bm['kl_div']/bm['non_masked_output_tokens'],
                    "avg_modified_reward": bm['modified_reward']/(bm['modified_samples']+1e-6),
                    "num_modified_samples": bm['modified_samples'],
                    "avg_delimiter_not_found": bm['delimiter_not_found']/(bm['modified_samples']+1e-6),
                    "avg_non_modified_reward": bm['non_modified_reward']/(batch_num_samples - bm['modified_samples']+1e-9),
                    "avg_max_reward_in_group": bm['max_reward_in_group']/batch_num_samples,
                    "perc_with_0_advantage": bm['advantage_is_zero']/batch_num_samples,
                    "perc_truncated_samples": bm['truncated_sample']/batch_num_samples,
                    "grad_norm": grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm,
                    "time_per_batch": batch_time,
                    "samples_per_second": batch_num_samples / batch_time,
                    "peak_memory_usage_GB": float(torch.cuda.max_memory_allocated() / 1e9),
                    "entropy": bm['entropy']/bm['output_tokens'],
                }
                metric_logger.log_sync(metrics_to_log)

            if total_samples_accumulated >= (args.min_samples_per_checkpoint + last_saved_samples):
                save_model(args, policy_model, accelerator, total_samples_accumulated)
                last_saved_samples = total_samples_accumulated
            
            #update both logprob and generation workers at the last step of the ref model update loop
            registry_actor_names = ["generation_vllm_registry", "logprob_vllm_registry"] if step == args.num_batches_per_ref_model_update - 1 else ["generation_vllm_registry"]
            update_vllm_worker_weights(policy_model, accelerator, registry_actor_names=registry_actor_names)
        
            

app = Typer(
    pretty_exceptions_show_locals=False,  # Hide local variables in tracebacks
    pretty_exceptions_short=True   
)

class FSDPShardingStrategyEnum(str, Enum):
    NO_SHARD = "NO_SHARD"
    SHARD_GRAD_OP = "SHARD_GRAD_OP"
    FULL_SHARD = "FULL_SHARD"
    HYBRID_SHARD = "HYBRID_SHARD"
    _HYBRID_SHARD_ZERO2 = "_HYBRID_SHARD_ZERO2"

class LogLevelEnum(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@app.command()
def main(
    model_name_or_path: str = Option("/dev/shm/DeepSeek-R1-Distill-Qwen-1.5B", help="Path to pre-trained model or identifier from huggingface.co/models."),
    learning_rate: float = Option(2e-6, help="Learning rate for training."),
    batch_size: int = Option(128, help="Global batch size of questions per gradient step. The batch will be split among GPUs even if not divisible by the number of GPUs."),
    lr_scheduler: str = Option("constant_with_warmup", help="Type of learning rate scheduler to use."),
    num_warmup_steps: int = Option(10, help="Number of warmup steps for the scheduler."),
    fsdp_sharding_strategy: FSDPShardingStrategyEnum = Option(
        FSDPShardingStrategyEnum.SHARD_GRAD_OP, 
        help="Sharding strategy for Fully Sharded Data Parallel.",
        case_sensitive=False
    ),
    experience_batcher_name: str = Option("experience_batcher", help="Name of the experience batcher actor."),
    max_tokens_per_gpu: int = Option(36000, help="Maximum number of tokens per GPU."),
    loss_chunksize: int = Option(None, help="Number of tokens to process at a time for the loss computation. This avoids creating the logits matrix all at once in memory. None means no chunking."),
    temperature: float = Option(0.6, help="Sampling temperature for generating experience."),
    max_generation_tokens: int = Option(8192, help="Maximum number of tokens to generate per rollout."),
    insert_reasoning_phrases: bool = Option(False, "--insert-reasoning-phrases/--no-insert-reasoning-phrases", help="Enable rewriting to insert reasoning phrases during inference."),
    data_path: str = Option("/new_data/aldo/v1_reasoning/grpo_feb_24th/deepscaler_phi_mini_nemotron.jsonl", help="Path to the data file."),
    min_samples_per_checkpoint: int = Option(30000, help="Minimum number of samples per checkpoint."),
    output_dir: str = Option("/new_data/experiments_rh/deepscaler_qwen1.5b_also_single_delimiter", help="Output directory where model checkpoints and configuration files will be saved."),
    infinite_sampler_seed: int = Option(37, help="Seed for InfiniteDistributedSampler, used to shuffle the data loader."),
    samples_per_question: int = Option(32, help="Number of samples per question to use in training."),
    constant_length_samples: int = Option(None, help="If set, forces all samples to be treated as having this output length for broadcasting advantages and other values. Defaults to None (use actual output lengths)."),
    kl_coeff: float = Option(0.001, help="KL coefficient for GRPO loss."),
    num_iterations: int = Option(1000000, help="Total number of training iterations."),
    num_batches_per_ref_model_update: int = Option(40, help="Number of training batches before updating the reference model weights."),
    logging_level: LogLevelEnum = Option(LogLevelEnum.INFO, help="Logging level", case_sensitive=False),
    global_rank: int = Option(int(os.environ.get("RANK", 0)), help="Global rank of the process."), # Add global_rank from env
    use_torch_compile: bool = Option(True, help="Use torch.compile to speed up training."),
):
    """
    Main training entry point for Async GRPO.
    """
    setup_logger(level=logging_level.value)

    args = ArgsNamespace(
        model_name_or_path=model_name_or_path,
        learning_rate=learning_rate,
        batch_size=batch_size,
        lr_scheduler=lr_scheduler,
        num_warmup_steps=num_warmup_steps,
        fsdp_sharding_strategy=fsdp_sharding_strategy.value,
        experience_batcher_name=experience_batcher_name,
        max_tokens_per_gpu=max_tokens_per_gpu,
        loss_chunksize=loss_chunksize,
        temperature=temperature,
        max_generation_tokens=max_generation_tokens,
        insert_reasoning_phrases=insert_reasoning_phrases,
        data_path=data_path,
        min_samples_per_checkpoint=min_samples_per_checkpoint,
        output_dir=output_dir,
        infinite_sampler_seed=infinite_sampler_seed,
        samples_per_question=samples_per_question,
        constant_length_samples=constant_length_samples,
        kl_coeff=kl_coeff,
        num_iterations=num_iterations,
        num_batches_per_ref_model_update=num_batches_per_ref_model_update,
        global_rank=global_rank, # Manually add global_rank
        use_torch_compile=use_torch_compile,
    )
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Log parameters only on rank 0
    if args.global_rank == 0:
        params_to_log = args.__dict__.copy()
        params_to_log.update({
            "logging_level": logging_level.value,
            "WORLD_SIZE": int(os.environ.get("WORLD_SIZE", 1))
        })
        params_path = output_path / f"training_params.json"
        with open(params_path, 'w') as f:
            json.dump(params_to_log, f, indent=4)
        print(f"Training with parameters: {json.dumps(params_to_log, separators=(',', ':'), indent=4)}")
        print(f"Training parameters saved to {params_path}")

    init_distributed_environment(args)
    model = setup_model(args)
    model, accelerator, optimizer, lr_scheduler = setup_training_components(args, model)

    asyncio.run(
        train(
            args,
            model, 
            optimizer,
            lr_scheduler,
            accelerator=accelerator,
        )
    )

if __name__ == "__main__":
    app()

'''
# set -x log_dir /new_data/experiments_rh/deepscaler_qwen1.5b_also_single_delimiter
set -x log_dir /new_data/experiments_rh/deepscaler_no_insert_qwen1.5b_base
     --insert_reasoning_phrases \
set -x log_dir /new_data/experiments_rh/deepscaler_with_inserts_qwen1.5b_base
set -x log_dir /new_data/experiments_rh/deepscaler_no_inserts_qwen1.5b_base_5e-6
set -x log_dir /new_data/experiments_rh/qwen1.5b_limo_s3143_deepscaler_64spq
set -x log_dir /new_data/experiments_rh/testing_vllm_failures
set -x log_dir /new_data/experiments_rh/qwen_base_1.5_deepscaler_128bs_64spq
set -x log_dir /new_data/experiments_rh/qwen_1.5b_r1_distill_deepscaler_test
set -x log_dir /new_data/experiments_rh/qwen_1.5b_r1_distill_deepscaler_v2


set -Ux NCCL_SOCKET_IFNAME eth1
set -Ux NCCL_IB_DISABLE 1
set -x log_dir /new_data/experiments_rh/phi_mini_2499716_deepscaler_128bs_8spq
set -x rank 0
mkdir -p $log_dir
cd /new_data/aldo/v1_reasoning/grpo_feb_24th/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --node_rank=$rank --nproc_per_node=8 --rdzv_id=101 \
    --rdzv_endpoint="10.241.128.19:54367" trainer_core.py \
     --output_dir $log_dir 2>&1 \
    | tee $log_dir/train_$rank.log
# torchrun --nproc_per_node=4 trainer_core.py 2>&1 | tee ~/grpo/train_countdown_3b.log
set -x rank 0
mkdir -p ~/grpo
torchrun --nnodes=1 --node_rank=$rank --nproc_per_node=1 --rdzv_id=101 \
    --rdzv_endpoint="10.241.128.19:54367" trainer_core.py 2>&1 | tee ~/grpo/train_countdown_3b.log
'''