import argparse
import asyncio
import os
from pathlib import Path
import time
from enum import Enum
import json

import torch
import ray
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoTokenizer
from typer import Typer, Option

from setup_model import setup_model, setup_training_components
from grpo_loss import compute_dual_clip_grpo_loss
from utils import get_fsdp_param_l1_stats, init_distributed_environment, log_rank_0, setup_logger, ArgsNamespace
from sample_processing_utils import post_process_batch
from batch_metrics import BatchMetrics
from file_writer_actor import get_or_create_filewriter
from vllm_experience_batcher import MessageType


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

def update_vllm_worker_weights(model, registry_actor_names=["reference_model_registry", "actor_model_registry"]):
    """
    Update the weights on all vLLM actors using the state dict obtained from the model.
    
    Args:
        model: The model whose weights should be updated on the remote actors.
        registry_actor_names: The names of the registries to update (the reference model and/or the actor models)
    
    Returns:
        The list of results from the update operations if on the main process; otherwise, None.
    """
    rank = torch.distributed.get_rank()
    print(f"\033[1;32mStarting to update weights on {registry_actor_names} Rank: {rank}\033[0m")
    torch.distributed.barrier()
    start = time.time()
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
    state_dict = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    
    # Only the main process performs the update.
    if rank == 0:
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

    print(f"\033[1;38;2;0;191;255mUpdated weights on {registry_actor_names} in {time.time() - start:.2f} seconds Rank: {rank}\033[0m")
    torch.distributed.barrier()
    torch.cuda.empty_cache()

def save_model(fsdp_model, samples_seen, output_dir, model_name_or_path):
    from huggingface_hub import split_torch_state_dict_into_shards
    from transformers import AutoTokenizer
    from safetensors.torch import save_file
    # Only on rank 0
    log_rank_0(f"Saving model at {samples_seen} samples")
    start = time.time()
    rank = torch.distributed.get_rank()
    save_directory = Path(output_dir) / "hf_format" / f"samples_{samples_seen}"
    os.makedirs(save_directory, exist_ok=True)
    # Get full state dict
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
    state_dict = get_model_state_dict(fsdp_model, options=StateDictOptions(full_state_dict=True))
    state_dict = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
    if rank == 0:
        pattern = "model{suffix}.safetensors"
        index_name = "model.safetensors.index.json"
        # Shard splitting
        split = split_torch_state_dict_into_shards(
            state_dict, filename_pattern=pattern, max_shard_size="5GB",
        )
        # Save shards
        for filename, tensors in split.filename_to_tensors.items():
            shard = {k: state_dict[k] for k in tensors}
            path = os.path.join(save_directory, filename)
            save_file(shard, path)
        # Save index if sharded
        if split.is_sharded:
            index = {"metadata": split.metadata, "weight_map": split.tensor_to_filename}
            with open(os.path.join(save_directory, index_name), "w") as f:
                json.dump(index, f, indent=2, sort_keys=True)
        # Save config and tokenizer (unwrap inner module)
        inner = getattr(fsdp_model, "module", fsdp_model)
        inner.config.to_json_file(os.path.join(save_directory, "config.json"))
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        tokenizer.save_pretrained(save_directory)
        log_rank_0(f"\033[1;38;2;0;255;255mSaved model at\033[0m {samples_seen} samples in {time.time() - start:.2f} seconds")
    torch.distributed.barrier()

async def remote_event_generator(global_rank: int,
                                       device: torch.device,
                                       batcher_actor_name: str = "experience_batcher",
                                       constant_length_samples: int | None = None):
    """
    Yield raw Message objects from the batcher, stopping on BATCH_DONE.
    """
    batcher_actor = ray.get_actor(batcher_actor_name)
    while True:
        msg = await batcher_actor.get_batch.remote(global_rank)
        yield msg
        if msg.type == MessageType.BATCH_DONE:
            break


def take_gradient_step(model, optimizer, lr_scheduler):
    """Scales gradients, applies clipping, and takes an optimization step."""
    rank = torch.distributed.get_rank()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    torch.distributed.barrier()
    grad_norm = grad_norm.full_tensor()
    print(f"\033[1;38;2;255;165;0mGlobal Grad Norm:\033[0m {grad_norm} \033[1;38;2;255;165;0mRank:\033[0m {rank}")
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    return grad_norm

async def train(args,
                policy_model, 
                optimizer,
                lr_scheduler):
    """
    Main training loop following Algorithm 1 from the paper.
    Simplified version since with μ=1, π_old = π_current during sampling.
    """
    policy_model.train()
    rank = torch.distributed.get_rank()
    is_main_process = torch.distributed.get_rank() == 0
    dataloader = iter(get_dataloader(args.batch_size, path=args.data_path, sampler_seed=args.infinite_sampler_seed))
    
    device = next(policy_model.parameters()).device
    world_size = int(os.environ["WORLD_SIZE"])

    batcher_actor = ray.get_actor(args.experience_batcher_name, namespace="test")

    # Set up loggers and dumpers
    metric_logger = None
    if is_main_process:
        metrics_file = os.path.join(args.output_dir, "training_metrics.jsonl")
        metric_logger = get_or_create_filewriter("metrics_writer", metrics_file)
    # Set up sample dumper actor if requested (submits across machines)
    dumper = None
    if args.dump_samples_filename:
        samples_file = os.path.join(args.output_dir, args.dump_samples_filename)
        dumper = get_or_create_filewriter("samples_writer", samples_file)

    total_samples_accumulated = 0
    last_saved_samples = 0
    batch_totals = BatchMetrics()

    get_fsdp_param_l1_stats(policy_model, initialize_baseline=True)
    
    # Outermost loop: Policy iteration
    for iteration in range(args.num_training_batches):
        log_rank_0(f"Starting iteration {iteration + 1}/{args.num_training_batches}")

        # process one batch per iteration
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
        if is_main_process:
            ray.get(batcher_actor.start_creating_batches.remote())
        torch.distributed.barrier()

        event_start_time = time.time()
        # Count of microbatches processed in this policy iteration
        num_microbatches_in_minibatch = 0
        total_non_masked_tokens = 0
        async for msg in remote_event_generator(
            args.global_rank,
            device,
            batcher_actor_name=args.experience_batcher_name,
            constant_length_samples=args.constant_length_samples,
        ):
            if msg.type is MessageType.MINIBATCH:
                if dumper:
                    for sample in msg.data:
                        dumper.append.remote(sample, timestamp=False, print_console=False)
                mb = post_process_batch(msg.data, device, constant_length_samples=args.constant_length_samples)
                # Compute dual-clip PPO loss and associated metrics
                loss, metrics = compute_dual_clip_grpo_loss(
                    policy_model,
                    mb,
                    args.clip_low,
                    args.clip_high,
                    args.clip_ratio_c,
                )

                # Scale loss by world size and normalize by total non-masked output tokens
                total_non_masked_tokens = mb["samples"][0]["total_non_masked_output_tokens"]
                loss *= world_size / (total_non_masked_tokens + 2e-6)
                loss.backward()
                torch.cuda.empty_cache()

                batch_totals.accumulate_minibatch_metrics(
                    output_tokens=mb["num_output_tokens"],
                    non_masked_output_tokens=mb["num_output_tokens_non_masked"],
                    samples=mb["num_samples"],
                    reward=mb["total_reward_rank"],
                    modified_reward=mb["total_modified_reward"],
                    modified_samples=mb["num_modified_samples"],
                    delimiter_not_found=mb["delimiter_not_found"],
                    non_modified_reward=mb["total_non_modified_reward"],
                    max_reward_in_group=mb["max_reward_in_group"],
                    loss=metrics["loss"],
                    backward_loss=loss.detach().item(),
                    kl_div=metrics["kl_div"],
                    entropy=metrics["entropy"],
                    pg_clip=metrics["pg_clip"],
                    advantage_is_zero=mb["advantage_is_zero"],
                    truncated_sample=mb["truncated_sample"],
                    maj_ratio=mb["maj_ratio"],
                    true_reward=mb["true_reward"],
                )
                num_microbatches_in_minibatch += 1
                print(f"\033[1;38;2;255;165;0mNum microbatches in minibatch:\033[0m {num_microbatches_in_minibatch} \033[1;38;2;255;165;0mRank:\033[0m {rank}")
            elif msg.type is MessageType.GRADIENT_STEP:
                # reduce metrics, take gradient step
                batch_totals.reduce_batch_metrics(device)
                bm = batch_totals.totals
                total_samples_accumulated += bm["samples"]

                grad_norm = take_gradient_step(
                    policy_model, optimizer, lr_scheduler,
                )

                model_norm, model_delta_norm, update_ratio_min, update_ratio_p25, update_ratio_p50, update_ratio_p75, update_ratio_max = get_fsdp_param_l1_stats(policy_model)

                if is_main_process:
                    # compute timing since event start
                    batch_time = time.time() - event_start_time
                    event_start_time = time.time()
                    # original logging fields, now per step
                    metrics_to_log = {
                        "iteration": iteration,
                        "total_samples_accumulated": total_samples_accumulated,
                        "samples_in_batch": bm["samples"],
                        "avg_reward": bm["reward"] / bm["samples"],
                        "avg_true_reward": bm["true_reward"] / bm["samples"],
                        "avg_maj_ratio": bm["maj_ratio"] / bm["samples"],
                        "avg_output_tokens": bm["output_tokens"] / bm["samples"],
                        "total_non_masked_tokens": total_non_masked_tokens,
                        "num_non_masked_tokens": bm["non_masked_output_tokens"],
                        "avg_loss": bm["loss"] / world_size / num_microbatches_in_minibatch,
                        "lr": lr_scheduler.get_last_lr()[0],
                        "backward_loss": bm["backward_loss"]/world_size/num_microbatches_in_minibatch,
                        "avg_pg_clip": bm["pg_clip"] / bm["non_masked_output_tokens"],
                        "avg_kl_div": bm["kl_div"] / bm["non_masked_output_tokens"],
                        "avg_modified_reward": bm["modified_reward"] / (bm["modified_samples"] + 1e-6),
                        "num_modified_samples": bm["modified_samples"],
                        "avg_delimiter_not_found": bm["delimiter_not_found"] / (bm["modified_samples"] + 1e-6),
                        "avg_non_modified_reward": bm["non_modified_reward"] / (bm["samples"] - bm["modified_samples"] + 1e-9),
                        "avg_max_reward_in_group": bm["max_reward_in_group"] / bm["samples"],
                        "perc_with_0_advantage": bm["advantage_is_zero"] / bm["samples"],
                        "perc_truncated_samples": bm["truncated_sample"] / bm["samples"],
                        "grad_norm": grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm,
                        "time_per_batch": batch_time,
                        "samples_per_second": bm["samples"] / batch_time,
                        "peak_memory_usage_GB": float(torch.cuda.max_memory_allocated() / 1e9),
                        "entropy": bm["entropy"] / bm["output_tokens"],
                        "batch_done": total_samples_accumulated % (args.batch_size * args.samples_per_question) == 0,
                        "model_norm": model_norm,
                        "model_delta_norm": model_delta_norm,
                        "update_ratio_min": update_ratio_min,
                        "update_ratio_p25": update_ratio_p25,
                        "update_ratio_p50": update_ratio_p50,
                        "update_ratio_p75": update_ratio_p75,
                        "update_ratio_max": update_ratio_max,
                    }
                    if metric_logger:
                        metric_logger.append.remote(metrics_to_log, timestamp=True, print_console=True)
                    batch_totals.reset_batch()
                    torch.cuda.empty_cache()
                    num_microbatches_in_minibatch = 0

            elif msg.type is MessageType.BATCH_DONE:
                print(f"\033[1;38;2;255;165;0mBatch Done:\033[0m {total_samples_accumulated} \033[1;38;2;255;165;0mRank:\033[0m {rank}")
                # checkpoint if threshold reached
                if total_samples_accumulated >= (args.min_samples_per_checkpoint + last_saved_samples):
                    save_model(policy_model, total_samples_accumulated, args.output_dir, args.model_name_or_path)
                    last_saved_samples = total_samples_accumulated
                # update both generation and reference logprobs every batch
                update_vllm_worker_weights(
                    policy_model,
                    registry_actor_names=["generation_vllm_registry", "logprob_vllm_registry"]
                )


app = Typer(
    pretty_exceptions_show_locals=False,  # Hide local variables in tracebacks
    pretty_exceptions_short=True   
)

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
    experience_batcher_name: str = Option("experience_batcher", help="Name of the experience batcher actor."),
    max_tokens_per_gpu: int = Option(36000, help="Maximum number of tokens per GPU."),
    temperature: float = Option(0.6, help="Sampling temperature for generating experience."),
    max_generation_tokens: int = Option(8192, help="Maximum number of tokens to generate per rollout."),
    insert_reasoning_phrases: bool = Option(False, "--insert-reasoning-phrases/--no-insert-reasoning-phrases", help="Enable rewriting to insert reasoning phrases during inference."),
    data_path: str = Option("/new_data/aldo/v1_reasoning/grpo_feb_24th/deepscaler_phi_mini_nemotron.jsonl", help="Path to the data file."),
    min_samples_per_checkpoint: int = Option(30000, help="Minimum number of samples per checkpoint."),
    output_dir: str = Option("/new_data/experiments_rh/deepscaler_qwen1.5b_also_single_delimiter", help="Output directory where model checkpoints and configuration files will be saved."),
    infinite_sampler_seed: int = Option(37, help="Seed for InfiniteDistributedSampler, used to shuffle the data loader."),
    samples_per_question: int = Option(32, help="Number of samples per question to use in training."),
    constant_length_samples: int = Option(None, help="If set, forces all samples to be treated as having this output length for broadcasting advantages and other values. Defaults to None (use actual output lengths)."),
    dump_samples_filename: str = Option(None, help="Filename (in output_dir) to which raw samples will be appended as JSONL; no dumping if None."),
    clip_low: float = Option(0.2, help="Lower epsilon for PPO dual-clip algorithm (ratio >= 1 - clip_low)."),
    clip_high: float = Option(0.28, help="Upper epsilon for PPO dual-clip algorithm (ratio <= 1 + clip_high)."),
    clip_ratio_c: float = Option(10.0, help="Dual clip constant c for negative advantages."),
    num_training_batches: int = Option(1000000, help="Total number of training batches."),
    train_minibatch_size: int = Option(4000, help="Number of samples after which to trigger a GRADIENT_STEP message."),
    logging_level: LogLevelEnum = Option(LogLevelEnum.INFO, help="Logging level", case_sensitive=False),
    global_rank: int = Option(int(os.environ.get("RANK", 0)), help="Global rank of the process."), # Add global_rank from env
    use_torch_compile: bool = Option(
        True,
        "--use-torch-compile/--no-use-torch-compile",
        help="Enable or disable torch.compile to speed up training.",
    ),
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
        experience_batcher_name=experience_batcher_name,
        max_tokens_per_gpu=max_tokens_per_gpu,
        temperature=temperature,
        max_generation_tokens=max_generation_tokens,
        insert_reasoning_phrases=insert_reasoning_phrases,
        data_path=data_path,
        min_samples_per_checkpoint=min_samples_per_checkpoint,
        output_dir=output_dir,
        infinite_sampler_seed=infinite_sampler_seed,
        samples_per_question=samples_per_question,
        constant_length_samples=constant_length_samples,
        dump_samples_filename=dump_samples_filename,
        clip_low=clip_low,
        clip_high=clip_high,
        clip_ratio_c=clip_ratio_c,
        num_training_batches=num_training_batches,
        train_minibatch_size=train_minibatch_size,
        global_rank=global_rank, # Manually add global_rank
        use_torch_compile=use_torch_compile,
        mode='training',
    )
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Log parameters only on rank 0
    if args.global_rank == 0:
        params_to_log = args.__dict__.copy()
        params_to_log.update({
            "logging_level": logging_level.value,
            "WORLD_SIZE": int(os.environ.get("WORLD_SIZE", 1))
        })
        params_path = Path(output_dir) / f"training_params.json"
        with open(params_path, 'w') as f:
            json.dump(params_to_log, f, indent=4)
        print(f"Training with parameters: {json.dumps(params_to_log, separators=(',', ':'), indent=4)}")
        print(f"Training parameters saved to {params_path}")

    init_distributed_environment(args)
    model = setup_model(args)
    model, optimizer, lr_scheduler = setup_training_components(args, model)

    # Launch training, sample dumper actor handled inside train()
    asyncio.run(train(args, model, optimizer, lr_scheduler))

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
set -x NCCL_SOCKET_IFNAME eth1; set -x NCCL_IB_DISABLE 1; set -x CUDA_VISIBLE_DEVICES 4,5,6,7; mamba activate grpo; cd /new_data/aldo/v1_reasoning/grpo_feb_24th/; torchrun   --nnodes=1   --node_rank=(math 1 - 1)   --nproc_per_node=4   --rdzv_id=101   --rdzv_endpoint=10.241.128.19:54367   trainer_core.py     --model-name-or-path         /dev/shm/DeepSeek-R1-Distill-Qwen-1.5B     --learning-rate              125e-8     --batch-size                 128     --lr-scheduler               constant_with_warmup     --num-warmup-steps           10     --fsdp-sharding-strategy     SHARD_GRAD_OP     --max-tokens-per-gpu         80000     --samples-per-question       8     --temperature                0.6     --max-generation-tokens      16000     --data-path                  /new_data/aldo/v1_reasoning/grpo_feb_24th/deepscaler_r1_qwen1.5b.jsonl     --min-samples-per-checkpoint 30000     --output-dir                 /new_data/experiments_rh/deepscaler_r1_qwen1.5b_1.25e-6_clipping_v1     --infinite-sampler-seed      53     --train-minibatch-size       128     --num-training-batches       1000000     --logging-level              INFO   2>&1 | tee /new_data/experiments_rh/deepscaler_r1_qwen1.5b_1.25e-6_clipping_v1/train_1.log
'''