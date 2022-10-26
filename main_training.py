import argparse
import os
import time

import colorama

import environment

import model_checkpointing
import torch

import torch.distributed as dist

from colorama import Fore

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
)

bf16_ready = environment.verify_bfloat_support

from torch.utils.data import DistributedSampler


colorama.init(autoreset=True)  # reset after every line

import performance

# import optimizers


def print_model(model, file_name, rank):
    if rank != 0:
        return

    fn = file_name
    with open(fn, "w") as external_file:

        print(f"model wrapping = \n{model}\n\n", file=external_file)

        external_file.close()


def setup():
    """we use torchrun for init so no params needed here"""
    dist.init_process_group("nccl")


def setup_environ_flags(cfg, rank):
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    if cfg.nccl_debug_handler:
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    if cfg.distributed_debug:
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        if rank == 0:
            print(f"--> running with torch dist debug set to detail")


def cleanup():
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    if rank == 0:
        print(f"clearing gpu cache for all ranks")
    torch.cuda.empty_cache()


def setup_tasks(rank, world_size, cfg):
    """keep the basic setup list here"""
    setup()
    clear_gpu_cache(rank)  # need to call torch set device first?
    # set_printing()
    setup_environ_flags(cfg, rank)


# ------ main code loop -----------------
def fsdp_main(wrap_policy, batch_size_training):
    """main process,  within each rank process"""

    cfg = config.train_config()  # loads from defaults

    torch.cuda.manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # torchrun specific
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        print(f"--> World Size = {world_size}\n")
        print(f"--> Device_count = {torch.cuda.device_count()}")
        print(f"--> running with these defaults {cfg}")
        # time_of_run = get_date_of_run()

    setup_tasks(rank, world_size, cfg)

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)

    # setup memory tracking for perf
    if local_rank == 0:
        memmax = performance.Memory_Maximizer()
    else:
        memmax = None

    # ====   use new transformer wrapper
    if rank == 0:
        print(f"policy is {wrap_policy}")
    dataset = config.get_dataset()
    train_sampler = DistributedSampler(
        dataset, rank=dist.get_rank(), num_replicas=dist.get_world_size(), shuffle=True
    )

    if cfg.run_validation:
        val_dataset = config.get_dataset(train=False)
        val_sampler = DistributedSampler(
            val_dataset, rank=dist.get_rank(), num_replicas=dist.get_world_size()
        )

    if local_rank == 0:
        print(f"\n--> Prepping {cfg.model_name} model ...\n")
    model = config.build_model(cfg.model_name)

    if local_rank == 0:
        print(f"--> {cfg.model_name} built.")
        num_params = (sum(p.numel() for p in model.parameters())) / 1e6
        print(f"built model with {num_params}M params")

    mp_policy = None

    if cfg.use_mixed_precision and bf16_ready:
        mp_policy = cfg.mp_policy

        if rank == 0:
            print(f"bf16 check passed")
            print(f"\n--> Running with mixed precision {cfg.mp_policy} policy")

    else:
        if rank == 0:
            print(f"--> Warning - bf16 support not available.  Using fp32")

    # if not using mixed precision, turn on TF32 for matmul?
    if not cfg.use_mixed_precision and cfg.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        if rank == 0:
            print(f"--> TF32 support for matmul enabled. ")

    if local_rank == 0:
        init_start = time.perf_counter()

    # preload checkpoint if desired
    if cfg.load_model_checkpoint:
        if cfg.checkpoint_type == StateDictType.FULL_STATE_DICT:
            model_checkpointing.load_model_checkpoint(model, rank, cfg)

        elif cfg.checkpoint_type == StateDictType.LOCAL_STATE_DICT:
            model_checkpointing.load_distributed_model_checkpoint(model, rank, cfg)

    prefetch_policy = cfg.backward_prefetch
    try:
        from torch.distributed.fsdp.wrap import ParamExecOrderPolicy

        if isinstance(wrap_policy, ParamExecOrderPolicy):
            prefetch_policy = None
    except ImportError:
        ...
    if rank == 0:
        print(f"backward prefetch set to {prefetch_policy}")
        print(f"limit all-gathers set to {cfg.limit_all_gathers}")
        print(f"sharding set to {cfg.sharding_strategy}")
        print(f"--> Batch Size = {batch_size_training}")

    # model weights to BF16?
    if cfg.model_weights_bf16:
        model = model.to(torch.bfloat16)
        mp_policy = None
        if rank == 0:
            print(f"--> Model converted to BF16.\nRunning in ** PURE ** BFloat mode")

    # ----- main FSDP init -----------
    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mp_policy,
        backward_prefetch=prefetch_policy,
        sharding_strategy=cfg.sharding_strategy,
        device_id=torch.cuda.current_device(),
        forward_prefetch=cfg.forward_prefetch,
        limit_all_gathers=cfg.limit_all_gathers,
    )

    if (
        cfg.load_model_checkpoint
        and cfg.checkpoint_type == StateDictType.SHARDED_STATE_DICT
    ):
        model_checkpointing.load_model_sharded(model, rank, cfg)

    if cfg.fsdp_activation_checkpointing:
        config.fsdp_checkpointing(model)
        if rank == 0:
            print(f"--> FSDP activation checkpointing in use")

    # print sharding plan?
    if rank == 0 and cfg.print_sharding_plan:
        print(model)

    # postload checkpoint if desired
    if (
        cfg.load_model_checkpoint
        and cfg.checkpoint_type == StateDictType.LOCAL_STATE_DICT
    ):
        model_checkpointing.load_distributed_model_checkpoint(model, rank, cfg)

    if local_rank == 0:
        init_time = time.perf_counter() - init_start
        print(f"local rank {local_rank} init time = {init_time}")

    # data loader -------------
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size_training,
        num_workers=cfg.num_workers_dataloader,
        pin_memory=False,
        sampler=train_sampler,
    )

    if cfg.run_validation:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.val_batch_size,
            num_workers=cfg.num_workers_dataloader,
            pin_memory=False,
            sampler=val_sampler,
        )

    # memory and timing tracking
    if local_rank == 0:
        memmax.start()
        # torch.cuda.reset_peak_memory_stats()
        tracking_duration = []
    else:
        tracking_duration = None

    # warmup, this is only used in the non-recursive ParamExecOrderPolicy
    config.train(
        model, data_loader, None, None, memmax, local_rank, tracking_duration, 1
    )
    if rank == 0:
        print("Finish warm up")
    model.zero_grad()

    # optimizer ----------
    optimizer = None
    lr = 8e-4
    weight_decay = 0.0

    if cfg.optimizer == "int8":
        import bitsandbytes as bnb

        optimizer = bnb.optim.Adam8bit(
            model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=False
        )
        if rank == 0:
            print(f"Running with 8 bit optimizer")

    elif cfg.optimizer == "AnyPrecision":
        import optimizers

        optimizer = optimizers.AnyPrecisionAdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum_dtype=cfg.ap_momentum_dtype,
            variance_dtype=cfg.ap_variance_dtype,
            use_kahan_summation=cfg.ap_use_kahan_summation,
        )
        if rank == 0:
            print(
                f"Running with AnyPrecision Optimizer, momo={cfg.ap_momentum_dtype}, var = {cfg.ap_variance_dtype}, kahan summation =  {cfg.ap_use_kahan_summation}"
            )

    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=False
        )
        if rank == 0:
            print(f"Running with AdamW optimizer")

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # load optimizer checkpoint
    if cfg.load_optimizer:
        model_checkpointing.load_optimizer_checkpoint(model, optimizer, rank, cfg)

    torch_profiler = None
    if cfg.run_profiler and rank == 0:
        print(f"Profiling active.  Traces will be saved at {cfg.profile_folder}")

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(cfg.profile_folder),
            profile_memory=True,
            with_stack=False,  # `True` causes seg fault with EFA
            record_shapes=False,  # `True` causes seg fault with `export_chrome_trace()`
        ) as torch_profiler:
            config.train(
                model,
                data_loader,
                torch_profiler,
                optimizer,
                memmax,
                local_rank,
                tracking_duration,
                cfg.total_steps_to_run,
            )
    else:
        for i in range(1, cfg.num_epochs + 1):
            config.train(
                model,
                data_loader,
                None,
                optimizer,
                memmax,
                local_rank,
                tracking_duration,
                cfg.total_steps_to_run,
            )
            if cfg.total_steps_to_run is not None:
                break

            if cfg.run_validation:
                config.validation(model, local_rank, rank, val_loader, world_size)

        # checkpointing for model and optimizer
        if cfg.save_model_checkpoint:

            if cfg.checkpoint_type == StateDictType.FULL_STATE_DICT:

                model_checkpointing.save_model_checkpoint(
                    model, optimizer, rank, cfg, epoch=1
                )
            elif cfg.checkpoint_type == StateDictType.LOCAL_STATE_DICT:
                print(f"Saving Model via Distributed Checkpoint")
                model_checkpointing.save_distributed_model_checkpoint(model, rank, cfg)

            elif cfg.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                model_checkpointing.save_model_sharded(model, rank, cfg)

        if cfg.save_optimizer:
            model_checkpointing.save_optimizer_checkpoint(
                model, optimizer, rank, cfg, epoch=1
            )

    # memory summary
    if local_rank == 0:

        # memory monitor
        memmax.stop()  # stop and display info

        stable_sum = sum(tracking_duration[1:])
        if cfg.total_steps_to_run is not None:
            stable_avg = stable_sum / cfg.total_steps_to_run
            stable_avg = round(stable_avg, 4)
            print(
                Fore.GREEN
                + f"\n--> Step avg speed based on {cfg.total_steps_to_run} steps: {stable_avg} seconds"
            )
        print(Fore.LIGHTBLUE_EX + f"\n--> Model Size =  {num_params} M Params")
        if cfg.print_memory_summary:
            print(
                f"\nCUDA Memory Summary After Training:\n {torch.cuda.memory_summary()}"
            )

    cleanup()


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch experiments with FSDP")
    parser.add_argument(
        "--model",
        default="deepvit",
        metavar="string",
        choices=["deepvit", "t5", "regnet"],
        help="choose model to run, available: `deepvit`, `t5`, `regnet` (default: deepvit)",
    )
    parser.add_argument(
        "--use-nonrecursive",
        action="store_true",
        help="whether to use non-recursive",
    )
    parser.add_argument(
        "--bucket-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=15,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    assert args.model in ["deepvit", "t5", "regnet"]
    if args.model == "deepvit":
        import config.deepvit_config as config
    elif args.model == "t5":
        import config.t5_config as config
    elif args.model == "regnet":
        import config.regnet_config as config

    wrap_policy = config.get_policy(args.use_nonrecursive, args.bucket_size)
    fsdp_main(wrap_policy, args.batch_size)
