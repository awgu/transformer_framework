import functools
from dataclasses import dataclass

import policies
import torch

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.fsdp import BackwardPrefetch, ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy, transformer_auto_wrap_policy


@dataclass
class base_config:

    # seed
    seed: int = 2022
    verbose: bool = True  # how much info to show...
    # how many mini batches to time with
    total_steps_to_run: int = 12

    # stats
    print_memory_summary: bool = False

    # training
    num_epochs: int = 2

    model_weights_bf16: bool = False  # warning, True will  move model weights to BF16...use BFF_AdamW optimizer

    # policies
    use_mixed_precision: bool = True
    # mp_policy = policies.bf16_policy
    mp_policy = policies.fp16_policy

    use_low_precision_gradient_policy: bool = False
    # this is only for fp32 scenario...
    use_tf32: bool = False

    # optimizer config
    optimizer: str = (
        "AnyPrecision"  # [AdamW, AnyPrecision, int8] (fp32, bf16, int8 optimizers)
    )

    ap_momentum_dtype = torch.float32  # momentum and variance
    ap_variance_dtype = torch.float32  # variance

    ap_use_kahan_summation: bool = False

    # sharding policy
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    print_sharding_plan: bool = True

    run_profiler: bool = True
    profile_folder: str = "fsdp/profile_tracing"

    backward_prefetch = BackwardPrefetch.BACKWARD_PRE
    # backward_prefetch = None
    forward_prefetch = False
    limit_all_gathers = True
    use_orig_params = True

    # log
    log_every: int = 1

    # dataloaders
    num_workers_dataloader: int = 2

    # training
    batch_size_training: int = 48

    # activation checkpointing
    fsdp_activation_checkpointing: bool = False

    # validation
    run_validation: bool = False
    val_batch_size = 18
    val_batch_size = 4

    # logging
    track_memory = True
    memory_report: bool = True
    nccl_debug_handler: bool = True
    distributed_debug: bool = True


def get_policy_base(use_nonrecursive, bucket_size, blocks):
    cfg = base_config()
    if not use_nonrecursive:
        return ModuleWrapPolicy(blocks)
    else:
        # The ParamExecOrderPolicy that is in development
        from torch.distributed.fsdp.wrap import (
            always_wrap_policy,
            HandleInitMode,
            ParamExecOrderPolicy,
        )

        return ParamExecOrderPolicy(
            handle_init_mode=HandleInitMode.MODULE_LEVEL,
            bucket_size=bucket_size,
            module_level_group_policy=always_wrap_policy,
        )


def fsdp_checkpointing_base(model, blocks):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        offload_to_cpu=False,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    check_fn = lambda submodule: isinstance(submodule, blocks)
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )
