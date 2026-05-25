from __future__ import annotations

import os
from typing import Any

from memory_control.core.config import RunConfig
from memory_control.core.types import Knobs


def distributed_info(cfg: RunConfig, torch: Any) -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = bool(cfg.distributed or world_size > 1)
    if not enabled:
        return False, 0, 0, 1

    if not torch.distributed.is_available():
        raise SystemExit("torch.distributed is not available in this PyTorch build")
    if world_size <= 1:
        raise SystemExit("distributed mode must be launched with torchrun and WORLD_SIZE > 1")
    return True, rank, local_rank, world_size


def init_distributed(enabled: bool, device: Any, torch: Any) -> None:
    if not enabled or torch.distributed.is_initialized():
        return
    backend = "nccl" if device.type == "cuda" else "gloo"
    torch.distributed.init_process_group(backend=backend)


def sync_knobs_distributed(
    *,
    enabled: bool,
    rank: int,
    device: Any,
    torch: Any,
    knobs: Knobs,
    knob_changed: bool,
) -> tuple[Knobs, bool]:
    if not enabled:
        return knobs, knob_changed
    tensor = torch.tensor(
        [int(knobs.micro_batch), int(knobs.grad_accum_steps), int(bool(knob_changed))],
        dtype=torch.int64,
        device=device,
    )
    torch.distributed.broadcast(tensor, src=0)
    synced = Knobs(int(tensor[0].item()), int(tensor[1].item()))
    return synced, bool(int(tensor[2].item())) if rank == 0 else synced != knobs
