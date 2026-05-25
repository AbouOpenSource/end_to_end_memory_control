from __future__ import annotations

import csv
import math
import os
import random
import time
from pathlib import Path

from model_zoo import build_model

from memory_control.controllers.adaptive import maybe_update_knobs
from memory_control.core.config import RunConfig, model_args_for_config
from memory_control.core.memory import estimate_peak_mb
from memory_control.core.types import ControllerState, Knobs
from memory_control.io.batches import build_batch_provider
from memory_control.io.telemetry import TELEMETRY_FIELDS, make_telemetry_row
from memory_control.runtime.distributed import distributed_info, init_distributed, sync_knobs_distributed
from memory_control.runtime.torch_utils import autocast_context, is_cuda_oom


def run_training(cfg: RunConfig, output: Path) -> None:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise SystemExit("PyTorch is required for this smoke test. Install it with: python -m pip install torch") from exc

    distributed, rank, local_rank, world_size = distributed_info(cfg, torch)

    seed = int(cfg.seed) + rank
    random.seed(seed)
    torch.manual_seed(seed)

    if distributed and str(cfg.device).startswith("cuda"):
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA device requested but torch.cuda.is_available() is false")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    init_distributed(distributed, device, torch)

    model, model_profile = build_model(cfg.model_name, model_args_for_config(cfg), nn)
    model = model.to(device)
    if cfg.compile_model:
        if not hasattr(torch, "compile"):
            raise SystemExit("compile_model=true requires torch.compile support")
        model = torch.compile(model)
    if distributed:
        ddp_kwargs = {"device_ids": [local_rank], "output_device": local_rank} if device.type == "cuda" else {}
        model = torch.nn.parallel.DistributedDataParallel(model, **ddp_kwargs)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model_parameters = sum(p.numel() for p in model.parameters())
    batch_provider = build_batch_provider(
        cfg,
        model_profile,
        device,
        torch,
        distributed_rank=rank,
        distributed_world_size=world_size,
    )

    knobs = Knobs(
        micro_batch=int(cfg.initial_micro_batch),
        grad_accum_steps=int(cfg.initial_grad_accum),
    )
    controller_state = ControllerState()

    if rank == 0:
        output.parent.mkdir(parents=True, exist_ok=True)

    output_path = output if rank == 0 else Path(os.devnull)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TELEMETRY_FIELDS)
        if rank == 0:
            writer.writeheader()

        for step in range(cfg.steps):
            step_knobs = knobs
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)

            started = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            total_loss = 0.0
            torch_oom = False

            compute_started = time.perf_counter()
            try:
                for _ in range(step_knobs.grad_accum_steps):
                    x, y = batch_provider.next(step_knobs.micro_batch)
                    with autocast_context(cfg, device, torch):
                        logits = model(x)
                        loss = F.cross_entropy(logits, y) / max(1, step_knobs.grad_accum_steps)
                    loss.backward()
                    total_loss += float(loss.detach().cpu()) * max(1, step_knobs.grad_accum_steps)

                opt.step()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            except RuntimeError as exc:
                if not (device.type == "cuda" and is_cuda_oom(exc)):
                    raise
                torch_oom = True
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                total_loss = float("nan")
            compute_s = time.perf_counter() - compute_started

            # Local single-process proxy for synchronization. Higher accumulation
            # amortizes fixed communication overhead, matching the SimGrid insight.
            comm_s = float(cfg.comm_base_s) + float(cfg.comm_scale_s) / math.sqrt(max(1, step_knobs.grad_accum_steps))
            io_s = float(cfg.io_s)
            step_s = (time.perf_counter() - started) + comm_s + io_s

            proxy_peak_mb = estimate_peak_mb(
                model_bytes=model_bytes,
                micro_batch=step_knobs.micro_batch,
                activation_units_per_sample=model_profile.activation_units_per_sample,
            )
            cuda_peak_allocated_mb = 0.0
            cuda_peak_reserved_mb = 0.0
            memory_source = "proxy"
            peak_mb = proxy_peak_mb
            if device.type == "cuda":
                cuda_peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                cuda_peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 * 1024)
                peak_mb = max(cuda_peak_reserved_mb, cuda_peak_allocated_mb)
                memory_source = "cuda"

            oom = torch_oom or peak_mb > cfg.budget_mb
            allocated_mb = cuda_peak_allocated_mb if memory_source == "cuda" else peak_mb * 0.85
            reserved_mb = cuda_peak_reserved_mb if memory_source == "cuda" else peak_mb * 0.95
            comm_frac = comm_s / max(step_s, 1e-12)
            io_frac = io_s / max(step_s, 1e-12)
            effective_samples = 0 if oom else step_knobs.micro_batch * step_knobs.grad_accum_steps
            effective_samples_per_s = effective_samples / max(step_s, 1e-12)
            knob_changed = False
            control_due = bool(oom or (step + 1) % cfg.control_interval == 0)
            if distributed:
                control_tensor = torch.tensor([int(control_due)], dtype=torch.int64, device=device)
                torch.distributed.all_reduce(control_tensor, op=torch.distributed.ReduceOp.MAX)
                control_due = bool(int(control_tensor.item()))

            if control_due:
                if rank == 0:
                    knobs, knob_changed = maybe_update_knobs(
                        cfg=cfg,
                        knobs=step_knobs,
                        state=controller_state,
                        step=step + 1,
                        peak_mb=peak_mb,
                        allocated_mb=allocated_mb,
                        reserved_mb=reserved_mb,
                        oom=oom,
                        compute_s=compute_s,
                        comm_s=comm_s,
                        io_s=io_s,
                        step_s=step_s,
                        comm_frac=comm_frac,
                        throughput=effective_samples_per_s,
                        model_bytes=model_bytes,
                        activation_units_per_sample=model_profile.activation_units_per_sample,
                    )
                knobs, knob_changed = sync_knobs_distributed(
                    enabled=distributed,
                    rank=rank,
                    device=device,
                    torch=torch,
                    knobs=knobs,
                    knob_changed=knob_changed,
                )

            if rank == 0:
                writer.writerow(
                    make_telemetry_row(
                        step=step + 1,
                        effective_samples=effective_samples,
                        step_s=step_s,
                        compute_s=compute_s,
                        comm_s=comm_s,
                        io_s=io_s,
                        comm_frac=comm_frac,
                        io_frac=io_frac,
                        peak_mb=peak_mb,
                        proxy_peak_mb=proxy_peak_mb,
                        cuda_peak_allocated_mb=cuda_peak_allocated_mb,
                        cuda_peak_reserved_mb=cuda_peak_reserved_mb,
                        memory_source=memory_source,
                        oom=oom,
                        step_knobs=step_knobs,
                        next_knobs=knobs,
                        cfg=cfg,
                        knob_changed=knob_changed,
                        controller_state=controller_state,
                        model_profile=model_profile,
                        distributed=distributed,
                        rank=rank,
                        world_size=world_size,
                        model_parameters=model_parameters,
                        total_loss=total_loss,
                        effective_samples_per_s=effective_samples_per_s,
                    )
                )

    if distributed and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

    if rank == 0:
        print(f"wrote {output}")
        print(
            "model="
            f"{model_profile.name} params={model_parameters} "
            f"dataset={cfg.dataset_name} precision={cfg.precision} "
            f"distributed={int(bool(distributed))} world_size={world_size} "
            f"input_shape={model_profile.input_shape} description={model_profile.description}"
        )
