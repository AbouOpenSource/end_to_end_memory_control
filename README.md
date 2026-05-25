# End-to-End Memory Control

[![CI/CD](https://github.com/AbouOpenSource/end_to_end_memory_control/actions/workflows/ci.yml/badge.svg)](https://github.com/AbouOpenSource/end_to_end_memory_control/actions/workflows/ci.yml)

This is an independent project scaffold for moving from the SimGrid control
study to real PyTorch training-loop experiments. It starts with reproducible CPU
smoke tests, then extends the same control interface to CUDA, CIFAR-10, optional
DDP, and checkpoint-backed RL policy transfer.

The goal is not to prove performance yet. The goal is to validate the integration
contract:

- collect per-step training telemetry;
- expose safe actuation points;
- change micro-batch and gradient accumulation online;
- record a CSV that can be analyzed like the SimGrid telemetry;
- keep a reproducible smoke test that works without a cluster.

## Files

- `train_tiny.py`: stable CLI entry point for one controller/seed run.
- `memory_control/cli.py`: command-line parsing and configuration assembly.
- `memory_control/core/config.py`: `RunConfig` and config/override handling.
- `memory_control/core/trainer.py`: PyTorch training loop and per-step control
  integration.
- `memory_control/controllers/adaptive.py`: `headroom`, `safe_greedy`, and
  checkpoint-backed `rl` controller decisions.
- `memory_control/io/batches.py`: synthetic and CIFAR-10 batch providers.
- `memory_control/io/telemetry.py`: telemetry schema and CSV row formatting.
- `memory_control/runtime/`: PyTorch autocast/OOM and distributed helper code.
- `model_zoo.py`: interchangeable model registry used by the training loop.
- `check_run.py`: small CSV checker that reports throughput, OOM rate, memory
  proxy, and final knobs.
- `run_matrix.py`: runs controller x seed matrices and writes summary plus
  mean +/- 95% CI aggregates.
- `cloud_validate.py`: cloud-oriented validation runner for CUDA, CIFAR-10,
  deterministic baselines, and optional RL transfer.
- `summarize_validation.py`: writes a Markdown validation report and plots from
  matrix outputs.
- `configs/smoke.json`: default smoke-test configuration.
- `configs/smoke_cnn.json`: same smoke test with a small CNN backbone.
- `configs/tight_memory.json`: tighter synthetic memory-budget setup for
  exercising fallback and greedy control.
- `configs/cloud_*.json`: longer CUDA validation configurations.

## Architecture

The project is organized around one stable contract: each training step emits
telemetry, and controllers can only change knobs at safe control points.

```text
configs/*.json
    |
    v
train_tiny.py -> memory_control.cli
    |
    +-- core
    |     +-- config.py
    |     +-- trainer.py
    |     +-- memory.py
    |
    +-- io
    |     +-- batches.py
    |     +-- telemetry.py
    |
    +-- controllers
    |     +-- adaptive.py
    |
    +-- runtime
    |     +-- torch_utils.py
    |     +-- distributed.py
    |
    v
per-step telemetry CSV
    |
    +-- check_run.py
    +-- run_matrix.py
    +-- summarize_validation.py
    v
summary.csv / aggregate_ci.csv / speedup_ci.csv / validation_report.md
```

### Runtime Components

- `RunConfig` in `memory_control/core/config.py` is the central experiment
  contract. It defines the model, dataset, device, precision, memory budget,
  controller, and RL checkpoint settings.
- `model_zoo.py` isolates model construction from the training loop. New
  architectures are added by registering a builder that returns a model and a
  `ModelProfile`.
- Batch providers in `memory_control/io/batches.py` isolate the data source.
  Synthetic data is used for fast CPU and CUDA smoke tests; CIFAR-10 is used
  for the first real dataset validation.
- Controllers operate only on fast knobs: `micro_batch` and
  `grad_accum_steps`. `memory_control/controllers/adaptive.py` holds the
  deterministic policies and RL policy adapter. The RL controller reuses the
  implementation from the sibling `rl_memory_agent` project.
- The telemetry CSV is the boundary between training and analysis. Downstream
  scripts do not inspect Python objects; they only consume CSV metrics emitted
  through `memory_control/io/telemetry.py`.

### Validation Layers

The validation stack is intentionally staged:

1. `train_tiny.py` runs one controller/seed pair and writes telemetry.
2. `run_matrix.py` runs controller x seed matrices and computes confidence
   intervals.
3. `cloud_validate.py` selects complete cloud presets for synthetic CUDA,
   CIFAR-10 CUDA, tight memory, optional DDP, and optional RL transfer.
4. `summarize_validation.py` produces a Markdown report and plots suitable for
   thesis/paper evidence.

DDP support is single-node and intentionally minimal: `torchrun` starts one
process per GPU, every rank trains, and rank 0 writes telemetry and broadcasts
controller knob updates.

## Runtime Flow

1. `train_tiny.py` loads a JSON configuration and applies CLI overrides.
2. The selected model is built through `model_zoo.py`.
3. The selected data path creates either synthetic batches or a CIFAR-10
   `DataLoader`.
4. The training loop executes `grad_accum_steps` micro-steps before each
   `optimizer.step()`.
5. Each optimizer step records compute time, communication proxy time, I/O
   proxy time, throughput, memory, OOM status, active knobs, next knobs, and
   controller metadata.
6. At `control_interval` boundaries, or after OOM, the selected controller
   decides whether to keep, shrink, increase, or roll back the knobs.
7. The telemetry is written as CSV.
8. Matrix and report scripts aggregate the CSV files across controllers and
   seeds.

## Controlled Knobs

The executable knobs are intentionally small and fast to change:

- `micro_batch`: number of samples processed in one forward/backward micro-step.
- `grad_accum_steps`: number of micro-steps accumulated before
  `optimizer.step()`.

The effective batch per optimizer step is:

```text
effective_samples = micro_batch * grad_accum_steps
```

The RL policy may internally reason over a richer `KnobConfig` from
`rl_memory_agent`, but this prototype only enacts `micro_batch` and
`grad_accum_steps`. This keeps the real-loop validation focused and avoids
mixing in framework-specific knobs before the core contract is stable.

## Controllers

### `static`

Keeps the initial knobs fixed for the whole run. This is the simplest baseline.

### `headroom`

Uses memory headroom and communication fraction heuristics:

- if the step is unsafe or OOM, move to safer knobs;
- if memory is far below budget and communication fraction is high, increase
  `grad_accum_steps`;
- if memory is far below budget and micro-batch can grow, increase
  `micro_batch`.

### `safe_greedy`

Deterministic stronger baseline:

- tracks the best safe knobs observed so far;
- tries larger knobs when predicted memory is below the safe limit;
- rolls back to the best known safe knobs when a trial does not improve enough;
- falls back to safer knobs after OOM or budget violation.

This is the main deterministic baseline to compare against the RL controller.

### `rl`

Loads a trained PPO/Lagrangian checkpoint from the sibling `rl_memory_agent`
project. The controller:

- imports `rl_memory_agent` from `../rl_memory_agent/src`;
- builds the RL observation window from real training telemetry;
- selects an action from the trained policy;
- applies the safety shield before enacting the proposed knobs;
- rolls back to the last safe RL config after OOM when available.

The default action profile is `fast_only`, matching the pretrained checkpoint
path used in the thesis experiments.

## Data Paths

### Synthetic

Synthetic data is the default. It is used for:

- fast local CPU smoke tests;
- reproducible CI;
- large synthetic CUDA stress tests;
- debugging the validation pipeline without downloading datasets.

Synthetic inputs are generated from the selected `ModelProfile`, so each model
receives tensors with the expected shape.

### CIFAR-10

CIFAR-10 is the first real dataset validation path. It requires `torchvision`
and is enabled with:

```bash
python train_tiny.py \
  --config configs/cloud_cifar10_cuda.json \
  --dataset cifar10 \
  --download-dataset \
  --data-dir data \
  --device cuda \
  --output out/cifar10_safe_greedy.csv
```

CIFAR-10 currently expects a model profile with input shape `(3, 32, 32)`.
Use `tiny_cnn` or add another compatible image model to `model_zoo.py`.

## Memory Measurement

CPU runs cannot observe real GPU memory. They use a proxy estimate:

- parameter bytes;
- optimizer-state proxy;
- activation proxy proportional to `micro_batch`;
- workspace proxy.

CUDA runs record real PyTorch memory telemetry:

- `torch.cuda.max_memory_allocated()`;
- `torch.cuda.max_memory_reserved()`.

When CUDA is enabled, the reported `peak_mb` is the maximum of allocated and
reserved CUDA memory. When CUDA is not enabled, `peak_mb` comes from the proxy.

## Telemetry Schema

Each `train_tiny.py` run writes one CSV row per optimizer step.

Important columns:

- `step`: optimizer-step index.
- `samples`: effective samples processed at that step, or `0` after OOM.
- `step_s`: total measured step time plus communication/I/O proxies.
- `compute_s`: measured PyTorch compute time.
- `comm_s`: local communication proxy.
- `io_s`: configured I/O proxy.
- `comm_frac`: `comm_s / step_s`.
- `io_frac`: `io_s / step_s`.
- `peak_mb`: memory value used by the controller.
- `proxy_peak_mb`: CPU/synthetic memory proxy.
- `cuda_peak_allocated_mb`: CUDA allocated peak, when available.
- `cuda_peak_reserved_mb`: CUDA reserved peak, when available.
- `memory_source`: `proxy` or `cuda`.
- `budget_mb`: memory budget for the run.
- `safe_limit_mb`: `budget_mb * (1 - headroom_margin)`.
- `oom`: `1` if PyTorch OOM or budget violation occurred.
- `micro_batch`: knobs used for the current step.
- `grad_accum_steps`: knobs used for the current step.
- `next_micro_batch`: knobs selected for the next step.
- `next_grad_accum_steps`: knobs selected for the next step.
- `controller`: selected controller.
- `knob_changed`: `1` when the controller changed knobs.
- `controller_action`: RL action name or controller metadata.
- `safety_allowed`: safety-shield decision for RL.
- `safety_reason`: safety-shield reason or rollback metadata.
- `model_name`: executed model.
- `dataset_name`: `synthetic` or `cifar10`.
- `precision`: `fp32` or `bf16`.
- `compile_model`: `1` when `torch.compile` was enabled.
- `distributed`: `1` when DDP was enabled.
- `world_size`: DDP process count.
- `loss`: current loss value.
- `effective_samples_per_s`: per-step throughput.
- `best_safe_throughput`: best safe throughput tracked by `safe_greedy`.

## Configuration Reference

Core fields:

- `steps`: number of optimizer steps.
- `seed`: random seed.
- `device`: `cpu`, `cuda`, or a CUDA device string.
- `dataset_name`: `synthetic` or `cifar10`.
- `data_dir`: dataset cache directory.
- `download_dataset`: whether to download CIFAR-10.
- `num_workers`: DataLoader worker count.
- `pin_memory`: DataLoader pinned-memory flag for CUDA.
- `precision`: `fp32` or `bf16`.
- `compile_model`: enable `torch.compile`.
- `model_name`: `linear`, `mlp`, or `tiny_cnn`.
- `model_args`: model-specific constructor arguments.
- `initial_micro_batch`: starting micro-batch.
- `initial_grad_accum`: starting gradient accumulation.
- `max_micro_batch`: upper bound for micro-batch.
- `max_grad_accum`: upper bound for gradient accumulation.
- `control_interval`: steps between controller decisions.
- `budget_mb`: memory budget.
- `headroom_margin`: safety margin below the budget.
- `learning_rate`: AdamW learning rate.
- `controller`: `static`, `headroom`, `safe_greedy`, or `rl`.
- `safe_greedy_min_improvement`: improvement threshold for accepting trials.
- `rl_checkpoint`: PPO checkpoint path for `controller=rl`.
- `rl_action_profile`: `fast_only` or `all`.
- `rl_window_len`: telemetry window length for the RL state builder.
- `rl_device`: device used for RL policy inference.

## Quick Start

From this folder:

```bash
python3 -m venv .venv-e2e
. .venv-e2e/bin/activate
python -m pip install -r requirements.txt

python train_tiny.py \
  --config configs/smoke.json \
  --output out/smoke_headroom.csv

python check_run.py out/smoke_headroom.csv
```

For a static baseline:

```bash
python train_tiny.py \
  --config configs/smoke.json \
  --controller static \
  --output out/smoke_static.csv
```

Then compare the two outputs:

```bash
python check_run.py out/smoke_static.csv
python check_run.py out/smoke_headroom.csv
```

For the stronger deterministic baseline:

```bash
python train_tiny.py \
  --config configs/smoke.json \
  --controller safe_greedy \
  --output out/smoke_safe_greedy.csv

python check_run.py out/smoke_safe_greedy.csv
```

For a trained RL checkpoint from the sibling `rl_memory_agent` project:

```bash
CKPT=../simgrid_cluster_env/runs_pretrained_long_compare/batch_20260430_pretrained_20000u/tight_budget__rl__seed0/agent_ckpts/final.pt

python train_tiny.py \
  --config configs/smoke.json \
  --controller rl \
  --rl-checkpoint "$CKPT" \
  --output out/smoke_rl.csv

python check_run.py out/smoke_rl.csv
```

The default `--rl-action-profile fast_only` matches the pretrained checkpoint
above. Use `--rl-device cuda` only for policy inference if CUDA is available and
the checkpoint/model load path has been validated on that host.

To run a small multi-seed matrix and generate confidence intervals:

```bash
python run_matrix.py \
  --config configs/smoke.json \
  --controllers static,headroom,safe_greedy \
  --seeds 0,1,2,3,4 \
  --out-dir out/matrix_smoke
```

This writes:

- `out/matrix_smoke/summary.csv`: one row per controller/seed run.
- `out/matrix_smoke/aggregate_ci.csv`: metric means with 95% CI.
- `out/matrix_smoke/speedup_ci.csv`: paired throughput speedups against the
  selected baseline.

For a tighter memory-pressure smoke test:

```bash
python run_matrix.py \
  --config configs/tight_memory.json \
  --controllers static,headroom,safe_greedy \
  --seeds 0,1,2 \
  --baseline static \
  --out-dir out/matrix_tight
```

## Cloud Validation

The complete validation code is included but is meant to run on a machine with
an NVIDIA GPU. On the cloud VM:

```bash
python3 -m venv .venv-e2e
. .venv-e2e/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cloud.txt
```

Run the deterministic CUDA validation suite:

```bash
python cloud_validate.py \
  --preset all \
  --download-dataset \
  --data-dir data \
  --seeds 0,1,2,3,4 \
  --precision bf16
```

For a multi-GPU cloud VM, add DDP:

```bash
python cloud_validate.py \
  --preset all \
  --download-dataset \
  --data-dir data \
  --seeds 0,1,2,3,4 \
  --precision bf16 \
  --distributed \
  --nproc-per-node 2
```

This runs:

- synthetic CUDA matrix: `static`, `headroom`, `safe_greedy`;
- CIFAR-10 CUDA matrix: `static`, `headroom`, `safe_greedy`;
- tighter CIFAR-10 memory-pressure matrix: `static`, `headroom`, `safe_greedy`.

Outputs are written under `out/cloud_validation/<timestamp>/`:

- one subdirectory per validation case;
- per-run telemetry CSVs;
- `summary.csv`, `aggregate_ci.csv`, and `speedup_ci.csv`;
- `validation_report.md`;
- optional PNG plots when `matplotlib` is installed.

If a trained RL checkpoint is available, add the transfer validation:

```bash
python cloud_validate.py \
  --preset all \
  --download-dataset \
  --data-dir data \
  --seeds 0,1,2,3,4 \
  --rl-checkpoint ../simgrid_cluster_env/runs_pretrained_long_compare/batch_20260430_pretrained_20000u/tight_budget__rl__seed0/agent_ckpts/final.pt \
  --rl-action-profile fast_only
```

To debug the pipeline locally without CUDA, use a short CPU run:

```bash
python cloud_validate.py \
  --preset synthetic \
  --allow-cpu \
  --device cpu \
  --steps 5 \
  --seeds 0
```

## Interchangeable Models

The training loop does not instantiate a hard-coded model anymore. It reads
`model_name` and `model_args` from the config, builds the model through
`model_zoo.py`, and generates synthetic inputs from the selected model profile.

Available models:

- `linear`: single linear classifier for the fastest smoke tests.
- `mlp`: configurable multi-layer perceptron, used by `configs/smoke.json`.
- `tiny_cnn`: small convolutional image classifier, used by `configs/smoke_cnn.json`.

Example with the default MLP:

```json
{
  "model_name": "mlp",
  "model_args": {
    "input_dim": 256,
    "hidden_dim": 512,
    "hidden_layers": 2,
    "num_classes": 10
  }
}
```

Switch to the CNN without changing the training code:

```bash
python train_tiny.py \
  --config configs/smoke_cnn.json \
  --output out/smoke_cnn.csv
```

You can also override the model from the command line:

```bash
python train_tiny.py \
  --config configs/smoke.json \
  --model linear \
  --model-arg input_dim=256 \
  --model-arg num_classes=10 \
  --output out/smoke_linear.csv
```

To add a new model, register a builder in `model_zoo.py`. The builder must
return both the `torch.nn.Module` and a `ModelProfile` containing the expected
input shape, number of classes, and activation-size proxy used by the memory
controller.

## Multi-GPU DDP Details

For a single-node multi-GPU VM:

```bash
python cloud_validate.py \
  --preset all \
  --download-dataset \
  --data-dir data \
  --seeds 0,1,2,3,4 \
  --precision bf16 \
  --distributed \
  --nproc-per-node 2
```

DDP behavior:

- `run_matrix.py` launches `train_tiny.py` through `torch.distributed.run`.
- Each rank executes training.
- Rank 0 writes telemetry.
- Rank 0 computes controller decisions.
- Updated knobs are broadcast to other ranks.
- The CSV records `distributed=1` and `world_size=<n>`.

This DDP path is intentionally minimal. It validates the control/telemetry
contract before adding FSDP or DeepSpeed.

## RL Transfer Details

If a trained RL checkpoint from the sibling SimGrid/RL workflow is available:

```bash
python cloud_validate.py \
  --preset all \
  --download-dataset \
  --data-dir data \
  --seeds 0,1,2,3,4 \
  --rl-checkpoint ../simgrid_cluster_env/runs_pretrained_long_compare/batch_20260430_pretrained_20000u/tight_budget__rl__seed0/agent_ckpts/final.pt \
  --rl-action-profile fast_only
```

For RL-only validation:

```bash
python cloud_validate.py \
  --preset rl \
  --download-dataset \
  --data-dir data \
  --seeds 0,1,2,3,4 \
  --rl-checkpoint path/to/final.pt \
  --rl-action-profile fast_only
```

The RL transfer case compares `rl` against `safe_greedy` on the tight CIFAR-10
CUDA configuration. Interpret this comparison only when both controllers use
the same seeds, same budget, same model, same dataset, and same device.

## Output Structure

A cloud validation run writes:

```text
out/cloud_validation/<timestamp>/
+-- metadata.json
+-- validation_report.md
+-- synthetic_cuda/
|   +-- static__seed0.csv
|   +-- headroom__seed0.csv
|   +-- safe_greedy__seed0.csv
|   +-- summary.csv
|   +-- aggregate_ci.csv
|   +-- speedup_ci.csv
|   +-- plots/
+-- cifar10_cuda/
+-- cifar10_tight_cuda/
```

`metadata.json` records:

- CLI arguments;
- PyTorch version;
- CUDA availability;
- CUDA version;
- GPU name;
- total GPU memory when available;
- validation cases selected.

`validation_report.md` is the main artifact to archive for the thesis. It
contains hardware details, per-case aggregate tables, speedup tables, and
validation checks.

## How To Interpret Results

Use `validation_report.md` first. The main values to compare are:

- throughput samples/s;
- throughput speedup percentage against the selected baseline;
- max peak memory;
- OOM rate;
- number of knob changes;
- final knobs in the raw telemetry;
- `memory_source`.

Recommended reading:

- `static` establishes the fixed-knob baseline.
- `headroom` shows whether simple memory heuristics are sufficient.
- `safe_greedy` is the stronger deterministic baseline.
- `rl` is meaningful only when compared against `safe_greedy` under identical
  conditions.

Do not report CPU proxy memory as real GPU memory. Only rows with
`memory_source=cuda` should be used as CUDA memory evidence.

## CI

The GitHub workflow performs:

- Python source compilation;
- CPU smoke tests for `static`, `headroom`, and `safe_greedy`;
- small multi-seed matrix generation;
- validation contract tests;
- CPU DDP smoke test.

The CI does not prove CUDA performance. It proves that the validation pipeline
and CSV contract remain executable without a GPU.

## Troubleshooting

### CUDA is not available

Error:

```text
CUDA device requested but torch.cuda.is_available() is false
```

Use a GPU VM and install the cloud dependencies:

```bash
python -m pip install -r requirements-cloud.txt
```

Also check:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

### CIFAR-10 requires torchvision

Error:

```text
dataset_name=cifar10 requires torchvision
```

Install:

```bash
python -m pip install -r requirements-cloud.txt
```

### RL checkpoint not found

Pass an absolute path or a path relative to this project directory:

```bash
--rl-checkpoint /absolute/path/to/final.pt
```

### RL action profile mismatch

Use the checkpoint action profile. For the thesis pretrained checkpoint, this is
usually:

```bash
--rl-action-profile fast_only
```

### CUDA OOM at the beginning of a run

Lower one of these fields in the selected config:

- `initial_micro_batch`;
- `max_micro_batch`;
- model width or hidden size.

Or increase `budget_mb` to match the selected GPU.

### No plots in the report

Install `matplotlib`. CSV summaries and Markdown tables are still generated
without plots.

## Current Limitations

- CPU memory is a proxy, not a real allocator measurement.
- CIFAR-10 is the first real dataset path; larger datasets and models still need
  to be added.
- The DDP implementation is single-node and minimal.
- FSDP and DeepSpeed are not implemented yet.
- The RL controller only enacts `micro_batch` and `grad_accum_steps` in this
  prototype.
- The communication time is still a local proxy, not a real NCCL timing model.

## Next Steps

1. Run the cloud validation suite above and archive `validation_report.md`.
2. Validate transfer of the trained RL policy against `safe_greedy` on CUDA.
3. Tune the CUDA budgets in `configs/cloud_*.json` for the selected GPU type.
4. Extend the DDP path toward FSDP or DeepSpeed only after single-GPU and DDP
   telemetry are stable.
