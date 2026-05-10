# End-to-End Memory Control

This is an independent project scaffold for moving from the SimGrid control
study to a real training-loop experiment. The first target is deliberately small:
a local PyTorch training loop on synthetic data, running on CPU by default, with
the same kind of knobs used in the simulator.

The goal is not to prove performance yet. The goal is to validate the integration
contract:

- collect per-step training telemetry;
- expose safe actuation points;
- change micro-batch and gradient accumulation online;
- record a CSV that can be analyzed like the SimGrid telemetry;
- keep a reproducible smoke test that works without a cluster.

## Files

- `train_tiny.py`: synthetic PyTorch training loop with `static`, `headroom`,
  `safe_greedy`, and checkpoint-backed `rl` controllers.
- `model_zoo.py`: interchangeable model registry used by the training loop.
- `check_run.py`: small CSV checker that reports throughput, OOM rate, memory
  proxy, and final knobs.
- `run_matrix.py`: runs controller x seed matrices and writes summary plus
  mean +/- 95% CI aggregates.
- `configs/smoke.json`: default smoke-test configuration.
- `configs/smoke_cnn.json`: same smoke test with a small CNN backbone.
- `configs/tight_memory.json`: tighter synthetic memory-budget setup for
  exercising fallback and greedy control.

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

## Next Steps

1. Validate transfer of the trained RL policy against `safe_greedy` on CUDA.
2. Add one real dataset/model pair behind the same `model_zoo.py` interface.
3. Test the same matrix on CUDA. When `--device cuda` is used, the CSV records
   `torch.cuda.max_memory_allocated()` and `torch.cuda.max_memory_reserved()`.
4. Test one framework integration path first, preferably plain PyTorch DDP before
   FSDP or DeepSpeed.
