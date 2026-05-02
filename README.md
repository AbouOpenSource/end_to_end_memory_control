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

- `train_tiny.py`: synthetic PyTorch training loop with `static` and `headroom`
  controllers.
- `check_run.py`: small CSV checker that reports throughput, OOM rate, memory
  proxy, and final knobs.
- `configs/smoke.json`: default smoke-test configuration.

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

## Next Steps

1. Replace the `headroom` controller with the trained RL policy.
2. Map the CSV fields to the paper metrics: throughput, step time, peak memory,
   OOM rate, and knob changes.
3. Add a real dataset/model pair after the smoke test is stable.
4. Add CUDA memory telemetry when running on GPU:
   `torch.cuda.max_memory_allocated()` and `torch.cuda.max_memory_reserved()`.
5. Test one framework integration path first, preferably plain PyTorch DDP before
   FSDP or DeepSpeed.
