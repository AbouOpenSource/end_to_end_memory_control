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
- `model_zoo.py`: interchangeable model registry used by the training loop.
- `check_run.py`: small CSV checker that reports throughput, OOM rate, memory
  proxy, and final knobs.
- `configs/smoke.json`: default smoke-test configuration.
- `configs/smoke_cnn.json`: same smoke test with a small CNN backbone.

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

1. Replace the `headroom` controller with the trained RL policy.
2. Add one real dataset/model pair behind the same `model_zoo.py` interface.
3. Map the CSV fields to the paper metrics: throughput, step time, peak memory,
   OOM rate, and knob changes.
4. Add CUDA memory telemetry when running on GPU:
   `torch.cuda.max_memory_allocated()` and `torch.cuda.max_memory_reserved()`.
5. Test one framework integration path first, preferably plain PyTorch DDP before
   FSDP or DeepSpeed.
