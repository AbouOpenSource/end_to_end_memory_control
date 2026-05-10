# Project Scope

This project is intentionally separate from `simgrid_cluster_env`.

Its role is to validate the smallest real training-loop integration before any
full cluster or framework work:

1. run a PyTorch model;
2. collect telemetry after each optimizer step;
3. expose safe control intervals;
4. change fast knobs online;
5. save a CSV that can be compared with the SimGrid metrics.
6. keep the model architecture interchangeable through a small registry, so the
   controller can be tested against MLP, CNN, and future real backbones without
   rewriting the training loop.

The first accepted milestone is a CPU smoke test. The next milestone is a GPU
single-process run with CUDA memory telemetry. Distributed PyTorch, FSDP, and
DeepSpeed should come only after those two milestones are stable.

Current completion target for the paper-facing scaffold:

- deterministic baselines: `static`, `headroom`, and `safe_greedy`;
- checkpoint-backed `rl` policy inference from the sibling `rl_memory_agent`
  project;
- per-step telemetry compatible with the paper metrics: throughput, step time,
  communication/I/O proxy fractions, peak memory, OOM, knob changes;
- CUDA telemetry when a CUDA device is selected;
- multi-seed runner with mean +/- 95% confidence intervals.
