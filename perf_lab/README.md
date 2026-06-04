# Auralis Performance Lab

Isolated CUDA/C++ performance experiments for Auralis. This lab is deliberately
separate from the training containers so kernel work can fail safely without
touching production runs.

## Goals

1. Measure the current bottlenecks with reproducible micro-benchmarks.
2. Prototype custom kernels behind small Python wrappers.
3. Verify numerical parity against the existing PyTorch implementation.
4. Promote only proven changes back into `src/auralis`.

## Current Experiments

- `rmsnorm_cuda`: fused RMSNorm forward/backward prototype.
- `bench_rmsnorm.py`: parity and speed test against the native PyTorch RMSNorm.
- `bench_training_candidate.py`: wrapper around the existing synthetic training
  benchmark, meant for comparing eager, compile, and later custom-kernel runs.

## Non-Goals

- No production training in this container.
- No checkpoint format changes.
- No model architecture changes.
- No replacement of the running `auralis-blackwell` container.

## Quick Start On <HOST>

```bash
cd <repo>
bash perf_lab/docker/build_image.sh
bash perf_lab/docker/run_shell.sh
```

Inside the container:

```bash
cd /workspace/v2data
python perf_lab/benchmarks/bench_rmsnorm.py --dtype bf16 --rows 4096 --dim 1280
python perf_lab/benchmarks/bench_training_candidate.py \
  --checkpoint checkpoints/runpod_import/pretrain_mix_v5_boosted_500m_a100_best_step9000/best.pt \
  --batch-size 2 --seq-length 2048 --grad-accum 4 --compile
```

