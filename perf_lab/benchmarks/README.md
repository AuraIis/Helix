# Benchmarks

Run these inside `auralis-perf-lab`.

## RMSNorm

```bash
python perf_lab/benchmarks/bench_rmsnorm.py --dtype bf16 --rows 4096 --dim 1280
```

Expected behavior:

- parity errors stay small enough for bf16/fp16 training use
- custom kernel is only promoted if faster than PyTorch at production-ish shapes

## Training Candidate

```bash
python perf_lab/benchmarks/bench_training_candidate.py \
  --checkpoint checkpoints/runpod_import/pretrain_mix_v5_boosted_500m_a100_best_step9000/best.pt \
  --batch-size 2 --seq-length 2048 --grad-accum 4 --compile
```

This does not train the real model. It uses synthetic token batches to compare
runtime variants.

## Huge-Vocab Cross-Entropy Baseline

```bash
python perf_lab/benchmarks/bench_vocab_ce.py \
  --tokens 8192 --d-model 1280 --vocab-size 200000 --dtype bf16 \
  --hidden-scale 1.0 --weight-scale 0.02795
```

Use `--variant both` to compare PyTorch full logits against the chunked exact
prototype. The chunked path is expected to save logits memory first; speed comes
later when the hot loop is moved into a custom CUDA kernel. Use
`--chunked-impl triton` only for the current hybrid experiment
(parallel Triton forward plus C++ chunked backward). Use
`--chunked-impl triton_fused` for the first correctness-first Triton forward
plus Triton backward prototype; it is not a production speed path yet.
Use `--triton-backward-mode atomic_lowmem` to test the same atomic layout with
grad buffers in parameter dtype. Use `--triton-backward-mode atomic_mixed` to
keep `grad_hidden` accumulation in fp32 while keeping the huge `grad_weight`
buffer low-memory. Append `_bf16match` to `atomic`, `atomic_lowmem`, or
`atomic_mixed` to round Triton logits to bf16 before CE math, matching the
current bf16 full-logits reference more closely. Use
`--triton-backward-mode partial_weight` for the bounded two-stage
`grad_weight` reduction experiment. Use
`--triton-backward-mode split_hidden` to test the no-hidden-atomics backward
variant, or `--triton-backward-mode split_weight` to test the no-weight-atomics
variant.

## Chunk-Size Sweep

```bash
python perf_lab/benchmarks/sweep_chunked_ce.py \
  --tokens 2048 --d-model 1280 --vocab-size 200000 --dtype bf16 \
  --hidden-scale 1.0 --weight-scale 0.02795 \
  --chunk-sizes 4096,8192,16384,32768 --impl cpp --include-full
```

This finds the speed/memory tradeoff before we commit to a CUDA kernel shape.

## Triton CE Forward Prototype

```bash
python perf_lab/benchmarks/bench_triton_ce_forward.py \
  --tokens 32 --d-model 256 --vocab-size 8192 --dtype bf16 \
  --block-v 64 --block-d 64 --mode parallel
```

This is forward-only and intended to validate a custom-kernel shape. It is not
yet the production training path. Use `--mode parallel` for the current best
prototype; `--mode block` is kept as a simple serial reference.

## Triton Fused CE Tile Sweep

```bash
python perf_lab/benchmarks/sweep_triton_fused_ce.py \
  --tokens 128 --d-model 128 --vocab-size 4096 --dtype bf16 \
  --hidden-scale 1.0 --weight-scale 0.0884 \
  --block-ms 16,32 --block-vs 32,64,128 --block-ds 32,64 \
  --backward-modes atomic,atomic_lowmem,atomic_mixed,atomic_mixed_bf16match,partial_weight,split_hidden,split_weight \
  --include-full
```

`atomic` is the first fused backward. `atomic_lowmem` keeps the same atomic
work decomposition but accumulates gradients into parameter-dtype buffers
instead of fp32 buffers. `atomic_mixed` keeps `grad_hidden` in fp32 and only
uses the low-memory buffer for `grad_weight`. The `_bf16match` suffix rounds
Triton chunk logits to bf16 before softmax so the candidate follows the same
quantization point as PyTorch bf16 full logits. `partial_weight` computes
bounded row-group partials for `grad_weight` and reduces them in a second kernel.
`split_hidden` computes `grad_hidden` without atomics and keeps atomics only on
`grad_weight`; `split_weight` does the reverse. The split and partial modes
trade extra recomputation or temporary buffers for fewer atomic updates.

## Loss-Drift Gate

```bash
python perf_lab/benchmarks/loss_drift_ce.py \
  --steps 20 --tokens 128 --input-dim 256 --d-model 256 \
  --vocab-size 8192 --dtype bf16 \
  --impl triton_fused --triton-backward-mode atomic_lowmem \
  --block-m 32 --block-v 32 --block-d 64
```

This is the promotion gate after single-step parity. It runs two identical
synthetic optimizer loops: PyTorch full-logits CE versus the candidate CE. It
tracks loss, gradient, and parameter drift over many updates. The fused path
must stay numerically stable here before it can be tested in real training.
