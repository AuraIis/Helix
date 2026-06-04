# Auralis Custom Kernel Plan

## Compatibility Rule

Current checkpoints stay loadable as long as parameter names, tensor shapes,
tokenizer vocabulary, and math semantics stay equivalent. Kernel changes are
safe when they only replace the implementation of the same operation.

Safe with existing checkpoints:

- fused RMSNorm with identical formula
- fused residual/gate blocks with identical parameter usage
- faster cross-entropy that consumes the same logits/labels semantics
- CUDA Graph replay of the same training step
- dataloader/token sampler changes that feed the same token stream format

Not safe without retraining:

- changing layer count, hidden size, head count, or vocab size
- switching Mamba backend when parameter layout differs
- changing tokenizer or special-token ids
- changing loss definition, shift direction, or label masking semantics

## Priority

1. **Benchmark first.** Keep synthetic and checkpoint-backed benchmarks for
   every optimization candidate.
2. **RMSNorm prototype.** Low risk, small kernel surface, easy parity tests.
3. **Chunked/fused linear cross-entropy.** Highest likely gain because the
   200k vocab projection and CE materialize very large logits.
4. **CUDA Graph training loop.** Medium gain, but only after graph-safe batch
   allocation and optimizer behavior are understood.
5. **GLA/Mamba kernels.** Potentially large but highest correctness risk.
   Start by profiling existing FLA/mamba_ssm operators before rewriting.
6. **Dataloader/token sampler.** Only if profiling shows CPU/disk stalls.
   Current production run is GPU-bound, so this is a later step.

## Promotion Gates

Each candidate must pass:

- numerical parity test against PyTorch/reference implementation
- backward-gradient parity when trainable parameters are involved
- no memory leaks across repeated runs
- speedup on the same GPU, same shape, same dtype
- checkpoint load test
- short synthetic training stability test

Only after these gates should a kernel be wired into `src/auralis`.

