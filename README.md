# Helix v2 / Auralis

A **from-scratch, ~0.9B-parameter hybrid language model** — built and trained as a solo
project (with an AI as a sparring partner), with its own tokenizer, data pipeline,
evaluations, and documentation.

> **Honest status:** this is an *experimental foundation model*, not a competitor to
> large production LLMs. It is a mid-/under-trained ~1B base, released for transparency
> and as a study in solo model engineering. See `docs/PROJEKT_STAND.md` for the full
> honest project history and `docs/POSTMORTEM_messung_vs_daten.md` for the debugging
> lessons (most "the model is broken" moments turned out to be measurement, not data).

## Architecture
- **28-layer heterogeneous hybrid:** 6× Mamba-2 + 16× GLA (Gated Linear Attention) + 6× Sparse-Attention
- Pre-Norm (RMSNorm), RoPE, SwiGLU FFN, **tied 200k SentencePiece** embeddings, bf16
- d_model 1280, 10 heads × 128, ~954M parameters
- Diagrams: `docs/helix_v2_architecture.svg`, `docs/helix_mamba2_blueprint.svg`,
  `docs/auralis_system_vision_blueprint.svg`

## Vision
One broad, frozen **universal base** + knowledge/skills loaded on top as **DoRA/LoRA
adapters**. The large 200k vocab is a deliberate universal substrate (adapters change
weights, not the token table). Built to scale — 1B is the foundation, not the target.

## What works / what doesn't (measured)
- ✅ Stable training; fluent bilingual DE/EN; history & geography facts well anchored
  (rigorous contrastive-margin probe, not greedy guessing)
- ⚠️ Science facts + translation weaker; free-form decoding still raw; no
  instruction-following yet (pre-SFT)
- Knowledge profile + data strategy: `docs/datastrategie_wissensprofil.md`

## Repository layout
- `src/auralis/` — model (Mamba-2 / GLA / Sparse layers, RMSNorm, RoPE), training, eval
- `scripts/` — data pipeline, pretraining, evaluation, monitoring
- `configs/model/` — architecture configs
- `docs/` — architecture blueprints, project history, post-mortems, data strategy
- `perf_lab/` — kernel / op micro-benchmarks · `tests/` — unit/integration tests

## License
- **Code:** Apache-2.0 — see `LICENSE`
- **Model weights:** OpenRAIL-M responsible-AI license with use restrictions — see
  `MODEL_LICENSE.md` (weights released separately, e.g. on Hugging Face)

## Acknowledgements
Trained on filtered public web / encyclopedic data (FineWeb2, RedPajama, HPLT,
Wikipedia). A solo project built with an AI coding/research sparring partner.
