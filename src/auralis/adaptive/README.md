# Auralis Adaptive Curriculum Trainer

An additive layer on top of the existing Helix trainer that fixes the exact
failure mode we hit at 500M: **you could not tell, during training, whether the
model was really learning the target facts or just lowering loss.** It adds two
things a plain trainer cannot:

1. **Learning observability** — every eval step it measures, per concept, the
   teacher-forced margin between the *correct* and the *wrong* continuation
   (`P(correct)` vs `P(wrong)`), a deterministic canary loss, and per-split
   pass-rates. Loss going down ≠ learning the fact; the margin says the truth.
2. **Adaptive curriculum** — the run is split into ordered stages (raw text →
   formatted/prompt → contrastive repair). The controller advances **on its
   own** when a stage is mastered or has plateaued, and **stops** when a
   retention metric regresses.

## Why it is built this way

The decision logic is **pure Python and torch-free** (`signals.py`,
`controller.py`, `stages.py`) so it is unit-tested without a GPU
(`tests/adaptive/`, 15 tests). Everything that touches torch is isolated in
`scoring.py`, `adapters.py`, `monitor.py`, `trainer.py`.

```
signals.py     plateau / stability / regression / trend  (pure)
stages.py      Stage + CurriculumSpec + YAML loader        (pure)
controller.py  the state machine: continue/advance/hold/stop (pure)
probes.py      MarginProbe (prompt + correct + wrong)       (pure)
scoring.py     teacher-forced margin / canary loss / greedy decode (torch)
adapters.py    tokenizer / model / stage data loaders       (torch)
monitor.py     compute metrics, write trace, optional W&B   (torch)
trainer.py     the loop                                     (torch)
```

## How "is it learning?" is measured

For a probe `("Die heutige Hauptstadt Deutschlands ist", " Berlin.", " Bonn.")`
the monitor computes `margin = NLL(" Bonn.") - NLL(" Berlin.")`. Positive and
growing = the model increasingly prefers the correct fact. Aggregated:

- `target_pass` / `retention_pass` — fraction of probes with positive margin
- `target_margin_mean` / `retention_margin_mean` — mean nats margin
- `margin_<family>` — per concept (capital, photo, faust, honesty, ...)
- `canary_loss` / `neg_canary_loss` — denoised fixed-batch loss

These are written to `<output-dir>/learning_trace.jsonl` (one line per eval, with
per-probe margins) so you can plot a per-concept "neuro-map" over time.

## Adaptive stages

See `configs/curriculum/helix_1b_curriculum_v1.yaml`. Each stage declares its
data, a **mastery** criterion (`stable_above` / `plateau` / `either`), a
**guard** (what must not regress), and `min/max_steps`. The controller:

- holds the minimum dwell time, then
- advances on mastery **or** plateau (whichever comes first), and
- stops/holds immediately if the guard metric regresses below its recent peak.

This is the "first lots of text, then tags/format" idea: stage 1 is raw text
(mastery = the facts stop getting more confident), stage 2 switches to formatted
chat data with assistant-only loss, stage 3 is targeted contrastive repair.

## Run it (inside the container)

```bash
python scripts/train/adaptive_curriculum.py \
  --model-config configs/model/helix_v2_1b.yaml \
  --curriculum   configs/curriculum/helix_1b_curriculum_v1.yaml \
  --probes       eval/adaptive_margin_probes_v1.yaml \
  --frozen-gate  eval/sft_response_frozen_target_retention_v2.yaml \
  --output-dir   runs/adaptive_1b_v1 \
  --batch-size 8 --seq-length 2048 --grad-accum 4 --max-steps 200000
```

With `--frozen-gate`, each eval step also runs the never-train frozen
target/retention response gate through greedy generation. It writes:

- `<output-dir>/frozen_gate_trace.jsonl`
- monitor metrics `frozen_target_pass`, `frozen_retention_pass`,
  `frozen_target_failures`, `frozen_retention_failures`, `frozen_promotable`

Use these as a live promotion/retention guard. The margin probes remain
training telemetry; the frozen gate remains the hard promotion contract.

## Status / honest caveats

- The **decision brain is tested** (`python -m unittest discover -s tests/adaptive`).
- The frozen-gate bridge was syntax-checked and smoke-tested in the container on
  a 500M checkpoint with a tiny generation budget. That validates wiring only;
  real scores require the normal max-token budget.
- **Thresholds in the curriculum YAML are placeholders.** Calibrate them by first
  running the monitor on the 500M baseline checkpoints to learn what "mastered"
  looks like in nats / pass-rate for this probe set. The same probe file then
  bridges 500M → 1B comparably.
- Margin probes are **training telemetry**, deliberately separate from the frozen
  promotion gate (`eval/sft_response_frozen_target_retention_v2.yaml`). Keep them
  paraphrased away from training data.
- Single-process trainer (no FSDP/TP yet). For multi-GPU 1B, the data loader
  needs rank-aware sharding first (see the earlier review note on `dataset.py`).
```
