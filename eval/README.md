# Auralis Evaluation

Three complementary evaluation tracks. All feed a single results dashboard.

## Track 0 - Capability Probes (early learning curve)

Runs a small fixed set of German-first probes against every early pretraining
checkpoint. This is cheaper and more interpretable than a full benchmark while
the base model is still learning to stop producing garbage.

* Probes:  `eval/capability_probes_clean_v2.yaml`
* Runner:  `scripts/eval/run_capability_probes.py`
* Results: `eval/results/capability/<tag>.json` and `.md`

Use this from the first few checkpoints onward:

```bash
python scripts/eval/run_capability_probes.py \
    --model-config configs/model/helix_v2_mid_500m_smart.yaml \
    --checkpoint checkpoints/pretrain_mix_v4_boosted_500m/step_10000.pt \
    --tag v4_boosted_500m_step_10000
```

Current local RTX 3090 example from WSL:

```bash
cd /mnt/i/KI/Auralis_datasets/local_3090_test/AuralisV2
. ../.venv/bin/activate
export AURALIS_USE_MAMBA_KERNEL=1
python scripts/eval/run_capability_probes.py \
  --model-config configs/model/helix_v2_mid_500m_smart.yaml \
  --checkpoint ../checkpoints/pretrain_mix_v4_boosted_500m/step_10000.pt \
  --tag v4_boosted_500m_step10000_local_3090 \
  --results-dir eval/results/capability_local_3090 \
  --device cuda \
  --max-new-tokens 32
```

Track aggregate score, per-category scores, repetition ratio, forbidden
associations, and HTML/template garbage. The goal is not a final score; the
goal is a clean trend across checkpoints.

## Track 1 — Custom Baseline (project-specific)

Runs **50 hand-curated questions** against the model and scores by expected
keyword overlap. Cheap, fast (~5 min), and tells us "is the model still
producing the kinds of answers we want, in the languages we trained for?".

* Questions:  `eval/baseline_questions.yaml`
* Tiers:      `eval/baseline_tiers.yaml` (smoke → pretrain → chat → domain)
* Runner:     `scripts/eval/run_baseline.py`
* Tier-runner: `scripts/eval/run_baseline_tier.py`
* Results:    `eval/results/<tag>.json`

This is the **honesty gate** — never edit the questions to chase a score.
Append-only.

## Track 2 — Industry Benchmarks (community-comparable)

Runs **canonical HF-dataset benchmarks** (HellaSwag, ARC, GSM8K, MMLU-Pro,
BBH, HumanEval, GPQA, plus DE-specific MMLU-DE / GermanQuAD / PAWS-X / XNLI).
Slower (~30–90 min for full tier), but lets us **compare to other public
models** on identical metrics.

* Suite definition: `eval/benchmarks_v1.yaml`
* Runner:           `scripts/eval/run_benchmarks.py`
* Results:          `eval/results/benchmarks/<tag>__<benchmark>.json`

### Tier schedule (see `benchmarks_v1.yaml` for exact lists)

| Tier            | When to run               | Time budget |
|-----------------|---------------------------|-------------|
| `smoke`         | every checkpoint          | <10 min     |
| `pretrain`      | every 5k steps Phase 1+   | ~30 min     |
| `post_pretrain` | once Phase 2 done         | ~60 min     |
| `post_sft`      | once Phase 3 done         | ~90 min     |
| `frontier`      | once v3 (3B+) exists      | ~30 min     |

### Running

```bash
# Sanity check the runner without loading a model:
python scripts/eval/run_benchmarks.py --tier smoke --tag dry --dry

# Full pretrain-tier run on the current best.pt:
python scripts/eval/run_benchmarks.py \
    --tier pretrain \
    --ckpt /workspace/v2data/checkpoints/phase1_pretrain/best.pt \
    --tag step18000

# Single benchmark, override sample count:
python scripts/eval/run_benchmarks.py \
    --benchmark gsm8k --n_samples -1 \
    --ckpt path/to/ckpt.pt --tag debug
```

## Honesty principles (apply to both tracks)

1. **Never tune to the test.** Eval data is held out from training corpora.
   `scripts/data/contamination_check.py` runs as a CI gate before each new
   checkpoint is released.
2. **Always report sample size and confidence.** A 0.32 score on 200 samples
   has ~±6.5pp 95%-CI; same score on 2000 samples ~±2pp. The result JSON
   includes `n_samples` for this reason.
3. **No retroactive edits.** Once a benchmark configuration is published in
   a tagged release, its YAML config does not change. New benchmarks → new
   suite version (`benchmarks_v2.yaml`).
4. **Floors and targets are honest.** `expected_floor` matches the
   random-baseline of the metric; `expected_target` is what we genuinely
   expect a healthy 1B base to hit. We don't pretend frontier scores are
   reachable on a 1B.

## Adapter contract

Both runners take a callable interface. The single contract is in
`src/auralis/eval/adapter.py`:

```python
class GenerateAdapter:
    generate: Callable[[str, dict], str]            # free-form generation
    score_choice: Callable[[str, str], float]       # log-prob of continuation
```

Anyone wiring up a different inference backend (vLLM, llama.cpp, HF
transformers) only has to implement these two callables.

## Calibration plots

`scripts/eval/regression_dashboard.py` reads ALL JSONs under
`eval/results/` and produces:

* a per-benchmark trend line over checkpoint steps
* a calibration plot for any benchmark with `expected_floor`/`expected_target`
* a side-by-side comparison table for tagged release runs

This is the artefact that goes on the public website and into release notes.
