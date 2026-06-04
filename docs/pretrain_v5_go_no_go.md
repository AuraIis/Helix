# Pretrain v5 Go/No-Go Gate

This gate exists to avoid repeating the v2 failure pattern: clean-looking loss,
but biased validation and noisy training data.

## Order

1. Finish `pretrain_clean_v32`.
2. Run clean-source forensics before building v5:

   ```bash
   python scripts/data/pretrain_clean_forensics_report.py \
     --clean-dir data/training/pretrain_clean_v32 \
     --output-dir data/eval/pretrain_clean_v32_forensics \
     --tokenizer tokenizer/helix_v2_tokenizer.model \
     --samples-per-source 2000
   ```

3. Build v5 only if the clean-source gate is acceptable.
4. Run `pretrain_forensics_report.py` on v5 after tokenization.
5. Compare v4 and v5 with the fixed probes in
   `eval/capability_probes_v4_v5_gate.yaml`.

## Clean-Source Gate

- Target: each main prose source should be near or below 10% sampled docs under
  100 tokens.
- Warning: any source above 20% sampled docs under 100 tokens needs either a
  stricter min-length filter or a deliberate reason to keep it.
- Hard fail: samples show repeated HTML, ChatML markers, adult/casino/shop spam,
  or URL-dense documents.
- `german_commons` is allowed to keep compact valid short prose, but v5 should
  drop hard noise, Wiki talk, table/index pages, short list-like fragments, and
  short broken sentence fragments that both start mid-sentence and lack a
  terminal boundary.
- Builder manifests must include both `observed` and `skipped` counters plus
  `skip_rates`, so each drop count has a denominator.

## v4 Canary Stop Rule

The v4 500M run is now a stability canary, not a quality decision. Stop it when
both are true:

- `pretrain_clean_v32` is complete.
- clean-source forensics is green enough to build v5.

Hard cap: stop v4 at step 20k even if v5 is not ready, unless there is a
specific reason to keep collecting stability data.

## Fixed Capability Probes

Use the same probe file for both v4 and v5 checkpoints:

```bash
python scripts/eval/run_capability_probes.py \
  --probes eval/capability_probes_v4_v5_gate.yaml \
  --model-config configs/model/helix_v2_mid_500m_smart.yaml \
  --checkpoint checkpoints/pretrain_mix_v4_boosted_500m/best.pt \
  --tag v4_500m_best_gate
```

Run the identical command against the v5 checkpoint tag when available.

## Follow-Up If v5 Long-Form Gets Worse

The v5 builder drops Wiki talk, table/index pages, hard noise, and short
list-like fragments. It still keeps longer list-like documents by default. If
the v5 long-form probe regresses against v4, run a controlled v5.x variant that
drops `list_like` documents unconditionally instead of only below 100 tokens.

## Tokenizer-v2 Backlog

If Knowledge-DNA is re-enabled, reserve these as real single-token special
tokens before the large run:

- `<memory>`
- `</memory>`
- `<recall>`
- `</recall>`
- `<|end|>`

The current tokenizer encodes them as two pieces. That is acceptable while DNA
is disabled, but not for a serious DNA/data-structure run.
