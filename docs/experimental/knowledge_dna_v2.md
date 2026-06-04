# Knowledge-DNA v2

Knowledge-DNA v2 is a controlled experiment for adding compact, curated
knowledge blocks to Auralis pretraining without changing the tokenizer.

## Current status

Status: experimental, not approved for the main mix yet.

The first tiny sanity ablation in
`data/eval/knowledge_dna_v2_ablation_smoke/report.md` was a `NO-GO`: probe
loss improved for structured variants, but exact answer matches stayed at 0%
and counterfact failures stayed at 100%. That means the idea is still useful as
a research branch, but it must not be treated as proven training data.

Next acceptable step: build a larger and cleaner concept set, run a fair 100M
or 500M-sidecar ablation, and compare against the same token budget without DNA.

It uses the existing Helix v2 special tokens:

- `<memory>`
- `</memory>`
- `<recall>`
- `</recall>`
- `<|end|>`

## Why this exists

The first Auralis base run learned German-looking text, but it did not hold
stable associations for facts, counterfacts, code, math, or definitions. The
Knowledge-DNA idea tests whether a small, very clean source can make these
associations easier to learn before the expensive 500M/1B runs.

This is not a replacement for broad pretraining data. If it works, it should
enter the real mix as a small 1-3% booster.

## Variants

- `plain`: the same facts as ordinary prose.
- `dna`: structured memory blocks plus recall blocks.
- `hybrid`: prose plus memory plus harder recall/counterfact/transfer tasks.

The expected useful candidate is `hybrid`, not pure `dna`. Pure structure can be
too tag-heavy and may train the model to imitate metadata instead of answering.

## Commands

Build the sample corpus:

```powershell
python scripts\experimental\knowledge_dna_v2.py sample --output-dir data\eval\knowledge_dna_v2_smoke --tokenizer tokenizer\helix_v2_tokenizer.model
```

Run the tiny sanity ablation:

```powershell
python scripts\experimental\knowledge_dna_v2_ablation.py --dna-dir data\eval\knowledge_dna_v2_smoke --output-dir data\eval\knowledge_dna_v2_ablation_smoke --model-config configs\model\helix_v2_debug_tiny.yaml --steps 40 --seq-len 128 --repeat 24 --max-train-rows 12 --device auto
```

Run the 100M ablation when GPU time is available:

```powershell
python scripts\experimental\knowledge_dna_v2_ablation.py --dna-dir data\eval\knowledge_dna_v2_smoke --output-dir data\eval\knowledge_dna_v2_ablation_100m --model-config configs\model\helix_v2_100m.yaml --steps 80 --seq-len 128 --repeat 32 --max-train-rows 32 --device cuda
```

## Go / No-Go

Hybrid may enter the next main pretraining mix only if:

- hybrid probe loss is no worse than plain;
- hybrid has at least some real answer matches;
- hybrid counterfact failures are no worse than plain and not 100%;
- hybrid tag echo rate is at most 10%.

If those conditions are not met, Knowledge-DNA stays experimental.
