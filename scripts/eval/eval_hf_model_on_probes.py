#!/usr/bin/env python3
"""Run a known-good HF model (e.g. Qwen2.5) through OUR probe set + scorer.

Diagnostic question: are our capability probes / eval harness sane, or are they
too harsh? We score a strong small model with the EXACT same semantic scorer
used on the Helix model (scripts/eval/frozen_response_gate.py):

  - HF model passes most probes  -> probes/eval are fine; our model is just
    under-trained (reassuring).
  - HF model also fails a lot     -> the probes are too strict / buggy; we have
    been measuring the wrong thing.

This validates the EVAL, not our training. Run in the container (needs
transformers + HF download access):

    python scripts/eval/eval_hf_model_on_probes.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --probes eval/sft_response_frozen_target_retention_v2.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.frozen_response_gate import evaluate_answer, load_probes  # noqa: E402

SYSTEM_DE = (
    "Du bist ein hilfreicher Assistent. Antworte korrekt, knapp und auf Deutsch. "
    "Wenn etwas erfunden oder unbekannt ist, sage das ehrlich."
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--probes", type=Path,
                    default=REPO_ROOT / "eval/sft_response_frozen_target_retention_v2.yaml")
    ap.add_argument("--output-json", type=Path, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--system", default=SYSTEM_DE)
    ap.add_argument("--no-system", action="store_true", help="omit the system prompt")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if (args.device.startswith("cuda") and torch.cuda.is_available()) else torch.float32
    print(f"loading {args.model} ({dtype}) ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(args.device).eval()
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    probes = load_probes(args.probes)
    print(f"{len(probes)} probes from {args.probes}")

    def generate(prompt: str) -> str:
        messages = []
        if not args.no_system and args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": prompt})
        try:
            text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        except Exception:
            text = prompt
        enc = tok(text, return_tensors="pt").to(args.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        gen = out[0][enc["input_ids"].shape[1]:]
        return tok.decode(gen, skip_special_tokens=True).strip()

    results, scored = [], []
    for p in probes:
        ans = generate(p.prompt)
        r = evaluate_answer(p, ans)
        scored.append(r)
        results.append({"id": p.id, "split": p.split, "answer": ans,
                        "semantic_score": r["semantic_score"], "issues": r["issues"]})
        mark = "ok " if r["semantic_score"] == 1.0 else "FAIL"
        print(f"  [{mark}] {p.split:9s} {p.id}: {ans[:80]!r}")

    def rate(split: str) -> tuple[int, int]:
        rows = [r for r in scored if r["split"] == split]
        return sum(1 for r in rows if r["semantic_score"] == 1.0), len(rows)

    tp, tn = rate("target")
    rp, rn = rate("retention")
    print(f"\n=== {args.model} on {args.probes.name} ===")
    print(f"target:    {tp}/{tn}")
    print(f"retention: {rp}/{rn}")
    print("Reading: high pass = probes are sane and answerable -> our model is just")
    print("under-trained. Low pass even here = the probes are too strict / buggy.")

    out_path = args.output_json or (REPO_ROOT / "eval/results" /
                                    f"hf_probe_{re.sub(r'[^A-Za-z0-9]', '_', args.model)}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"model": args.model, "probes": str(args.probes),
         "target_pass": [tp, tn], "retention_pass": [rp, rn], "results": results},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
