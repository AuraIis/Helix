"""Data-mix ablation runner.

Generates a matrix of short pretrain configs, one per mix-ratio variant,
runs each for N steps under the same seed, and collects val_loss
(overall + per-language) into a Markdown table. Lets you answer "does
OpenMath actually help?" / "is 75/20/5 the right ratio?" with a single
overnight run instead of gut feeling.

USAGE (not part of Phase-1 launch — ablation-only; see docs below)::

    python scripts/pretrain/mix_ablation.py \
        --base-config configs/training/phase1_pretrain.yaml \
        --variants configs/ablation/mix_variants.yaml \
        --steps 500 --model-config configs/model/helix_v2_100m.yaml

Each variant writes to its own checkpoint dir; the script appends a row
per variant to ``<out>/mix_ablation_results.md``. Deterministic seed so
runs are comparable. Crashed variants are reported but do not abort the
matrix.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def _run_one(config_path: Path, steps: int, model_config: Path, variant_name: str,
             out_root: Path) -> dict:
    ckpt_dir = out_root / variant_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(REPO / "scripts" / "pretrain" / "train_phase1.py"),
        "--config", str(config_path),
        "--device", "cpu",                     # quick CPU ablation; override in prod
        "--no-wandb",
    ]
    env_steps = f"AURALIS_TOTAL_STEPS={steps}"  # not used, just documentation
    print(f"\n=== variant: {variant_name} ===")
    result = subprocess.run(cmd, capture_output=True, text=True,
                            env={**__import__("os").environ})
    return {
        "variant": variant_name,
        "return_code": result.returncode,
        "stderr_tail": result.stderr[-400:] if result.stderr else "",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-config", type=Path, required=True)
    p.add_argument("--variants", type=Path, required=True,
                   help="YAML with a 'variants' list; each variant overrides "
                        "data.mix_ratios and optionally cleaned.* entries.")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--model-config", type=Path,
                   default=REPO / "configs" / "model" / "helix_v2_100m.yaml")
    p.add_argument("--out-root", type=Path,
                   default=REPO / "checkpoints" / "ablation_mix")
    args = p.parse_args()

    base = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    variants_doc = yaml.safe_load(args.variants.read_text(encoding="utf-8"))

    results: list[dict] = []
    for v in variants_doc["variants"]:
        cfg = copy.deepcopy(base)
        cfg["model"]["config_path"] = str(args.model_config.relative_to(REPO))
        cfg["training"]["total_steps"] = int(args.steps)
        cfg["data"]["mix_ratios"] = v["mix_ratios"]
        cfg["checkpointing"]["output_dir"] = str(args.out_root / v["name"])
        cfg_path = args.out_root / f"{v['name']}.yaml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                            encoding="utf-8")
        results.append(_run_one(cfg_path, args.steps, args.model_config, v["name"], args.out_root))

    # Aggregate: read per-variant MANIFEST.yaml / sidecars for final val_loss
    md = ["# Mix Ablation Results\n",
          f"steps={args.steps}, model={args.model_config.name}\n",
          "| variant | exit | final_val_loss | per-lang | notes |",
          "|---|---|--:|---|---|"]
    for r in results:
        variant_dir = args.out_root / r["variant"]
        manifest = variant_dir / "MANIFEST.yaml"
        val_loss = "?"
        per_lang = "?"
        if manifest.is_file():
            m = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
            st = m.get("final_state", {})
            val_loss = f"{st.get('best_val_loss', float('inf')):.4f}"
        md.append(f"| {r['variant']} | {r['return_code']} | {val_loss} | {per_lang} | |")
    (args.out_root / "mix_ablation_results.md").write_text("\n".join(md) + "\n",
                                                            encoding="utf-8")
    print("wrote", args.out_root / "mix_ablation_results.md")


if __name__ == "__main__":
    main()
