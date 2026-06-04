#!/usr/bin/env python3
"""Run an adaptive curriculum training session.

Wires the existing Helix pieces (model, optimizer, data) to the adaptive
framework (curriculum controller + learning monitor). Run inside the
``auralis-blackwell`` container where torch/CUDA are available.

Example:
    python scripts/train/adaptive_curriculum.py \
        --model-config configs/model/helix_v2_1b.yaml \
        --curriculum   configs/curriculum/helix_1b_curriculum_v1.yaml \
        --output-dir   runs/adaptive_1b_v1 \
        --batch-size 8 --seq-length 2048 --grad-accum 4 --max-steps 200000

The curriculum decides on its own when to move from raw text to formatted/prompt
data, and stops early if retention regresses. Watch the trace at
``<output-dir>/learning_trace.jsonl``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-config", type=Path, required=True)
    ap.add_argument("--curriculum", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, default=None,
                    help="SentencePiece model; defaults to the path in the model config.")
    ap.add_argument("--probes", type=Path, default=None,
                    help="Margin-probe YAML; defaults to the built-in probe set.")
    ap.add_argument("--frozen-gate", type=Path, default=None,
                    help="Optional frozen target/retention gate YAML to run live during eval.")
    ap.add_argument("--frozen-gate-max-new-tokens", type=int, default=64)
    ap.add_argument("--frozen-gate-every-evals", type=int, default=1,
                    help="Run expensive frozen free-form gate every N monitor evals.")
    # Per-language bits-per-byte logging (fair cross-language loss metric).
    ap.add_argument("--bpb-val-data-dir", type=Path, default=None,
                    help="Tokenized .bin dir for per-language val bpb. Enables bpb logging.")
    ap.add_argument("--bpb-langs", default="english,german",
                    help="Comma-separated languages for bpb (need <lang>.bin in the val dir).")
    ap.add_argument("--bpb-val-split-bytes", type=int, default=8_000_000)
    ap.add_argument("--bpb-batches", type=int, default=20)
    ap.add_argument("--bpb-tokens-per-byte", default=None,
                    help="Optional 'english=0.199,german=0.2338' to skip on-the-fly measurement.")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-length", type=int, default=2048)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=200_000)
    ap.add_argument("--lr", type=float, default=3.0e-4)
    ap.add_argument("--warmup-steps", type=int, default=2000)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    import torch

    from auralis.model.helix_model import build_model
    from auralis.training.optimizer import build_optimizer, build_scheduler
    from auralis.adaptive.adapters import ModelAdapter, TokenizerAdapter
    from auralis.adaptive.bpb import LanguageBpbEvaluator, combine_extra_metrics
    from auralis.adaptive.controller import CurriculumController
    from auralis.adaptive.frozen_gate import FrozenGateLiveEvaluator
    from auralis.adaptive.monitor import LearningMonitor
    from auralis.adaptive.probes import DEFAULT_PROBES, load_margin_probes
    from auralis.adaptive.stages import CurriculumSpec
    from auralis.adaptive.trainer import AdaptiveCurriculumTrainer, TrainerConfig

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- model ---
    model = build_model(args.model_config)
    model.to(args.device)
    print(f"model params: {model.count_parameters()/1e6:.1f}M")

    # --- tokenizer ---
    tok_path = args.tokenizer or model.config.tokenizer_path
    if tok_path is None:
        raise SystemExit("no tokenizer path (pass --tokenizer or set it in the model config)")
    tokenizer = TokenizerAdapter(tok_path)

    # --- optimizer / scheduler ---
    optimizer = build_optimizer(model, {
        "name": "adamw", "lr": args.lr, "betas": [0.9, 0.95],
        "weight_decay": args.weight_decay, "eps": 1e-8, "fused": True,
    })
    scheduler = build_scheduler(optimizer, {
        "type": "cosine", "warmup_steps": args.warmup_steps, "min_lr_ratio": 0.1,
    }, total_steps=args.max_steps)

    # --- curriculum / probes ---
    spec = CurriculumSpec.from_yaml(args.curriculum)
    probes = load_margin_probes(args.probes) if args.probes else DEFAULT_PROBES
    print(f"curriculum '{spec.name}': {len(spec.stages)} stages, {len(probes)} margin probes")

    # --- adaptive stack ---
    model_adapter = ModelAdapter(model, device=args.device, autocast_dtype=torch.bfloat16)
    frozen_extra_metrics = None
    if args.frozen_gate:
        frozen_extra_metrics = FrozenGateLiveEvaluator(
            model_adapter,
            tokenizer,
            args.frozen_gate,
            max_new_tokens=args.frozen_gate_max_new_tokens,
            every_n_evals=args.frozen_gate_every_evals,
            trace_path=args.output_dir / "frozen_gate_trace.jsonl",
        )
        print(
            f"frozen gate live metrics: {args.frozen_gate} "
            f"(every {args.frozen_gate_every_evals} evals)"
        )

    bpb_eval = None
    if args.bpb_val_data_dir:
        tpb = None
        if args.bpb_tokens_per_byte:
            tpb = {kv.split("=")[0]: float(kv.split("=")[1])
                   for kv in args.bpb_tokens_per_byte.split(",") if "=" in kv}
        bpb_eval = LanguageBpbEvaluator(
            model_adapter, tokenizer, args.bpb_val_data_dir,
            [l.strip() for l in args.bpb_langs.split(",") if l.strip()],
            seq_length=args.seq_length, val_split_bytes=args.bpb_val_split_bytes,
            batch_size=args.batch_size, batches=args.bpb_batches, tokens_per_byte=tpb,
        )
        print(f"per-language bpb logging: {args.bpb_langs} from {args.bpb_val_data_dir}")

    controller = CurriculumController(spec)
    monitor = LearningMonitor(
        model_adapter, tokenizer, probes,
        trace_path=args.output_dir / "learning_trace.jsonl",
        use_wandb=args.wandb,
        extra_metrics_fn=combine_extra_metrics(frozen_extra_metrics, bpb_eval),
    )
    trainer = AdaptiveCurriculumTrainer(
        model_adapter, optimizer, scheduler, tokenizer, spec, controller, monitor,
        TrainerConfig(
            batch_size=args.batch_size, seq_length=args.seq_length,
            grad_accum=args.grad_accum, max_grad_norm=args.max_grad_norm,
            max_steps=args.max_steps, checkpoint_dir=str(args.output_dir / "checkpoints"),
            gradient_checkpointing=args.grad_checkpointing,
        ),
    )

    summary = trainer.run()
    print(f"\n=== run finished: {summary.status} after {summary.steps} steps "
          f"(final stage: {summary.final_stage}) ===")
    import json
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if summary.status in ("done", "max_steps") else 1


if __name__ == "__main__":
    raise SystemExit(main())
