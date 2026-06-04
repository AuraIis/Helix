"""End-to-end smoke test of the full Phase-1 pipeline.

Runs a short training loop through ``PretrainTrainer`` and reports:

- peak VRAM on CUDA
- tokens/sec throughput
- loss trajectory (should drop after the first handful of steps)
- checkpoint save + reload roundtrip

Two data modes:

- ``--use-real-data``: reads the actual tokenized ``*.bin`` files from
  ``configs/data_paths.yaml`` (the NAS data dir). Validates the full path
  through the memmap sampler and the mix-ratio partitioning.
- default (synthetic): generates random-token .bin files in a tmpdir. Used
  by CI / offline environments.

Two precision modes:

- ``--dtype fp32`` (default on CPU)
- ``--dtype bf16`` — wraps forward in ``torch.autocast``. The real Phase-1
  run on RunPod uses bf16; this is our pre-flight check that autocast
  doesn't produce NaNs with our layer stack.
"""

from __future__ import annotations

import argparse
import gc
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

# Windows cp1252 cannot print unicode arrows / warning signs.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from auralis.model import build_model
from auralis.training.dataset import MixedDataLoader
from auralis.training.optimizer import build_optimizer, build_scheduler
from auralis.training.trainer import PretrainTrainer
from auralis.training.utils import load_yaml, set_seed


DTYPES: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def _write_synthetic_bins(data_dir: Path, vocab_size: int, tokens_per_lang: int) -> None:
    rng = np.random.default_rng(0)
    for lang in ("english", "german", "code"):
        arr = rng.integers(0, vocab_size, size=tokens_per_lang, dtype=np.uint32)
        arr.tofile(data_dir / f"{lang}.bin")


def _resolve_real_data_dir() -> Path:
    """Read data_paths.yaml to find the tokenized directory on the NAS."""
    cfg = load_yaml(REPO / "configs" / "data_paths.yaml")
    root = Path(cfg["data_root"])
    return root / "tokenized" / "phase1"


def _make_autocast_ctx(device: torch.device, dtype: torch.dtype):
    """Return a context manager for mixed-precision forward/backward."""
    if dtype is torch.float32:
        return nullcontext()
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    # autocast on CPU supports bf16
    return torch.autocast(device_type="cpu", dtype=dtype)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "training" / "phase1_pretrain.yaml")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="fp32", choices=list(DTYPES))
    parser.add_argument("--model-config", type=Path,
                        default=REPO / "configs" / "model" / "helix_v2_100m.yaml")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-length", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=None,
                        help="Override scheduler.warmup_steps (default: take from config).")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override optimizer.lr (default: from config).")
    parser.add_argument("--use-real-data", action="store_true",
                        help="Read actual tokenized .bin files from the NAS.")
    parser.add_argument("--val-split-bytes", type=int, default=0,
                        help="If > 0, reserve a tail slice per .bin as held-out val; "
                             "the trainer will then compute val_loss mid-run.")
    parser.add_argument("--eval-every", type=int, default=0,
                        help="Run evaluation every N steps (requires --val-split-bytes).")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable torch gradient checkpointing on the model.")
    args = parser.parse_args()

    # ---- Resolve device ----
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA requested but not available")

    dtype = DTYPES[args.dtype]
    print(f"device={device} dtype={args.dtype}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        props = torch.cuda.get_device_properties(0)
        print(f"gpu: {props.name} ({props.total_memory/1e9:.1f} GB VRAM)")

    set_seed(0)

    # ---- Build model ----
    t_build_0 = time.time()
    model = build_model(args.model_config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.1f} M params from {args.model_config.name}")
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print(f"  gradient_checkpointing: ENABLED (is_gradient_checkpointing="
              f"{model.is_gradient_checkpointing})")
    model = model.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"  build+move: {time.time() - t_build_0:.1f}s")

    full_cfg = load_yaml(args.config)

    # ---- Data ----
    tmp_ctx = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    tmp_path = Path(tmp_ctx.name) if not args.use_real_data else None

    if args.use_real_data:
        data_dir = _resolve_real_data_dir()
        if not all((data_dir / f"{lang}.bin").exists()
                   for lang in full_cfg["data"]["mix_ratios"]):
            missing = [l for l in full_cfg["data"]["mix_ratios"]
                       if not (data_dir / f"{l}.bin").exists()]
            sys.exit(f"real data requested but missing on NAS: {missing}\n  in {data_dir}")
        print(f"data: {data_dir} (real tokenized NAS data)")
        ckpt_dir = Path(tempfile.mkdtemp(prefix="ckpt_")) / "ckpt"
    else:
        assert tmp_path is not None
        data_dir = tmp_path / "data"; data_dir.mkdir()
        _write_synthetic_bins(data_dir, vocab_size=model.config.vocab_size,
                              tokens_per_lang=max(10_000, args.seq_length * 200))
        ckpt_dir = tmp_path / "ckpt"
        print(f"data: synthetic random tokens in {data_dir}")

    loader = MixedDataLoader(
        data_dir=data_dir,
        mix_ratios=full_cfg["data"]["mix_ratios"],
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        seed=0,
        split="train",
        val_split_bytes=args.val_split_bytes,
    )
    val_loader = None
    if args.val_split_bytes > 0:
        val_loader = MixedDataLoader(
            data_dir=data_dir,
            mix_ratios=full_cfg["data"]["mix_ratios"],
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            seed=0,
            split="val",
            val_split_bytes=args.val_split_bytes,
        )
        print(f"  val enabled: {args.val_split_bytes/1e6:.1f} MB per lang held out")

    # ---- Optim + sched (with optional CLI overrides for fast smoke) ----
    opt_cfg = dict(full_cfg["training"]["optimizer"])
    if args.lr is not None:
        opt_cfg["lr"] = args.lr
    sched_cfg = dict(full_cfg["training"]["scheduler"])
    if args.warmup_steps is not None:
        sched_cfg["warmup_steps"] = args.warmup_steps
    opt = build_optimizer(model, opt_cfg)
    sched = build_scheduler(opt, sched_cfg, total_steps=args.steps)

    # ---- Custom run loop (wraps fwd+bwd in autocast for bf16/fp16) ----
    eval_every = args.eval_every if args.eval_every > 0 else 10**9
    run_cfg = {
        "data": {"seq_length": args.seq_length},
        "training": {
            "batch_size_per_device": args.batch_size,
            "gradient_accumulation": args.grad_accum,
            "gradient_clip_norm": 1.0,
            "total_steps": args.steps,
            "dtype": args.dtype,                       # trainer reads this for autocast
        },
        "logging": {"log_every": max(1, args.steps // 10),
                    "eval_every": eval_every, "save_every": args.steps},
        "checkpointing": {"output_dir": str(ckpt_dir), "save_last_n": 1},
        "evaluation": {"max_val_batches": 4},
    }

    trainer = PretrainTrainer(
        model=model, optimizer=opt, scheduler=sched,
        dataloader=loader, val_dataloader=val_loader,
        config=run_cfg, device=device,
    )
    print(f"  trainer amp_dtype: {trainer._amp_dtype}")

    # Capture metrics (trainer handles autocast itself now).
    tokens_per_step = args.batch_size * args.grad_accum * args.seq_length
    losses: list[float] = []
    val_losses: list[float] = []
    original_log = trainer.log

    def capture(metrics, step):
        if "train/loss" in metrics:
            losses.append(metrics["train/loss"])
        if "eval/val_loss" in metrics:
            val_losses.append(metrics["eval/val_loss"])
        original_log(metrics, step)
    trainer.log = capture

    peak_vram_gb = 0.0

    t0 = time.time()
    state = trainer.train()
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    total_dt = time.time() - t0
    tokens_total = tokens_per_step * args.steps
    tok_per_sec = tokens_total / max(total_dt, 1e-6)

    # Checkpoint roundtrip
    checkpoints = list(ckpt_dir.glob("step_*.pt"))
    assert checkpoints, "no checkpoint written"
    trainer.load_checkpoint(checkpoints[0])
    assert trainer.state.step == args.steps

    # ---- Report ----
    print()
    print("=" * 60)
    print("  GPU SMOKE TEST REPORT")
    print("=" * 60)
    print(f"  device            : {device}")
    print(f"  dtype             : {args.dtype}")
    print(f"  model params      : {n_params/1e6:.1f} M")
    print(f"  steps             : {args.steps}")
    print(f"  batch × seq       : {args.batch_size} × {args.seq_length}"
          f" (grad-accum {args.grad_accum})")
    print(f"  tokens total      : {tokens_total:,}")
    print(f"  wall time         : {total_dt:.1f} s")
    print(f"  tokens / second   : {tok_per_sec:,.0f}")
    if device.type == "cuda":
        print(f"  peak VRAM         : {peak_vram_gb:.2f} GB")
    print(f"  loss first        : {losses[0]:.4f}")
    print(f"  loss last         : {losses[-1]:.4f}")
    drop = losses[0] - losses[-1]
    verdict = "✓ learning" if drop > 0.05 else "⚠ loss flat" if drop > -0.05 else "✗ loss rising"
    print(f"  loss delta        : {drop:+.4f}   {verdict}")
    if val_losses:
        print(f"  val_loss observed : {val_losses}")
        print(f"  best_val_loss     : {trainer.state.best_val_loss:.4f}")
    elif args.val_split_bytes > 0:
        print(f"  val_loss          : (none — no eval step triggered within {args.steps} steps)")
    print(f"  checkpoint        : {checkpoints[0]}  (reloaded OK)")
    print("=" * 60)

    # Clean up memmap handles before tempdir removal (Windows)
    del loader, trainer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if tmp_ctx is not None:
        tmp_ctx.cleanup()


if __name__ == "__main__":
    main()
