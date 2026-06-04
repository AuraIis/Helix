"""Find the largest (batch_size, seq_length) combination that fits on this GPU.

Spins up the trainer model + does ONE forward + backward + optimizer step per
candidate (batch, seq) pair, captures peak VRAM and tokens-per-second, then
reports the largest combo that succeeded plus the highest-throughput one.

Why we need this: the 250M canary fit batch=12 seq=1024. The 1B model has
~4x parameters and asymmetrically more activation memory. Guessing the right
config means losing hours when the OOM hits at step 1500 instead of step 1.
This script de-risks that in ~5 minutes.

Usage (from inside the container):
    # Sweep for the 1B model with both seq_lens:
    python scripts/utils/batch_size_sweep.py \
        --model-config configs/model/helix_v2_1b.yaml \
        --data-dir /workspace/v2data/tokenized/curated_40b \
        --batch-sizes 1 2 4 6 8 \
        --seq-lens 1024 2048

    # Quick sanity sweep for the 250M model (should match what we already know):
    python scripts/utils/batch_size_sweep.py \
        --model-config configs/model/helix_v2_250m.yaml \
        --batch-sizes 8 12 16 \
        --seq-lens 1024 2048
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))


@dataclass
class TrialResult:
    batch_size: int
    seq_length: int
    success: bool
    peak_vram_gb: float = 0.0
    forward_ms: float = 0.0
    backward_ms: float = 0.0
    step_ms: float = 0.0
    tokens_per_sec: float = 0.0
    error: str = ""
    notes: list = field(default_factory=list)


def _gb(b: int) -> float:
    return b / (1024 ** 3)


def _try_one(
    model_cfg_path: Path,
    data_dir: Path,
    batch: int,
    seq: int,
    mix: dict,
    grad_ckpt: bool,
) -> TrialResult:
    import torch

    from auralis.model.helix_model import build_model
    from auralis.training.dataset import MixedDataLoader
    from auralis.training.utils import apply_gradient_checkpointing

    res = TrialResult(batch_size=batch, seq_length=seq, success=False)
    print(f"\n--- trial: batch={batch}  seq={seq}  grad_ckpt={grad_ckpt} ---", flush=True)

    # Cleanup leftover state from previous trial
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    base_alloc = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    print(f"  base GPU alloc: {_gb(base_alloc):.2f} GB", flush=True)

    try:
        # Build model fresh; this resets all internal caches and then applies
        # the requested checkpointing mode explicitly for this trial.
        model = build_model(model_cfg_path).to("cuda")
        apply_gradient_checkpointing(model, grad_ckpt)
        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"  model: {n_params / 1e6:.1f} M params, "
            f"GPU after build: {_gb(torch.cuda.memory_allocated()):.2f} GB",
            flush=True,
        )

        loader = MixedDataLoader(
            data_dir=str(data_dir),
            mix_ratios=mix,
            batch_size=batch,
            seq_length=seq,
            seed=42,
            split="train",
            val_split_bytes=50_000_000,
        )
        data_iter = iter(loader)
        batch_dict = next(data_iter)
        batch_dict = {k: v.to("cuda", non_blocking=True) for k, v in batch_dict.items()}
        torch.cuda.synchronize()

        optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Forward
        t0 = time.time()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=batch_dict["input_ids"], labels=batch_dict["labels"])
            loss = out["loss"]
        torch.cuda.synchronize()
        res.forward_ms = (time.time() - t0) * 1000

        # Backward
        tb0 = time.time()
        loss.backward()
        torch.cuda.synchronize()
        res.backward_ms = (time.time() - tb0) * 1000

        # Optimizer
        ts0 = time.time()
        optim.step()
        optim.zero_grad()
        torch.cuda.synchronize()
        res.step_ms = res.forward_ms + res.backward_ms + (time.time() - ts0) * 1000

        peak = torch.cuda.max_memory_allocated()
        res.peak_vram_gb = _gb(peak)
        res.tokens_per_sec = (batch * seq) / (res.step_ms / 1000)
        res.success = True

        print(
            f"  OK forward={res.forward_ms:.0f}ms  backward={res.backward_ms:.0f}ms  "
            f"peak={res.peak_vram_gb:.2f}GB  tok/s={res.tokens_per_sec:.0f}",
            flush=True,
        )

    except RuntimeError as e:
        msg = str(e)
        if "out of memory" in msg.lower() or "OutOfMemoryError" in msg:
            res.error = "OOM"
            print(f"  X OOM ({msg.split(chr(10))[0][:140]})", flush=True)
        else:
            res.error = type(e).__name__ + ": " + msg.split("\n")[0][:200]
            print(f"  X {res.error}", flush=True)
            res.notes.append(traceback.format_exc()[:1000])
    except Exception as e:  # noqa: BLE001
        res.error = type(e).__name__ + ": " + str(e).split("\n")[0][:200]
        print(f"  X {res.error}", flush=True)
        res.notes.append(traceback.format_exc()[:1000])
    finally:
        try:
            del model
        except UnboundLocalError:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    return res


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model-config",
        type=Path,
        required=True,
        help="Path to e.g. configs/model/helix_v2_1b.yaml",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/workspace/v2data/tokenized/curated_40b"),
        help="Tokenized data dir for the dataloader.",
    )
    p.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 6, 8],
        help="Batch sizes to try (descending strategy: stop at first OOM).",
    )
    p.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[1024, 2048],
        help="Sequence lengths to try.",
    )
    p.add_argument(
        "--no-grad-ckpt",
        action="store_true",
        help="Disable gradient checkpointing for the sweep.",
    )
    p.add_argument(
        "--mix",
        default="english=0.70,german=0.25,code=0.05",
        help="Mix ratios as 'lang=ratio,...'; affects batch composition.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("/workspace/v2data/logs/batch_sweep_results.json"),
    )
    args = p.parse_args()

    mix = {}
    for kv in args.mix.split(","):
        k, v = kv.split("=")
        mix[k.strip()] = float(v.strip())

    print("=== batch_size_sweep ===")
    print(f"model:   {args.model_config}")
    print(f"data:    {args.data_dir}")
    print(f"batches: {args.batch_sizes}")
    print(f"seqs:    {args.seq_lens}")
    print(f"mix:     {mix}")
    print(f"grad_ckpt: {not args.no_grad_ckpt}")

    grad_ckpt = not args.no_grad_ckpt
    results: list[TrialResult] = []

    for seq in args.seq_lens:
        oomed = False
        for batch in sorted(args.batch_sizes, reverse=True):
            if oomed:
                pass
            r = _try_one(args.model_config, args.data_dir, batch, seq, mix, grad_ckpt)
            results.append(r)
            if not r.success and r.error == "OOM":
                oomed = False

    print("\n=== SUMMARY ===")
    print(
        f"{'seq':>6}  {'batch':>5}  {'peak GB':>8}  {'fwd ms':>7}  {'bwd ms':>7}  "
        f"{'step ms':>8}  {'tok/s':>10}  status"
    )
    print("-" * 88)
    successful = [r for r in results if r.success]
    for r in results:
        status = "OK" if r.success else r.error[:20]
        print(
            f"{r.seq_length:>6}  {r.batch_size:>5}  {r.peak_vram_gb:>8.2f}  "
            f"{r.forward_ms:>7.0f}  {r.backward_ms:>7.0f}  {r.step_ms:>8.0f}  "
            f"{r.tokens_per_sec:>10.0f}  {status}"
        )

    if successful:
        max_batch_per_seq: dict[int, int] = {}
        for r in successful:
            max_batch_per_seq[r.seq_length] = max(
                max_batch_per_seq.get(r.seq_length, 0),
                r.batch_size,
            )
        best_throughput = max(successful, key=lambda r: r.tokens_per_sec)

        print("\n=== RECOMMENDATIONS ===")
        print("Max batch per seq_len that fits:")
        for s, b in sorted(max_batch_per_seq.items()):
            print(f"  seq={s:>5}: batch={b}")
        print("\nHighest throughput config:")
        print(
            f"  seq={best_throughput.seq_length}  batch={best_throughput.batch_size}  "
            f"-> {best_throughput.tokens_per_sec:.0f} tok/s  "
            f"({best_throughput.peak_vram_gb:.1f} GB peak)"
        )
    else:
        print("\n!! NO config succeeded. Try smaller batches or seq_lens.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
    print(f"\nresults: {args.output}")


if __name__ == "__main__":
    main()
