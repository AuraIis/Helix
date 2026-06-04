"""Step-0 / checkpoint eval diagnostic — the decisive "artifact vs real" test.

Loads ONE checkpoint and evaluates it WITHOUT any training, on a FIXED set of
validation sequences, under several numeric regimes. This separates a genuine
val-loss regression from a measurement artifact, and establishes the TRUE
apples-to-apples baseline for a warm-start run.

Why this exists (five-AI review consensus + our own code read):

1. The online ``_evaluate`` advances a stateful RNG, so every eval samples a
   DIFFERENT slice of the val region. Here we materialise the val batches ONCE
   and reuse the identical tokens for every mode → no sampling noise.
2. The German val window in the run is the Wikipedia *tail* of the concatenated
   bin (fineweb_v1 → fineweb_v2 → wiki), so it measures wiki-German only. Here
   we ALSO sample uniformly across the WHOLE file (= representative of the mix,
   all sources) and report both, side by side, so the wiki-vs-mix gap is visible.
3. The "1.172 baseline" was measured on a DIFFERENT (old-pool) val set, so the
   1.172 → 1.222 "regression" may be an invalid comparison. This script reports
   the warm-start source checkpoint's loss ON THE CURRENT val set → the real
   step-0 baseline to compare step-250 against.
4. ``tokens_per_byte`` (0.2338 for German) looks ~30% too high. We measure it
   from the actual bin and report bpb with the MEASURED value.
5. Kernels-on (Mamba/GLA/FlashAttn) vs kernels-off vs fp32 on the SAME
   checkpoint + SAME tokens isolates any kernel/precision numeric artifact.

Run (inside the Blackwell container, where mamba_ssm/fla are installed)::

    python scripts/pretrain/eval_diagnostic.py \
        --config configs/training/pretrain_1b_bilingual_de55_en45_foundation_warmstart_v3.yaml \
        --checkpoint /workspace/v2data/checkpoints/<warmstart_source>/best.pt \
        --n-seqs 256 \
        --out /workspace/v2data/diag/step0_eval.json

This is READ-ONLY: it never writes a checkpoint and never trains.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from auralis.adaptive.bpb import bits_per_byte  # noqa: E402
from auralis.training.utils import load_yaml  # noqa: E402

LN2 = math.log(2)

# Numeric regimes. Each rebuilds the model from scratch because the kernel
# choice is read from env flags at construction time.
MODES = {
    "kernels_on_bf16": dict(mamba=True, gla=True, flash=True, dtype="bf16"),
    "kernels_off_bf16": dict(mamba=False, gla=False, flash=False, dtype="bf16"),
    "kernels_off_fp32": dict(mamba=False, gla=False, flash=False, dtype="fp32"),
}


def _set_kernel_env(mamba: bool, gla: bool, flash: bool) -> None:
    os.environ["AURALIS_USE_MAMBA_KERNEL"] = "1" if mamba else "0"
    os.environ["AURALIS_USE_GLA_KERNEL"] = "1" if gla else "0"
    os.environ["AURALIS_USE_FLASH_ATTN"] = "1" if flash else "0"


def _sample_blocks(
    bin_path: Path,
    seq_length: int,
    n_seqs: int,
    *,
    seed: int,
    region: str,
    val_tail_tokens: int,
) -> torch.Tensor:
    """Materialise a FIXED [n_seqs, seq_length+1] int64 tensor of token windows.

    region="full" → uniform across the whole file (representative of the mix).
    region="tail" → only the last ``val_tail_tokens`` (mimics the run's val).
    Deterministic given ``seed`` — identical tokens for every numeric mode.
    """
    mm = np.memmap(bin_path, dtype=np.uint32, mode="r")
    n_tokens = int(mm.shape[0])
    span = seq_length + 1
    hi = n_tokens - span
    if hi <= 0:
        raise ValueError(f"{bin_path} too small ({n_tokens} tokens) for seq_length {seq_length}")

    rng = np.random.default_rng(seed)
    if region == "full":
        lo_bound = 0
    elif region == "tail":
        lo_bound = max(0, n_tokens - int(val_tail_tokens))
        if hi - lo_bound < span:
            lo_bound = max(0, hi - span)
    else:
        raise ValueError(region)

    starts = rng.integers(lo_bound, hi, size=n_seqs)
    blocks = np.empty((n_seqs, span), dtype=np.int64)
    for i, s in enumerate(starts):
        blocks[i] = mm[s : s + span].astype(np.int64, copy=False)
    return torch.from_numpy(blocks)


@torch.no_grad()
def _eval_loss(model, blocks: torch.Tensor, device, dtype: str, micro_bs: int) -> dict:
    """Mean per-token NLL (nats) over a fixed block set, replicating the
    trainer's input/label convention (labels = input_ids, model shifts)."""
    total_loss = 0.0
    total_main = 0.0
    n_batches = 0
    n_main = 0
    use_autocast = dtype == "bf16"
    for i in range(0, blocks.shape[0], micro_bs):
        chunk = blocks[i : i + micro_bs]
        input_ids = chunk[:, :-1].contiguous().to(device, non_blocking=True)
        labels = input_ids.clone()
        if use_autocast:
            ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            ctx = torch.autocast(device_type="cuda", enabled=False)
        with ctx:
            out = model(input_ids=input_ids, labels=labels)
        total_loss += float(out["loss"].item())
        n_batches += 1
        lm = out.get("loss_main")
        if lm is not None:
            total_main += float(lm.item())
            n_main += 1
    res = {"loss": total_loss / max(1, n_batches)}
    if n_main:
        res["loss_main"] = total_main / n_main
    return res


def _measure_tpb(bin_path: Path, tokenizer_path: Path, sample_tokens: int = 300_000) -> float:
    """tokens/byte measured from the bin (the honest value vs the config guess)."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    mm = np.memmap(bin_path, dtype=np.uint32, mode="r")
    n = int(mm.shape[0])
    take = min(sample_tokens, n)
    # sample from the middle to avoid any header/tail bias
    lo = max(0, (n - take) // 2)
    ids = [int(x) for x in mm[lo : lo + take]]
    text = sp.decode(ids)
    nbytes = len(text.encode("utf-8"))
    return take / max(1, nbytes)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True, help="Training YAML (for model/data/eval cfg).")
    ap.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint .pt to evaluate (read-only).")
    ap.add_argument("--tokenizer", type=Path, default=REPO / "tokenizer" / "helix_v2_tokenizer.model")
    ap.add_argument("--n-seqs", type=int, default=256, help="Val sequences per language per region.")
    ap.add_argument("--micro-bs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260601)
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    ap.add_argument("--regions", nargs="+", default=["full", "tail"], choices=["full", "tail"])
    ap.add_argument("--out", type=Path, default=None, help="Write results JSON here.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("eval_diagnostic needs CUDA (the full model needs the kernels on GPU).")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    config = load_yaml(args.config)
    model_cfg_path = REPO / config["model"]["config_path"]
    data_dir = Path(config["data"]["data_dir"])
    seq_length = int(config["data"]["seq_length"])
    mix = config["data"]["mix_ratios"]
    langs = [l for l, w in mix.items() if w > 0]
    val_tail_tokens = int(config["data"].get("val_split_bytes", 8_000_000)) // 4
    cfg_tpb = {str(k): float(v) for k, v in (config.get("evaluation", {}).get("tokens_per_byte") or {}).items()}

    print(f"checkpoint : {args.checkpoint}")
    print(f"config     : {args.config.name}")
    print(f"languages  : {langs}  | seq_length={seq_length} | n_seqs={args.n_seqs}/lang/region")

    # --- checkpoint metadata (the claimed baseline) ---
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    src_state = payload.get("state") or {}
    print(f"  source step={src_state.get('step', '?')}  source best_val_loss={src_state.get('best_val_loss', '?')}")

    # --- measure the honest tokens/byte + materialise fixed blocks ---
    tpb_measured: dict[str, float] = {}
    blocks: dict[tuple[str, str], torch.Tensor] = {}
    for li, lang in enumerate(langs):
        bin_path = data_dir / f"{lang}.bin"
        tpb_measured[lang] = _measure_tpb(bin_path, args.tokenizer)
        print(
            f"  tokens/byte[{lang}]: measured={tpb_measured[lang]:.4f}"
            f"  config={cfg_tpb.get(lang, float('nan')):.4f}"
        )
        for region in args.regions:
            blocks[(lang, region)] = _sample_blocks(
                bin_path, seq_length, args.n_seqs,
                seed=args.seed + li * 7919 + (0 if region == "full" else 13),
                region=region, val_tail_tokens=val_tail_tokens,
            )

    # --- import build_model + state aligner once ---
    from auralis.model import build_model
    from auralis.training.trainer import _align_model_state_for_load

    results: dict = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "source_step": src_state.get("step"),
        "source_best_val_loss": src_state.get("best_val_loss"),
        "tokens_per_byte_measured": tpb_measured,
        "tokens_per_byte_config": cfg_tpb,
        "n_seqs": args.n_seqs,
        "seq_length": seq_length,
        "modes": {},
    }

    for mode in args.modes:
        spec = MODES[mode]
        print(f"\n=== mode: {mode}  (mamba={spec['mamba']} gla={spec['gla']} flash={spec['flash']} dtype={spec['dtype']}) ===")
        _set_kernel_env(spec["mamba"], spec["gla"], spec["flash"])
        try:
            model = build_model(model_cfg_path).to(device)
            state = _align_model_state_for_load(model, payload["model"])
            missing, extra = model.load_state_dict(state, strict=True)
            if missing or extra:
                raise RuntimeError(f"ckpt mismatch: missing={missing[:3]} extra={extra[:3]}")
        except Exception as exc:
            # The fused (kernels-on) Mamba and the reference (kernels-off) Mamba
            # have DIFFERENT parameter layouts — a checkpoint saved under one is
            # not weight-compatible with the other. Skip this mode loudly rather
            # than aborting the whole diagnostic (later modes still run).
            print(f"  SKIP mode {mode}: could not load checkpoint into this model variant.")
            print(f"       reason: {type(exc).__name__}: {str(exc)[:300]}")
            results["modes"][mode] = {"skipped": True, "reason": str(exc)[:500]}
            if "model" in dir():
                del model
            torch.cuda.empty_cache()
            continue
        model.eval()

        mode_res: dict = {}
        for lang in langs:
            tpb = tpb_measured[lang]
            for region in args.regions:
                ev = _eval_loss(model, blocks[(lang, region)], device, spec["dtype"], args.micro_bs)
                loss = ev["loss"]
                bpb_m = bits_per_byte(loss, tpb)
                bpb_c = bits_per_byte(loss, cfg_tpb[lang]) if lang in cfg_tpb else float("nan")
                key = f"{lang}/{region}"
                mode_res[key] = {
                    "loss": loss,
                    "loss_main": ev.get("loss_main"),
                    "bpb_measured": bpb_m,
                    "bpb_config_tpb": bpb_c,
                }
                print(
                    f"  {key:18s} loss={loss:.4f}  bpb(meas)={bpb_m:.4f}  bpb(cfg)={bpb_c:.4f}"
                    + (f"  loss_main={ev['loss_main']:.4f}" if ev.get("loss_main") is not None else "")
                )
        results["modes"][mode] = mode_res
        del model
        torch.cuda.empty_cache()

    # --- verdict hints ---
    print("\n=== verdict hints ===")
    if "kernels_on_bf16" in results["modes"] and "kernels_off_bf16" in results["modes"]:
        on, off = results["modes"]["kernels_on_bf16"], results["modes"]["kernels_off_bf16"]
        worst = 0.0
        for k in on:
            if k in off and off[k]["loss"] > 0:
                rel = abs(on[k]["loss"] - off[k]["loss"]) / off[k]["loss"]
                worst = max(worst, rel)
        print(f"  kernels on vs off: max relative loss diff = {worst*100:.2f}%")
        print("   → <1%: no kernel artifact.  >3%: kernel numerics suspect.")
    if "kernels_off_bf16" in results["modes"] and "kernels_off_fp32" in results["modes"]:
        bf, fp = results["modes"]["kernels_off_bf16"], results["modes"]["kernels_off_fp32"]
        worst = 0.0
        for k in bf:
            if k in fp and fp[k]["loss"] > 0:
                rel = abs(bf[k]["loss"] - fp[k]["loss"]) / fp[k]["loss"]
                worst = max(worst, rel)
        print(f"  bf16 vs fp32 (kernels off): max relative loss diff = {worst*100:.2f}%")
        print("   → <1%: bf16 precision is fine.  >3%: precision suspect.")
    print("  full-vs-tail gap shows how unrepresentative the wiki-tail val is.")
    print("  compare the step-0 bpb(full,german) here against the run's step-250 bpb_german.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
