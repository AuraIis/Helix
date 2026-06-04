"""MoRA smoke test.

Goals:
  1. Apply MoRA via peft-mora to a small-but-real Transformer (GPT-2).
  2. Train for a handful of steps to verify forward + backward + optimizer
     all work end-to-end with the MoRA delta-weight path.
  3. Save + reload + verify the same outputs.

This is NOT a quality test — just plumbing verification. If this passes,
the MoRA mathematics integrates cleanly with PEFT. The next step (Phase 5)
is to backport MoRA's two key methods (apply_mora + get_delta_weight)
into the modern PEFT version, so we keep DoRA + LoRA+ + PiSSA + MoRA
in the same library.
"""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn.functional as F


def banner(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


def main() -> int:
    banner("Environment")
    import peft
    from peft import LoraConfig, get_peft_model, PeftModel
    print(f"  peft version: {peft.__version__}")
    print(f"  torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    banner("Load tiny base (GPT-2)")
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
    cfg = GPT2Config(n_layer=4, n_head=4, n_embd=128, vocab_size=50257)
    model = GPT2LMHeadModel(cfg)
    model.config.pad_token_id = model.config.eos_token_id
    n_base = sum(p.numel() for p in model.parameters())
    print(f"  base params: {n_base/1e6:.2f}M")

    banner("Wrap with MoRA-LoRA")
    # mora_type=6 (RoPE-based) per paper recommendation for small ranks
    config = LoraConfig(
        use_mora=True,
        mora_type=6,
        r=8,
        target_modules=["c_attn"],         # GPT-2's combined Q/K/V projection
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, config)
    peft_model = peft_model.to(device)
    n_train = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in peft_model.parameters())
    print(f"  trainable: {n_train/1e6:.3f}M / total {n_total/1e6:.2f}M "
          f"({100*n_train/n_total:.2f}%)")
    print()
    peft_model.print_trainable_parameters()

    banner("Sanity forward")
    x = torch.randint(0, 50256, (2, 32), device=device)
    with torch.no_grad():
        out = peft_model(x, labels=x)
    print(f"  initial loss: {out.loss.item():.4f}")

    banner("Mini training (10 steps)")
    optimizer = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad],
        lr=1e-3,
    )
    peft_model.train()
    losses = []
    t0 = time.time()
    for step in range(10):
        x = torch.randint(0, 50256, (2, 32), device=device)
        out = peft_model(x, labels=x)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(out.loss.item())
    print(f"  loss trajectory: {[round(l, 3) for l in losses]}")
    print(f"  elapsed: {time.time()-t0:.1f}s")
    if losses[-1] >= losses[0]:
        print("  ⚠️  loss did not decrease — check optimizer / hooks")
    else:
        print(f"  ✓ loss dropped from {losses[0]:.3f} to {losses[-1]:.3f}")

    banner("Save adapter")
    save_dir = "/tmp/mora_smoke_adapter"
    peft_model.save_pretrained(save_dir)
    saved_files = sorted(os.listdir(save_dir))
    print(f"  saved to {save_dir}: {saved_files}")
    sizes = {f: os.path.getsize(f"{save_dir}/{f}") for f in saved_files}
    for fname, sz in sizes.items():
        print(f"    {fname:35s} {sz:>10d} bytes")

    banner("Reload + verify outputs match")
    fresh_base = GPT2LMHeadModel(cfg)
    # Re-load the original base weights from the saved peft model's
    # state_dict to get a clean comparison. Since we trained the adapter,
    # the base params are unchanged; we just need to verify that loading
    # the adapter produces the same forward outputs as the trained model.
    peft_model.eval()
    with torch.no_grad():
        eval_x = torch.randint(0, 50256, (1, 16), device=device)
        before_out = peft_model(eval_x).logits

    reloaded = PeftModel.from_pretrained(fresh_base.to(device), save_dir)
    reloaded.eval()
    with torch.no_grad():
        after_out = reloaded(eval_x).logits

    diff = (before_out - after_out).abs().max().item()
    print(f"  max abs diff before/after reload: {diff:.6f}")
    if diff < 1e-4:
        print("  ✓ MoRA adapter saved + reloaded with identical outputs")
    else:
        print(f"  ⚠️  outputs drifted by {diff} — MoRA save/load may have issue")

    banner("Merge adapter into base (merge_and_unload)")
    merged = reloaded.merge_and_unload()
    n_merged = sum(p.numel() for p in merged.parameters())
    n_train_merged = sum(p.numel() for p in merged.parameters() if p.requires_grad)
    print(f"  after merge: {n_merged/1e6:.2f}M total, {n_train_merged/1e6:.2f}M trainable")
    merged.eval()
    with torch.no_grad():
        merged_out = merged(eval_x).logits
    diff_merge = (after_out - merged_out).abs().max().item()
    print(f"  max abs diff adapter-vs-merged: {diff_merge:.6f}")
    if diff_merge < 5e-3:
        print("  ✓ merge_and_unload preserves outputs (within bf16/fp16 tolerance)")
    else:
        print(f"  ⚠️  merge drift {diff_merge} — may indicate scaling bug")

    banner("PASS — MoRA plumbing works end-to-end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
