"""Inference-compatibility smoke: load a checkpoint, generate N tokens, roundtrip chat.

Runs off a freshly-written ``best.pt`` or a step checkpoint to verify:

- the saved weights load back into the same architecture without key mismatch
- the model emits coherent logits shapes (no silent numerical breakage)
- greedy decoding actually produces new tokens
- the SentencePiece tokenizer can encode a prompt AND decode the continuation
- the chat-template encode/decode roundtrip is byte-exact (v1-L-001 regression)

This is the quick pre-flight you run at every milestone. It is intentionally
minimal — NOT a benchmark. For vLLM / llama.cpp compat checks see the
conversion scripts (will land in Phase 1.5 / Phase 5).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import sentencepiece as spm
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

# Auto-enable mamba_ssm CUDA kernels when CUDA is available and mamba_ssm is
# installed.  Checkpoints saved with the kernel back-end (parameter layout
# uses _impl.inner.*) cannot be loaded with the native back-end — so this
# must be set BEFORE build_model() is called.
def _maybe_enable_mamba_kernel() -> bool:
    if os.environ.get("AURALIS_USE_MAMBA_KERNEL", "") == "1":
        return True  # already set
    if not torch.cuda.is_available():
        return False
    try:
        import mamba_ssm  # noqa: F401
        os.environ["AURALIS_USE_MAMBA_KERNEL"] = "1"
        return True
    except ImportError:
        return False

_kernel_active = _maybe_enable_mamba_kernel()

from auralis.model import build_model                         # noqa: E402
from auralis.tokenizer.chat_template import build_inference_prompt  # noqa: E402


def _greedy_generate(
    model,
    sp: spm.SentencePieceProcessor,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[str, list[int]]:
    ids = sp.EncodeAsIds(prompt)
    x = torch.tensor([ids], device=device, dtype=torch.long)
    new_tokens: list[int] = []
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(input_ids=x)
            next_id = int(out["logits"][0, -1].argmax().item())
            new_tokens.append(next_id)
            x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)
    return sp.DecodeIds(new_tokens), new_tokens


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Optional .pt to load. If omitted, tests a fresh-init model.")
    p.add_argument("--tokenizer", type=Path, default=REPO / "tokenizer" / "helix_v2_tokenizer.model")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--prompt", default="Hallo, wer bist du?")
    args = p.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    print(f"device: {device}")

    # 1. Build + load
    model = build_model(args.model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.1f} M params")
    loaded_step = None
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
        # Strip torch.compile prefix (_orig_mod.) that is present when the
        # checkpoint was saved while torch.compile was active.
        state = {k.replace("_orig_mod.", ""): v for k, v in payload["model"].items()}
        miss, extra = model.load_state_dict(state, strict=False)
        if miss or extra:
            print(f"  state_dict mismatch — missing={len(miss)} extra={len(extra)}")
            print(f"    first missing: {miss[:3]}")
            print(f"    first extra  : {extra[:3]}")
            if _kernel_active:
                print("  hint: mamba_ssm kernel is active — layout uses _impl.inner.*")
            else:
                print("  hint: mamba_ssm not active — if checkpoint was saved with CUDA kernel,")
                print("        set AURALIS_USE_MAMBA_KERNEL=1 or install mamba_ssm.")
            raise SystemExit(1)
        loaded_step = payload.get("state", {}).get("step")
        print(f"  mamba backend : {'mamba_ssm' if _kernel_active else 'native'}")
        print(f"  loaded step   : {loaded_step}")
    else:
        print("  (no --checkpoint, testing fresh-init model)")

    # 2. Tokenizer sanity
    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    assert sp.GetPieceSize() == model.config.vocab_size, \
        f"vocab mismatch: tokenizer={sp.GetPieceSize()} model.cfg={model.config.vocab_size}"

    # 3. Chat-template roundtrip (v1-L-001 regression guard)
    inf_prompt = build_inference_prompt(
        [{"role": "user", "content": args.prompt}],
        default_system="Du bist Auralis.",
    )
    ids = sp.EncodeAsIds(inf_prompt)
    decoded = sp.DecodeIds(ids)
    ok = decoded == inf_prompt
    print(f"  chat-template roundtrip byte-exact: {'OK' if ok else 'FAIL'}")
    if not ok:
        print(f"    original : {inf_prompt!r}")
        print(f"    decoded  : {decoded!r}")
        raise SystemExit(2)

    # 4. Greedy generation
    print(f"  generating {args.max_new_tokens} tokens...")
    text, new_ids = _greedy_generate(model, sp, inf_prompt, args.max_new_tokens, device)
    print(f"  new token ids: {new_ids[:8]}...")
    print(f"  decoded      : {text[:120]!r}")

    # 5. Shape sanity
    with torch.no_grad():
        probe = torch.randint(0, model.config.vocab_size, (1, 8), device=device)
        out = model(input_ids=probe)
    assert out["logits"].shape == (1, 8, model.config.vocab_size), out["logits"].shape
    assert torch.isfinite(out["logits"]).all()
    print("  logits shape + finite: OK")

    print("\ninference-compat smoke: PASS")


if __name__ == "__main__":
    main()
