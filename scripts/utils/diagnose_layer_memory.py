"""Per-layer VRAM diagnostic for the Helix v2 trainer.

Wraps every HelixBlock.forward with a memory-delta logger so we can see which
specific layer index causes a memory spike. Runs only N steps then exits.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO))

import torch
from auralis.training.utils import load_yaml, set_seed
from auralis.training.dataset import MixedDataLoader
from auralis.model.helix_model import HelixBlock, build_model


def _gb(b: int) -> float:
    return b / (1024 ** 3)


def install_block_hook(model):
    blocks = [(n, m) for n, m in model.named_modules() if isinstance(m, HelixBlock)]
    print(f'  found {len(blocks)} HelixBlocks to instrument', flush=True)
    for name, blk in blocks:
        original_forward = blk.forward
        idx = blk.layer_idx
        ltype = blk.layer_config.type

        def make_wrapped(orig, idx, ltype):
            def wrapped(x, rope=None):
                torch.cuda.synchronize()
                mem_before = torch.cuda.memory_allocated()
                t0 = time.time()
                try:
                    if ltype in ('sparse_attention', 'plain_attention'):
                        out = orig(x, rope=rope)
                    else:
                        out = orig(x)
                    torch.cuda.synchronize()
                    elapsed_ms = (time.time() - t0) * 1000
                    mem_after = torch.cuda.memory_allocated()
                    mem_peak = torch.cuda.max_memory_allocated()
                    delta = _gb(mem_after - mem_before)
                    print(f'  L{idx:02d} {ltype:18s} dt={elapsed_ms:8.1f}ms '
                          f'mem={_gb(mem_after):6.2f}GB d={delta:+.3f}GB '
                          f'peak={_gb(mem_peak):6.2f}GB', flush=True)
                    torch.cuda.reset_peak_memory_stats()
                    return out
                except Exception as e:
                    print(f'  L{idx:02d} {ltype} CRASHED: {e}', flush=True)
                    raise
            return wrapped
        blk.forward = make_wrapped(original_forward, idx, ltype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=None,
                    help='override training.batch_size_per_device')
    ap.add_argument('--no-grad-ckpt', action='store_true',
                    help='disable gradient checkpointing for this run')
    args = ap.parse_args()

    print(f'=== diagnose_layer_memory ===', flush=True)
    print(f'config: {args.config}', flush=True)
    print(f'steps:  {args.steps}', flush=True)

    cfg = load_yaml(args.config)
    bs = args.batch_size if args.batch_size else cfg['training']['batch_size_per_device']
    grad_ckpt = not args.no_grad_ckpt and bool(cfg['training'].get('gradient_checkpointing', True))
    seq_len = cfg['data']['seq_length']
    print(f'batch_size={bs}  seq_length={seq_len}  gradient_checkpointing={grad_ckpt}', flush=True)
    print(f'mix_ratios={cfg["data"]["mix_ratios"]}', flush=True)

    set_seed(cfg['data'].get('dataloader_seed', 42))
    torch.cuda.empty_cache()
    print(f'GPU pre-build: {_gb(torch.cuda.memory_allocated()):.2f} GB allocated', flush=True)

    print('building model...', flush=True)
    model = build_model(REPO / cfg['model']['config_path']).to('cuda')
    if grad_ckpt:
        model.gradient_checkpointing_enable()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  parameters: {n_params/1e6:.1f} M', flush=True)
    print(f'GPU after model: {_gb(torch.cuda.memory_allocated()):.2f} GB allocated', flush=True)

    install_block_hook(model)

    loader = MixedDataLoader(
        data_dir=cfg['data']['data_dir'],
        mix_ratios=cfg['data']['mix_ratios'],
        batch_size=bs,
        seq_length=seq_len,
        seed=cfg['data'].get('dataloader_seed', 42),
        split='train',
        val_split_bytes=cfg['data'].get('val_split_bytes', 0),
    )
    print('loader ready', flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    print(f'\n=== STARTING TEST STEPS ===', flush=True)

    data_iter = iter(loader)
    for step in range(args.steps):
        print(f'\n--- step {step+1}/{args.steps} ---', flush=True)
        torch.cuda.reset_peak_memory_stats()
        batch = next(data_iter)
        batch = {k: v.to('cuda', non_blocking=True) for k, v in batch.items()}
        torch.cuda.synchronize()
        try:
            t0 = time.time()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = model(input_ids=batch['input_ids'], labels=batch['labels'])
                loss = out['loss']
            torch.cuda.synchronize()
            print(f'  fwd done in {(time.time()-t0)*1000:.0f}ms loss={loss.item():.3f} '
                  f'mem={_gb(torch.cuda.memory_allocated()):.2f}GB', flush=True)

            print(f'  >>> backward starting', flush=True)
            tb0 = time.time()
            loss.backward()
            torch.cuda.synchronize()
            print(f'  bwd done in {(time.time()-tb0)*1000:.0f}ms '
                  f'mem={_gb(torch.cuda.memory_allocated()):.2f}GB '
                  f'peak={_gb(torch.cuda.max_memory_allocated()):.2f}GB', flush=True)
            optimizer.step()
            optimizer.zero_grad()
        except Exception:
            print('  STEP CRASHED:', flush=True)
            traceback.print_exc()
            sys.exit(1)

    print('\n=== ALL STEPS DONE ===', flush=True)


if __name__ == '__main__':
    main()
