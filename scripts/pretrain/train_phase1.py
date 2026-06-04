"""CLI entry point for Phase-1 pretraining.

Runs a single-device training loop backed by :class:`PretrainTrainer`. For
multi-GPU runs on RunPod, invoke via ``torchrun`` and wrap the model with
FSDP or DeepSpeed outside this script — the module deliberately stays
single-process so the smoke-test path (CPU) and the production path share
the same code.

What this wires up from ``configs/training/phase1_pretrain.yaml``:

- ``training.dtype`` → forward-pass autocast (bf16 / fp16 on GPU)
- ``training.gradient_checkpointing`` → torch checkpoint per block
- ``data.val_split_bytes`` → reserve N bytes at tail of each .bin for a
  held-out validation loader (never overlapping with train samples)
- ``logging.wandb.enabled`` → init W&B run, pass its logger to the trainer
- ``checkpointing.external_backup`` → ``Trainer`` copies ckpt to NAS every N steps

Typical invocation (GPU host)::

    python scripts/pretrain/train_phase1.py \
        --config configs/training/phase1_pretrain.yaml

Resume from the last checkpoint::

    python scripts/pretrain/train_phase1.py --resume checkpoints/phase1_pretrain/step_25000.pt

Warm-start weights only from a checkpoint while starting a fresh optimizer/run::

    python scripts/pretrain/train_phase1.py --warm-start checkpoints/phase1_pretrain/best.pt

Dry-run (preflight only, no weights loaded)::

    python scripts/pretrain/train_phase1.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from auralis.model import build_model
from auralis.model.backend_info import describe_model_backends, format_backend_summary
from auralis.training.dataset import MixedDataLoader
from auralis.training.health import HealthStop
from auralis.training.optimizer import build_optimizer, build_scheduler
from auralis.training.run_report import write_end_manifest, write_start_manifest
from auralis.training.trainer import PretrainTrainer
from auralis.training.utils import (
    apply_gradient_checkpointing,
    load_yaml,
    preflight_check,
    resolve_gradient_checkpointing,
    set_seed,
)


def _resolve_device(cfg_device: str, override: str | None) -> torch.device:
    wanted = override or cfg_device
    if wanted == "cuda" and not torch.cuda.is_available():
        print("warn: cuda requested but not available, falling back to cpu", file=sys.stderr)
        wanted = "cpu"
    return torch.device(wanted)


def _setup_distributed() -> tuple[bool, int, int, int]:
    """Initialise the process group for multi-GPU runs (launched via torchrun).

    No-op when launched single-process (WORLD_SIZE unset or 1) — the single-GPU
    path stays byte-identical. Returns (is_distributed, rank, world_size, local_rank).
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise SystemExit("multi-GPU (WORLD_SIZE>1) requires CUDA")
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        if rank == 0:
            print(f"distributed: nccl backend, world_size={world_size}", flush=True)
    return world_size > 1, rank, world_size, local_rank


def _maybe_init_wandb(config: dict, args: argparse.Namespace):
    """Return (logger_callable, run) — logger is a no-op when W&B is off."""
    wandb_cfg = (config.get("logging") or {}).get("wandb") or {}
    if not wandb_cfg.get("enabled") or args.no_wandb:
        return (lambda _m, _s: None), None
    try:
        import wandb  # type: ignore
    except ImportError:
        print("warn: wandb not installed, skipping W&B logger", file=sys.stderr)
        return (lambda _m, _s: None), None

    run = wandb.init(
        project=wandb_cfg.get("project", "auralis-v2"),
        name=config.get("experiment", {}).get("name"),
        tags=list(wandb_cfg.get("tags", []) or []),
        config=config,
    )
    return (lambda metrics, step: wandb.log(metrics, step=step)), run


def _configure_kernel_env(config: dict, device: torch.device) -> None:
    """Enable optional CUDA kernels before model construction.

    Helix layer implementations intentionally use env flags so the same code
    can run on CPU-only test machines. Production configs can opt in here
    without relying on ad-hoc shell exports.
    """
    if device.type != "cuda":
        return
    kcfg = (config.get("training") or {}).get("kernels") or {}
    enabled = bool(kcfg.get("enabled", False))
    os.environ["AURALIS_USE_MAMBA_KERNEL"] = "1" if enabled and kcfg.get("mamba", True) else "0"
    os.environ["AURALIS_USE_GLA_KERNEL"] = "1" if enabled and kcfg.get("gla", True) else "0"
    os.environ["AURALIS_USE_FLASH_ATTN"] = "1" if enabled and kcfg.get("flash_attention", True) else "0"


def _configure_torch_runtime(config: dict, device: torch.device) -> None:
    tcfg = config.get("training") or {}
    precision = tcfg.get("matmul_precision")
    if precision:
        torch.set_float32_matmul_precision(str(precision))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "training" / "phase1_pretrain.yaml")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--init-weights", type=Path, default=None,
                        help="Deprecated alias for --warm-start.")
    parser.add_argument("--warm-start", type=Path, default=None,
                        help="Load model weights from a checkpoint, but start fresh optimizer/scheduler/state.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default=None, help="Override config.training.device")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Skip wandb.init even if config.logging.wandb.enabled.")
    args = parser.parse_args()
    warm_start = args.warm_start or args.init_weights
    if args.resume and warm_start:
        raise SystemExit("--resume and --warm-start/--init-weights are mutually exclusive")

    config = load_yaml(args.config)
    set_seed(42)

    is_distributed, rank, world_size, local_rank = _setup_distributed()
    if is_distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = _resolve_device(config["training"]["device"], args.device)
    _configure_torch_runtime(config, device)
    _configure_kernel_env(config, device)

    preflight_check(
        data_dir=Path(config["data"]["data_dir"]),
        required_data_files=[f"{lang}.bin" for lang in config["data"]["mix_ratios"]],
        checkpoint_dir=Path(config["checkpointing"]["output_dir"]),
        required_free_gb=50.0,
        require_cuda=(device.type == "cuda"),
    )

    if args.dry_run:
        print("preflight ok — exiting (--dry-run)")
        return

    # ---- Model ----
    print(f"building model from {config['model']['config_path']}")
    model = build_model(REPO / config["model"]["config_path"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params/1e9:.2f} B ({n_params/1e6:.1f} M)")

    # Gradient checkpointing defaults to the model config, but an explicit
    # training.gradient_checkpointing true/false must override it either way.
    gc_flag = resolve_gradient_checkpointing(model, config.get("training", {}) or {})
    apply_gradient_checkpointing(model, gc_flag)
    if gc_flag:
        print("  gradient_checkpointing: ENABLED")
    else:
        print("  gradient_checkpointing: disabled")

    # torch.compile gate
    if config["training"].get("torch_compile") and not args.no_compile and device.type == "cuda":
        compile_mode = config["training"].get("torch_compile_mode")
        mode_suffix = f" mode={compile_mode}" if compile_mode else ""
        print(f"  torch.compile: compiling model{mode_suffix}…")
        model = torch.compile(model, mode=compile_mode) if compile_mode else torch.compile(model)

    # Data-parallel wrap AFTER compile: each rank trains its own replica and
    # gradients all-reduce on backward. Single-process runs skip this entirely,
    # so the single-GPU code path is unchanged.
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        if rank == 0:
            print(f"  DistributedDataParallel: device_ids=[{local_rank}], world_size={world_size}")

    # dtype announce
    dtype_str = str(config["training"].get("dtype", "fp32"))
    print(f"  autocast dtype: {dtype_str}")

    # ---- Data ----
    dcfg = config["data"]
    val_split_bytes = int(dcfg.get("val_split_bytes", 0))
    # rank / world_size already resolved by _setup_distributed() above.
    train_loader = MixedDataLoader(
        data_dir=dcfg["data_dir"],
        mix_ratios=dcfg["mix_ratios"],
        batch_size=config["training"]["batch_size_per_device"],
        seq_length=dcfg["seq_length"],
        seed=int(dcfg.get("dataloader_seed", 42)),
        split="train",
        val_split_bytes=val_split_bytes,
        rank=rank,
        world_size=world_size,
    )
    print(f"  train expected rows/batch per language: {train_loader.rows_per_language}")
    if world_size > 1:
        print(f"  dataloader sharding: rank {rank}/{world_size}")

    val_loader = None
    if val_split_bytes > 0:
        val_loader = MixedDataLoader(
            data_dir=dcfg["data_dir"],
            mix_ratios=dcfg["mix_ratios"],
            batch_size=config["training"]["batch_size_per_device"],
            seq_length=dcfg["seq_length"],
            seed=int(dcfg.get("dataloader_seed", 42)),
            split="val",
            val_split_bytes=val_split_bytes,
            rank=rank,
            world_size=world_size,
        )
        print(f"  val enabled: hold-out {val_split_bytes/1e6:.1f} MB per language")
    else:
        print("  val disabled: set data.val_split_bytes > 0 to enable")

    # ---- Optim + sched ----
    optimizer = build_optimizer(model, config["training"]["optimizer"])
    scheduler = build_scheduler(
        optimizer, config["training"]["scheduler"], total_steps=int(config["training"]["total_steps"])
    )

    # ---- W&B (rank 0 only; other ranks get a no-op logger) ----
    if rank == 0:
        wandb_logger, wandb_run = _maybe_init_wandb(config, args)
    else:
        wandb_logger, wandb_run = (lambda _m, _s: None), None

    # ---- Trainer ----
    trainer = PretrainTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=train_loader,
        val_dataloader=val_loader,
        wandb_logger=wandb_logger,
        config=config,
        device=device,
        world_size=world_size,
        rank=rank,
    )
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"  resumed from {args.resume} at step {trainer.state.step}")
    elif warm_start:
        trainer.warm_start_from_checkpoint(warm_start)

    # Kernel/back-end summary (unwrap DDP so .inner is reachable)
    core_model = model.module if is_distributed else model
    backends = describe_model_backends(core_model.inner if hasattr(core_model, "inner") else core_model)
    print(format_backend_summary(backends))

    # Run-report: start (rank 0 only — avoid concurrent writes to MANIFEST.yaml)
    manifest_path = Path(config["checkpointing"]["output_dir"]) / "MANIFEST.yaml"
    if rank == 0:
        write_start_manifest(
            path=manifest_path,
            config=config,
            metadata=trainer.metadata,
            backend_summary=backends,
        )
        print(f"  manifest: {manifest_path}")

    exit_reason = "completed"
    try:
        trainer.train()
    except HealthStop as e:
        exit_reason = f"health_stop:{e}"
        print(f"  exit: {exit_reason}")
    except KeyboardInterrupt:
        exit_reason = "keyboard_interrupt"
    except Exception as e:                                     # noqa: BLE001
        exit_reason = f"{type(e).__name__}: {e}"
        raise
    finally:
        # Run-report: end (rank 0 only; always written, even on exception)
        if rank == 0:
            write_end_manifest(
                path=manifest_path,
                state=trainer.state,
                exit_reason=exit_reason,
                health_summary=trainer.health.summary(),
            )
        if wandb_run is not None:
            wandb_run.finish()
        if is_distributed:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
