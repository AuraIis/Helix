"""PretrainTrainer: minimal but complete pretraining loop.

Scope: gradient accumulation, cosine LR schedule, grad clipping, checkpoints
with rotation, NaN detection, periodic eval. No distributed-training logic —
that is layered on via FSDP / DeepSpeed in ``scripts/pretrain/train_phase1.py``
which is what actually runs on RunPod.

All stateful configuration passes through a plain dict (from YAML) rather than
a dataclass, so future additions (e.g. MoE loss aux, MTP loss) don't need
signature changes here — only dict-key lookups.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from auralis.adaptive.bpb import bits_per_byte, bpb_gap
from auralis.training.health import (
    AlertLevel,
    HealthConfig,
    HealthMonitor,
    HealthStop,
)


_AMP_DTYPES: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
}


def _safe_log(logger: Callable[[dict[str, float], int], None]):
    """Wrap a metrics logger so a logging error never kills training."""
    def inner(metrics: dict[str, float], step: int) -> None:
        try:
            logger(metrics, step)
        except Exception as e:                                 # noqa: BLE001
            # Very noisy to print every failure; emit the type once per call.
            print(f"  warn: metrics logger failed: {type(e).__name__}: {e}", flush=True)
    return inner


def _git_sha(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _sha256_short(path: Path) -> str:
    if not path or not Path(path).is_file():
        return ""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


@dataclass
class TrainerState:
    """Pure-data training state — persisted in every checkpoint."""

    step: int = 0
    best_val_loss: float = float("inf")
    consecutive_val_increases: int = 0
    # Previous eval's val_loss — lets the guard count TRUE consecutive rises
    # (val higher than the eval before it) rather than "failed to beat best".
    # inf = no prior eval yet (first eval never counts as a rise).
    last_val_loss: float = float("inf")
    tokens_seen: int = 0
    wall_clock_seconds: float = 0.0
    alerts: list[str] = field(default_factory=list)
    # Cheap counters for post-hoc analysis of backups + logging reliability
    external_backups_ok: int = 0
    external_backups_failed: int = 0
    # Consecutive failures (resets on success). Used by the health monitor
    # to stop training before the checkpoint graveyard fills with bogus runs.
    external_backups_consecutive_failed: int = 0


@dataclass
class RunMetadata:
    """Run-level provenance captured once at Trainer construction.

    Written into every checkpoint alongside TrainerState so a reloaded
    checkpoint is self-explanatory ("which git rev, which config, which
    tokenizer file, which machine produced this").
    """

    git_sha: str = "unknown"
    config_sha16: str = ""
    config_path: str = ""
    tokenizer_sha16: str = ""
    tokenizer_path: str = ""
    hostname: str = ""
    python_version: str = ""
    torch_version: str = ""
    cuda_version: str | None = None
    gpu_name: str | None = None
    dtype: str = "fp32"


def _align_model_state_for_load(
    model: nn.Module,
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Align torch.compile ``_orig_mod.`` prefixes before loading weights."""
    model_prefixed = any(k.startswith("_orig_mod.") for k in model.state_dict())
    ckpt_prefixed = any(k.startswith("_orig_mod.") for k in state)
    if model_prefixed and not ckpt_prefixed:
        return {f"_orig_mod.{k}": v for k, v in state.items()}
    if ckpt_prefixed and not model_prefixed:
        return {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    return state


class PretrainTrainer:
    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LambdaLR,
        dataloader: Iterator[dict[str, torch.Tensor]],
        config: dict[str, Any],
        state: TrainerState | None = None,
        device: str | torch.device = "cpu",
        val_dataloader: Iterator[dict[str, torch.Tensor]] | None = None,
        wandb_logger: Callable[[dict[str, float], int], None] | None = None,
        world_size: int = 1,
        rank: int = 0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.state = state or TrainerState()
        self.device = torch.device(device)
        self.log = _safe_log(wandb_logger or (lambda _metrics, _step: None))

        # ---- Distributed (DDP) awareness; single-process when world_size == 1 ----
        self._world_size = max(1, int(world_size))
        self._rank = int(rank)
        self._is_distributed = self._world_size > 1
        self._is_main = self._rank == 0
        # Checkpoints stay DDP-agnostic: save/load the underlying module so a
        # multi-GPU checkpoint reloads single-GPU (no "module." key prefix).
        self._core_model = (
            model.module
            if isinstance(model, nn.parallel.DistributedDataParallel)
            else model
        )

        # Unpack knobs that get hit every step.
        tcfg = config["training"]
        self._total_steps = int(tcfg["total_steps"])
        self._grad_accum = int(tcfg.get("gradient_accumulation", 1))
        self._clip_norm = float(tcfg.get("gradient_clip_norm", 1.0))
        self._batch_tokens = int(
            tcfg["batch_size_per_device"] * self._grad_accum * config["data"]["seq_length"]
        )

        # AMP dtype — config.training.dtype drives the forward-pass autocast.
        # We do NOT silently fall back: picking fp16 on CPU is almost always a
        # config mistake that would later manifest as NaNs at unknown cost.
        dtype_str = str(tcfg.get("dtype", "fp32")).lower()
        if dtype_str not in _AMP_DTYPES:
            raise ValueError(f"unknown dtype {dtype_str!r}; one of {sorted(_AMP_DTYPES)}")
        self._amp_dtype = _AMP_DTYPES[dtype_str]

        # Matmul precision: "high" enables TF32 on Ampere+ GPUs for any fp32
        # matmuls that survive the bf16/fp16 autocast (e.g. the cross-entropy
        # logits cast or fp32 norm scales). Cost-free on bf16-dominant runs,
        # ~3% boost when the model has fp32 hot-spots. Off by default — opt
        # in via config.training.matmul_precision: "high" (or "highest" /
        # "medium"). See torch.set_float32_matmul_precision docs.
        matmul_prec = tcfg.get("matmul_precision")
        if matmul_prec and torch.cuda.is_available():
            torch.set_float32_matmul_precision(str(matmul_prec).lower())
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # fp16 requires a GradScaler to stay numerically stable; bf16 and fp32
        # do not. Create the scaler only when fp16 on CUDA is actually active.
        self._use_amp = self._amp_dtype != torch.float32
        self._scaler: torch.amp.GradScaler | None = None
        if self._amp_dtype == torch.float16:
            if self.device.type != "cuda":
                raise ValueError("fp16 training requires a CUDA device.")
            self._scaler = torch.amp.GradScaler("cuda")

        lcfg = config["logging"]
        self._log_every = int(lcfg.get("log_every", 10))
        self._eval_every = int(lcfg.get("eval_every", 1000))
        self._save_every = int(lcfg.get("save_every", 2500))

        ccfg = config["checkpointing"]
        self._ckpt_dir = Path(ccfg["output_dir"])
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._keep_last = int(ccfg.get("save_last_n", 3))
        # Codex P5: respect the save_best flag. Smoke / ablation configs that
        # set save_best=false were silently overridden because the trainer
        # always wrote best.pt on val improvement. Default true preserves
        # the existing Phase-1 behaviour.
        self._save_best = bool(ccfg.get("save_best", True))
        self._skip_step_save_if_checkpoint_written = bool(
            ccfg.get("skip_step_save_if_checkpoint_written", False)
        )
        self._last_checkpoint_step: int | None = None
        self._last_checkpoint_name: str | None = None
        self._external_backup = ccfg.get("external_backup") or {}

        # ---- Health monitor (auto-stop guards) ----
        mon_cfg = (config.get("monitoring") or {}).get("health") or {}
        self.health = HealthMonitor(HealthConfig(
            **{k: v for k, v in mon_cfg.items() if k in HealthConfig.__dataclass_fields__}
        ))
        # Tolerance (nats) below which a val change is treated as noise, not a
        # real improvement OR a real rise. Stops the guard tripping on the
        # normal sub-noise wobble of a converged checkpoint. 0.0 = exact.
        self._val_min_delta = float(mon_cfg.get("val_min_delta", 0.0))

        # ---- Cost tracker ----
        cost_cfg = config.get("cost") or {}
        self._cost_per_gpu_hour = float(cost_cfg.get("gpu_hourly_usd", 0.0))
        self._cost_budget = float(cost_cfg.get("budget_usd", 0.0))

        # ---- Run metadata (captured once, persisted with every ckpt) ----
        repo_root = Path(__file__).resolve().parents[3]
        self.metadata = RunMetadata(
            git_sha=str(_git_sha(repo_root)),
            config_sha16=hashlib.sha256(json.dumps(config, sort_keys=True, default=str)
                                        .encode("utf-8")).hexdigest()[:16],
            config_path=str(config.get("_source_path", "")),
            tokenizer_sha16=str(_sha256_short(Path(
                config.get("data", {}).get("tokenizer_path")
                or repo_root / "tokenizer" / "helix_v2_tokenizer.model"
            ))),
            tokenizer_path=str(config.get("data", {}).get("tokenizer_path")
                               or repo_root / "tokenizer" / "helix_v2_tokenizer.model"),
            hostname=str(socket.gethostname()),
            python_version=str(platform.python_version()),
            # torch.__version__ is a TorchVersion object; str() makes it yaml-safe.
            torch_version=str(torch.__version__),
            cuda_version=(str(torch.version.cuda) if torch.cuda.is_available() else None),
            gpu_name=(str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None),
            dtype=str(dtype_str),
        )

    def _autocast(self):
        """Context manager for mixed-precision forward/backward."""
        if not self._use_amp:
            return nullcontext()
        # torch.autocast is device-type specific.
        return torch.autocast(device_type=self.device.type, dtype=self._amp_dtype)

    # ------------------------------------------------------------------
    def train(self) -> TrainerState:
        self.model.train()
        data_iter = iter(self.dataloader)
        window_loss_sum = 0.0
        window_loss_main_sum = 0.0
        window_loss_mtp_sum = 0.0
        window_loss_main_count = 0
        window_loss_mtp_count = 0
        window_t0 = time.time()
        window_data_time = 0.0
        window_compute_time = 0.0

        while self.state.step < self._total_steps:
            loss_acc = 0.0
            self.optimizer.zero_grad(set_to_none=True)

            for micro in range(self._grad_accum):
                t_data_0 = time.time()
                batch = next(data_iter)
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                window_data_time += time.time() - t_data_0

                t_compute_0 = time.time()
                # DDP: defer the gradient all-reduce to the LAST micro-step so it
                # fires once per optimizer step, not once per accumulation micro-step.
                is_last_micro = micro == self._grad_accum - 1
                sync_ctx = (
                    self.model.no_sync()
                    if (self._is_distributed and not is_last_micro)
                    else nullcontext()
                )
                with sync_ctx:
                    with self._autocast():
                        out = self.model(input_ids=batch["input_ids"], labels=batch["labels"])
                        loss = out["loss"] / self._grad_accum
                    if not torch.isfinite(loss):
                        msg = f"non-finite loss at step {self.state.step}: {loss.item()}"
                        self.state.alerts.append(msg)
                        raise RuntimeError(msg)
                    if self._scaler is not None:
                        self._scaler.scale(loss).backward()
                    else:
                        loss.backward()
                loss_acc += loss.item()
                loss_main = out.get("loss_main")
                if loss_main is not None:
                    window_loss_main_sum += float(loss_main.detach().item())
                    window_loss_main_count += 1
                loss_mtp = out.get("loss_mtp")
                if loss_mtp is not None:
                    window_loss_mtp_sum += float(loss_mtp.detach().item())
                    window_loss_mtp_count += 1
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                window_compute_time += time.time() - t_compute_0

            # Clip + step (unscale first if fp16 + GradScaler)
            if self._scaler is not None:
                self._scaler.unscale_(self.optimizer)
            # error_if_nonfinite=True: a NaN/Inf gradient (e.g. a destabilised
            # warm-start, a bad bf16 accumulation) raises instead of silently
            # producing a non-finite update that quietly poisons the weights.
            # fp16+GradScaler is the documented exception (it expects infs and
            # skips the step), so only arm it on the bf16/fp32 path.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self._clip_norm,
                error_if_nonfinite=(self._scaler is None),
            )
            if self._scaler is not None:
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                self.optimizer.step()
            self.scheduler.step()

            self.state.step += 1
            self.state.tokens_seen += self._batch_tokens * self._world_size
            window_loss_sum += loss_acc

            if self.state.step % self._log_every == 0:
                elapsed = time.time() - window_t0
                self.state.wall_clock_seconds += elapsed
                avg_loss = window_loss_sum / self._log_every
                tps = (self._batch_tokens * self._world_size * self._log_every) / max(elapsed, 1e-9)
                lr = self.scheduler.get_last_lr()[0]
                metrics: dict[str, float] = {
                    "train/loss": avg_loss,
                    "train/grad_norm": float(grad_norm),
                    "train/lr": lr,
                    "train/tokens_per_second": tps,
                    "train/tokens_seen": self.state.tokens_seen,
                    "train/data_frac": window_data_time / max(elapsed, 1e-9),
                    "train/compute_frac": window_compute_time / max(elapsed, 1e-9),
                }
                if window_loss_main_count:
                    metrics["train/loss_main"] = window_loss_main_sum / window_loss_main_count
                if window_loss_mtp_count:
                    metrics["train/loss_mtp"] = window_loss_mtp_sum / window_loss_mtp_count
                metrics["system/step_time_ms"] = (elapsed / max(self._log_every, 1)) * 1000.0
                if self.device.type == "cuda":
                    metrics["train/vram_alloc_gb"] = torch.cuda.memory_allocated() / 1e9
                    metrics["train/vram_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
                    metrics["train/vram_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
                    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                    metrics["train/vram_total_gb"] = total_vram_gb
                    # VRAM-pressure alert (may request STOP)
                    for level, msg in self.health.observe_vram(
                        metrics["train/vram_alloc_gb"], total_vram_gb, self.state.step,
                    ):
                        print(f"  health[{level.value}]: {msg}", flush=True)

                # Cost tracking ($/step, projected total, ETA vs budget)
                if self._cost_per_gpu_hour > 0:
                    hours = self.state.wall_clock_seconds / 3600.0
                    spent = hours * self._cost_per_gpu_hour
                    steps_per_hour = self.state.step / max(hours, 1e-9)
                    eta_hours = max(0, (self._total_steps - self.state.step)) / max(steps_per_hour, 1e-9)
                    metrics["cost/usd_spent"] = spent
                    metrics["cost/usd_projected_total"] = spent + eta_hours * self._cost_per_gpu_hour
                    metrics["cost/usd_per_1k_steps"] = (spent / max(self.state.step, 1)) * 1000
                    metrics["cost/usd_per_1b_tokens"] = (spent / max(self.state.tokens_seen, 1)) * 1e9
                    metrics["cost/eta_hours"] = eta_hours
                    if self._cost_budget > 0 and metrics["cost/usd_projected_total"] > self._cost_budget:
                        # Not a hard stop by default — alert level. Raise to STOP
                        # by setting monitoring.health.grad_explosion_threshold=0 etc.
                        self.state.alerts.append(
                            f"projected cost ${metrics['cost/usd_projected_total']:.0f} "
                            f"> budget ${self._cost_budget:.0f}"
                        )

                if self._is_main:
                    self.log(metrics, self.state.step)

                # Health check — runs on every rank so per-rank grad/VRAM
                # anomalies are caught; only rank 0 prints to avoid N-way noise.
                for level, msg in self.health.observe(metrics, self.state.step):
                    if self._is_main:
                        print(f"  health[{level.value}]: {msg}", flush=True)
                if self._is_main:
                    extra = ""
                    if self.device.type == "cuda":
                        extra = f" | vram {metrics['train/vram_alloc_gb']:.1f}/{metrics['train/vram_peak_gb']:.1f}GB"
                    print(
                        f"step {self.state.step:6d} | loss {avg_loss:6.4f} | "
                        f"lr {lr:.2e} | grad_norm {float(grad_norm):5.2f} | "
                        f"tok/s {tps/1e3:6.1f}k | "
                        f"data {metrics['train/data_frac']*100:4.1f}%"
                        f"{extra}",
                        flush=True,
                    )
                window_loss_sum = 0.0
                window_loss_main_sum = 0.0
                window_loss_mtp_sum = 0.0
                window_loss_main_count = 0
                window_loss_mtp_count = 0
                window_t0 = time.time()
                window_data_time = 0.0
                window_compute_time = 0.0

            if self.val_dataloader is not None and self.state.step % self._eval_every == 0:
                # Eval is forward-only (no gradient all-reduce), so rank 0 runs it
                # alone while the others wait at the barrier below — no DDP desync.
                if self._is_main:
                    val_loss = self._evaluate()
                    self._track_val(val_loss)
                    # Health may also STOP on sustained val regression.
                    for level, msg in self.health.observe_val(
                        val_loss,
                        self.state.best_val_loss,
                        self.state.consecutive_val_increases,
                        self.state.step,
                    ):
                        print(f"  health[{level.value}]: {msg}", flush=True)
                if self._is_distributed:
                    torch.distributed.barrier()

            if self.state.step % self._save_every == 0:
                self._save_step_checkpoint_if_needed(f"step_{self.state.step}")

            # Stop decision must be unanimous across ranks or DDP collectives
            # would hang. Reduce the local flag with MAX so any rank's stop wins.
            stop_local = self.health.should_stop()
            if self._is_distributed:
                flag = torch.tensor([1.0 if stop_local else 0.0], device=self.device)
                torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
                stop_global = bool(flag.item() > 0)
            else:
                stop_global = stop_local
            if stop_global:
                reason = self.health.state.stop_reason or "peer_requested_stop"
                if self._is_main:
                    print(f"  AUTO-STOP: {reason}. Saving emergency ckpt and exiting.", flush=True)
                    self.save_checkpoint(f"step_{self.state.step}_emergency")
                if self._is_distributed:
                    torch.distributed.barrier()
                raise HealthStop(reason)

        # Final checkpoint — skip if the last in-loop save already wrote
        # step_{N}.pt (happens when total_steps % save_every == 0; e.g. for
        # 80000 % 2500). Avoids overwriting the freshly written file with
        # identical content and triggering a redundant external backup.
        if self.state.step % self._save_every != 0:
            self._save_step_checkpoint_if_needed(f"step_{self.state.step}")
        return self.state

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _evaluate(self) -> float:
        """Compute val_loss overall + per-language (pretrain-mix diagnostic).

        Per-language slices are drawn via ``val_dataloader.sample_language``
        if that method exists (our MixedDataLoader). Falls back to only the
        mixed loss for generic iterables.
        """
        self.model.eval()
        assert self.val_dataloader is not None
        # Rewind the val RNGs so EVERY eval sees the identical token stream.
        # Otherwise the stateful sampler advances and each eval draws different
        # val tokens — turning the val trajectory into model-change + sampling
        # noise (the confound the five-AI review flagged and we confirmed).
        reset = getattr(self.val_dataloader, "reset_rngs", None)
        if callable(reset):
            reset()
        eval_cfg = self.config.get("evaluation", {})
        max_batches = int(eval_cfg.get("max_val_batches", 50))
        per_lang_batches = int(eval_cfg.get("per_language_batches", 8))
        tokens_per_byte = {
            str(k): float(v)
            for k, v in (eval_cfg.get("tokens_per_byte") or {}).items()
        }

        metrics: dict[str, float] = {}

        # Overall mixed-batch val loss
        losses: list[float] = []
        it = iter(self.val_dataloader)
        for _ in range(max_batches):
            try:
                batch = next(it)
            except StopIteration:
                break
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            with self._autocast():
                out = self.model(input_ids=batch["input_ids"], labels=batch["labels"])
            losses.append(out["loss"].item())
        val = sum(losses) / max(1, len(losses))
        metrics["eval/val_loss"] = val

        # Per-language slices
        sample_lang = getattr(self.val_dataloader, "sample_language", None)
        batch_size = int(self.config.get("training", {}).get("batch_size_per_device", 4))
        if callable(sample_lang) and per_lang_batches > 0:
            lang_losses: dict[str, list[float]] = {}
            for lang in getattr(self.val_dataloader, "mix_ratios", {}):
                if self.val_dataloader.mix_ratios.get(lang, 0) <= 0:
                    continue  # skip languages that are weighted out
                lang_losses[lang] = []
                for _ in range(per_lang_batches):
                    try:
                        batch = sample_lang(lang, batch_size)
                    except (KeyError, ValueError):
                        break
                    batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                    with self._autocast():
                        out = self.model(input_ids=batch["input_ids"], labels=batch["labels"])
                    lang_losses[lang].append(out["loss"].item())
            for lang, ls in lang_losses.items():
                if ls:
                    lang_val = sum(ls) / len(ls)
                    metrics[f"eval/val_loss/{lang}"] = lang_val
                    if lang in tokens_per_byte:
                        metrics[f"eval/bpb/{lang}"] = bits_per_byte(lang_val, tokens_per_byte[lang])
            bpbs = {
                k.rsplit("/", 1)[-1]: v
                for k, v in metrics.items()
                if k.startswith("eval/bpb/")
            }
            if len(bpbs) >= 2:
                metrics["eval/bpb_gap_max"] = bpb_gap(bpbs)

        self.model.train()
        self.log(metrics, self.state.step)
        pretty = " ".join(f"{k.removeprefix('eval/').replace('/', '_')}={v:.3f}" for k, v in metrics.items())
        print(f"  eval @ step {self.state.step}: {pretty}", flush=True)
        for level, msg in self.health.observe_bpb(metrics, self.state.step):
            print(f"  health[{level.value}]: {msg}", flush=True)
        return val

    def _track_val(self, val_loss: float) -> None:
        """Update best-checkpoint + the consecutive-rise guard.

        Two SEPARATE decisions, previously conflated into one bug:

        * best.pt is written only on a real improvement (beat best by more than
          ``min_delta``), so sub-noise wobble doesn't churn the checkpoint.
        * ``consecutive_val_increases`` counts TRUE consecutive rises — val_loss
          higher than the PREVIOUS eval by more than ``min_delta`` — not "failed
          to beat the all-time best". The old code incremented on every eval
          that wasn't a new best, so a model sitting just above its optimum (the
          normal case) looked like a runaway regression and tripped the stop.
        """
        delta = self._val_min_delta
        # best tracking (checkpoint) — real improvement only
        if val_loss < self.state.best_val_loss - delta:
            self.state.best_val_loss = val_loss
            if self._save_best:
                self.save_checkpoint("best")

        # true consecutive-rise detection vs the previous eval
        prev = self.state.last_val_loss
        if prev != float("inf") and val_loss > prev + delta:
            self.state.consecutive_val_increases += 1
        else:
            self.state.consecutive_val_increases = 0
        self.state.last_val_loss = val_loss

        if self.state.consecutive_val_increases >= 3:
            msg = (
                f"val_loss rose {self.state.consecutive_val_increases} evals in a row; "
                f"best={self.state.best_val_loss:.4f} now={val_loss:.4f}"
            )
            self.state.alerts.append(msg)
            print(f"  ALERT: {msg}", flush=True)

    # ------------------------------------------------------------------
    def _save_step_checkpoint_if_needed(self, name: str) -> Path | None:
        if (
            self._skip_step_save_if_checkpoint_written
            and self._last_checkpoint_step == self.state.step
        ):
            print(
                f"  ckpt {name} skipped; {self._last_checkpoint_name}.pt "
                f"already written at step {self.state.step}",
                flush=True,
            )
            return None
        return self.save_checkpoint(name)

    def save_checkpoint(self, name: str) -> Path:
        import random

        import numpy as np

        path = self._ckpt_dir / f"{name}.pt"
        # Only rank 0 writes checkpoints in a distributed run; the others would
        # race on the same path (and hold identical replicated weights anyway).
        if not self._is_main:
            return path
        tmp = path.with_suffix(".pt.tmp")
        # RNG state so a resume continues the same random stream.
        rng = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
        if torch.cuda.is_available():
            rng["cuda"] = torch.cuda.get_rng_state_all()
        payload = {
            "model": self._core_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "state": asdict(self.state),
            "metadata": asdict(self.metadata),
            "scaler": self._scaler.state_dict() if self._scaler is not None else None,
            "rng": rng,
        }
        t_write_0 = time.time()
        torch.save(payload, tmp)
        tmp.replace(path)
        write_seconds = time.time() - t_write_0

        # Sidecar JSON for quick introspection without loading the tensors.
        sidecar = path.with_suffix(".json")
        sidecar.write_text(
            json.dumps(
                {"state": asdict(self.state), "metadata": asdict(self.metadata)},
                indent=2,
            ),
            encoding="utf-8",
        )

        # External backup (e.g. NAS) every N steps.
        backup_cfg = self._external_backup
        backup_ran = False
        if backup_cfg.get("enabled"):
            interval = int(backup_cfg.get("interval_steps", 0))
            if interval > 0 and self.state.step % interval == 0:
                backup_root = Path(backup_cfg["path"])
                ok = False
                try:
                    backup_root.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, backup_root / path.name)
                    shutil.copy2(sidecar, backup_root / sidecar.name)
                    self.state.external_backups_ok += 1
                    self.state.external_backups_consecutive_failed = 0
                    ok = True
                    backup_ran = True
                except OSError as e:
                    # Non-fatal at the per-call level — but the health monitor
                    # below decides whether the *cumulative pattern* is fatal.
                    self.state.external_backups_failed += 1
                    self.state.external_backups_consecutive_failed += 1
                    self.state.alerts.append(f"backup to {backup_root} failed: {e}")
                # Tell the health monitor whether this attempt succeeded.
                # Three consecutive failures trip stop_requested and the next
                # main-loop tick will save an emergency checkpoint and exit.
                for level, msg in self.health.observe_backup(
                    ok=ok,
                    fail_count=self.state.external_backups_consecutive_failed,
                ):
                    print(f"  health[{level.value}]: {msg}", flush=True)

        self.log(
            {
                "ckpt/write_seconds": float(write_seconds),
                "ckpt/bytes": float(path.stat().st_size),
                "ckpt/external_backups_ok": float(self.state.external_backups_ok),
                "ckpt/external_backups_failed": float(self.state.external_backups_failed),
            },
            self.state.step,
        )
        # Ckpt-write anomaly detection (over rolling median)
        for level, msg in self.health.observe_checkpoint_write(write_seconds, self.state.step):
            print(f"  health[{level.value}]: {msg}", flush=True)
        note = f"  ckpt {name} written in {write_seconds:.1f}s" \
               f" ({path.stat().st_size/1e9:.2f} GB)"
        if backup_ran:
            note += " + backup OK"
        print(note, flush=True)

        self._rotate_checkpoints()
        self._last_checkpoint_step = self.state.step
        self._last_checkpoint_name = name
        return path

    def _rotate_checkpoints(self) -> None:
        # Emergency snapshots (step_<n>_emergency.pt) are exempt from rotation:
        # they mark a health-stop event and must survive for forensics, even
        # when a regular save at the same step would otherwise evict them.
        def _step_sort_key(path: Path) -> int | None:
            match = re.match(r"step_(\d+)$", path.stem)
            return int(match.group(1)) if match else None

        step_ckpts = sorted(
            (
                p
                for p in self._ckpt_dir.glob("step_*.pt")
                if not p.name.endswith(".tmp") and _step_sort_key(p) is not None
            ),
            key=lambda p: _step_sort_key(p) or -1,
            reverse=True,
        )
        for p in step_ckpts[self._keep_last :]:
            p.unlink(missing_ok=True)
            p.with_suffix(".json").unlink(missing_ok=True)

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore full training state (use for resume).

        Warns if the checkpoint's git_sha or config_sha16 don't match the
        current run — that's usually what you want, but silent drift is
        what turns "resume" into "Frankenrun".
        """
        # weights_only=False is required: the payload holds optimizer / RNG /
        # dataclass state, not just tensors. Only ever load self-produced ckpts.
        payload = torch.load(path, map_location="cpu", weights_only=False)

        # Align torch.compile key prefixes so a checkpoint saved from a compiled
        # model resumes into a non-compiled one (and vice versa) instead of
        # crashing on a `_orig_mod.` key mismatch.
        state = _align_model_state_for_load(self._core_model, payload["model"])
        self._core_model.load_state_dict(state)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.state = TrainerState(**payload["state"])
        if self._scaler is not None and payload.get("scaler") is not None:
            self._scaler.load_state_dict(payload["scaler"])

        # Restore RNG so the random stream continues (best-effort; never fatal).
        rng = payload.get("rng") or {}
        try:
            import random

            import numpy as np

            if rng.get("torch") is not None:
                torch.set_rng_state(rng["torch"])
            if rng.get("numpy") is not None:
                np.random.set_state(rng["numpy"])
            if rng.get("python") is not None:
                random.setstate(rng["python"])
            cuda_rng = rng.get("cuda")
            if (cuda_rng is not None and torch.cuda.is_available()
                    and len(cuda_rng) == torch.cuda.device_count()):
                torch.cuda.set_rng_state_all(cuda_rng)
        except Exception as exc:  # RNG mismatch must not block a resume
            print(f"  warn: could not fully restore RNG state: {exc}", flush=True)

        old_meta = payload.get("metadata") or {}
        mismatches = []
        for key in ("git_sha", "config_sha16", "tokenizer_sha16"):
            old = old_meta.get(key)
            new = getattr(self.metadata, key, None)
            if old and new and old != new:
                mismatches.append(f"{key}: {old[:16]} → {new[:16]}")
        if mismatches:
            msg = "resume provenance mismatch — " + "; ".join(mismatches)
            self.state.alerts.append(msg)
            print(f"  warn: {msg}", flush=True)

    def warm_start_from_checkpoint(self, path: str | Path, *, strict: bool = True) -> None:
        """Load model weights only while keeping a fresh run state.

        This is intentionally different from :meth:`load_checkpoint`: optimizer,
        scheduler, scaler, RNG, and ``TrainerState.step`` are not restored. Use
        it when a short ramp found useful weights but its LR schedule is too
        short to resume for a longer foundation run.
        """
        payload = torch.load(path, map_location="cpu", weights_only=False)
        # Hard tokenizer-identity check: warm-starting weights trained under a
        # different tokenizer means the embedding/lm_head rows no longer map to
        # the same token IDs — a silent corpus-level corruption. Fail loudly.
        src_meta = payload.get("metadata") or {}
        src_tok = src_meta.get("tokenizer_sha16")
        cur_tok = self.metadata.tokenizer_sha16
        if src_tok and cur_tok and src_tok != cur_tok:
            raise RuntimeError(
                f"warm-start tokenizer mismatch: checkpoint was trained with "
                f"tokenizer sha {src_tok} but this run uses {cur_tok}. Token IDs "
                f"would not align — refusing to warm-start. Re-tokenize with the "
                f"matching tokenizer or warm-start from a matching checkpoint."
            )
        state = _align_model_state_for_load(self._core_model, payload["model"])
        missing, extra = self._core_model.load_state_dict(state, strict=strict)
        if missing or extra:
            raise RuntimeError(
                f"warm-start checkpoint mismatch: missing={len(missing)} extra={len(extra)}; "
                f"first_missing={missing[:3]} first_extra={extra[:3]}"
            )
        source_state = payload.get("state") or {}
        source_step = source_state.get("step", "?")
        source_best = source_state.get("best_val_loss", "?")
        note = (
            f"warm-started model weights from {path} "
            f"(source step={source_step}, source best_val_loss={source_best})"
        )
        self.state.alerts.append(note)
        print(f"  {note}; optimizer/scheduler/state kept fresh", flush=True)


__all__ = ["PretrainTrainer", "TrainerState"]
