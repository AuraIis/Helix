"""AdaptiveCurriculumTrainer: the training loop that wires it all together.

It owns the optimizer step, but delegates *what to train on* and *when to move
on* to the curriculum controller, and *whether it is learning* to the monitor.
The loop:

    for each global step:
        train grad_accum microbatches from the current stage's data
        every eval_every steps:
            snapshot = monitor.evaluate(...)
            decision = controller.update(snapshot)
            act on decision  (continue / advance+switch data / hold / stop)

Torch-dependent; validate on GPU in the container. The decision logic it relies
on is already unit-tested (see tests/adaptive).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import torch

from .adapters import ModelAdapter, TokenizerAdapter, build_stage_loader
from .controller import CurriculumController, Decision, DecisionKind
from .monitor import LearningMonitor
from .stages import CurriculumSpec


@dataclass
class TrainerConfig:
    batch_size: int = 8
    seq_length: int = 2048
    grad_accum: int = 1
    max_grad_norm: float = 1.0
    max_steps: int = 1_000_000
    data_seed: int = 42
    checkpoint_dir: str | None = None
    gradient_checkpointing: bool = False


@dataclass
class RunSummary:
    status: str                                   # "done" | "stopped" | "max_steps"
    steps: int
    final_stage: str
    decisions: list[dict] = field(default_factory=list)


class AdaptiveCurriculumTrainer:
    def __init__(
        self,
        model_adapter: ModelAdapter,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        tokenizer: TokenizerAdapter,
        spec: CurriculumSpec,
        controller: CurriculumController,
        monitor: LearningMonitor,
        cfg: TrainerConfig,
    ):
        self.ma = model_adapter
        self.opt = optimizer
        self.sched = scheduler
        self.tok = tokenizer
        self.spec = spec
        self.ctrl = controller
        self.mon = monitor
        self.cfg = cfg
        self._base_lrs = list(getattr(scheduler, "base_lrs", []))
        if cfg.gradient_checkpointing and hasattr(self.ma.model, "gradient_checkpointing_enable"):
            self.ma.model.gradient_checkpointing_enable()

    # ------------------------------------------------------------------
    def _build_loader(self, stage_idx: int) -> Iterator[dict[str, torch.Tensor]]:
        stage = self.spec.stages[stage_idx]
        # Per-stage context length enables a curriculum-by-length (short -> long,
        # the Qwen2.5-Coder file->repo pattern). Falls back to the global default.
        seq_length = stage.seq_length or self.cfg.seq_length
        return build_stage_loader(
            stage.data,
            tokenizer=self.tok,
            seq_length=seq_length,
            batch_size=self.cfg.batch_size,
            seed=self.cfg.data_seed + stage_idx,
        )

    def _apply_lr_scale(self, scale: float) -> None:
        if self._base_lrs and hasattr(self.sched, "base_lrs"):
            self.sched.base_lrs = [b * scale for b in self._base_lrs]

    def _eval_every(self, stage_idx: int) -> int:
        s = self.spec.stages[stage_idx]
        return s.eval_every or self.spec.eval_every

    def _save(self, tag: str, step: int) -> None:
        if not self.cfg.checkpoint_dir:
            return
        out = Path(self.cfg.checkpoint_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"adaptive_{tag}_step{step}.pt"
        torch.save(
            {
                "model": self.ma.model.state_dict(),
                "optimizer": self.opt.state_dict(),
                "scheduler": self.sched.state_dict() if hasattr(self.sched, "state_dict") else None,
                "step": step,
                "stage": self.ctrl.stage.name,
            },
            path,
        )
        print(f"[ckpt] saved {path}")

    # ------------------------------------------------------------------
    def run(self) -> RunSummary:
        loader = self._build_loader(self.ctrl.stage_index)
        self._apply_lr_scale(self.ctrl.stage.lr_scale)

        # Capture a fixed canary batch from stage 0 if the monitor has none.
        if self.mon.canary_batch is None:
            b = next(loader)
            self.mon.canary_batch = {
                "input_ids": b["input_ids"].detach().clone(),
                "labels": b["labels"].detach().clone(),
            }

        decisions: list[dict] = []
        stage_step = 0
        self.ma.model.train()

        for step in range(1, self.cfg.max_steps + 1):
            self._train_step(loader)
            stage_step += 1

            if step % self._eval_every(self.ctrl.stage_index) != 0:
                continue

            snap = self.mon.evaluate(step, stage_step, self.ctrl.stage.name)
            decision = self.ctrl.update(snap)
            decisions.append(_decision_dict(step, decision))
            self._announce(step, decision)

            if decision.kind == DecisionKind.ADVANCE:
                self._save(f"pre_advance_{decision.from_stage}", step)
                loader = self._build_loader(self.ctrl.stage_index)
                self._apply_lr_scale(self.ctrl.stage.lr_scale)
                stage_step = 0
                self.ma.model.train()
            elif decision.kind == DecisionKind.HOLD:
                # Guard tripped, non-fatal: stop training but keep the model.
                self._save("hold", step)
                return RunSummary("stopped", step, decision.from_stage, decisions)
            elif decision.is_terminal:
                self._save(decision.kind.value, step)
                status = "done" if decision.kind == DecisionKind.DONE else "stopped"
                return RunSummary(status, step, decision.from_stage, decisions)

        self._save("max_steps", self.cfg.max_steps)
        return RunSummary("max_steps", self.cfg.max_steps, self.ctrl.stage.name, decisions)

    # ------------------------------------------------------------------
    def _train_step(self, loader: Iterator[dict[str, torch.Tensor]]) -> None:
        self.opt.zero_grad(set_to_none=True)
        for _ in range(self.cfg.grad_accum):
            batch = next(loader)
            loss = self.ma.train_loss(batch) / self.cfg.grad_accum
            loss.backward()
        if self.cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.ma.model.parameters(), self.cfg.max_grad_norm)
        self.opt.step()
        self.sched.step()

    @staticmethod
    def _announce(step: int, d: Decision) -> None:
        arrow = f" -> {d.to_stage}" if d.to_stage else ""
        print(f"[ctrl step {step}] {d.kind.value}{arrow}: {d.reason}")


def _decision_dict(step: int, d: Decision) -> dict:
    return {
        "step": step,
        "kind": d.kind.value,
        "reason": d.reason,
        "from_stage": d.from_stage,
        "to_stage": d.to_stage,
    }


__all__ = ["AdaptiveCurriculumTrainer", "TrainerConfig", "RunSummary"]
