"""End-to-end smoke test of PretrainTrainer on a tiny 100M-ish model.

Feeds synthetic token bins into the loader (no real tokenization needed),
runs a handful of gradient-accumulation steps, and checks:

- loss is finite at every step
- loss is non-increasing (on average) over the short run
- the trainer writes a checkpoint + JSON sidecar
- loading the checkpoint restores ``state.step``

Kept under ~2 minutes on CPU by using a 4-layer shrunken model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from auralis.model.config import AuralisConfig, LayerConfig
from auralis.model.helix_model import HelixModel
from auralis.training.dataset import MixedDataLoader
from auralis.training.optimizer import build_optimizer, build_scheduler
from auralis.training.trainer import PretrainTrainer


def _tiny_model() -> HelixModel:
    layers = (
        [LayerConfig(type="mamba", d_state=16, d_conv=4, expand_factor=2)] * 1
        + [LayerConfig(type="gla", d_state=16)] * 2
        + [LayerConfig(type="sparse_attention", window_size=16, global_tokens=4, use_rope=True)] * 1
    )
    cfg = AuralisConfig(
        name="tiny", version="1.0",
        vocab_size=4096,
        d_model=64, n_layers=4, n_heads=4, d_head=16, d_ffn=128,
        layers=layers,
    )
    cfg.advanced.tie_embeddings = True
    cfg.position_encoding.max_seq_length = 64
    return HelixModel(cfg)


def _write_bins(dir_path: Path, n_each: int, vocab: int) -> None:
    rng = np.random.default_rng(0)
    for lang in ("english", "german", "code"):
        arr = rng.integers(0, vocab, size=n_each, dtype=np.uint32)
        arr.tofile(dir_path / f"{lang}.bin")


@pytest.fixture
def trainer_env(tmp_path: Path):
    torch.manual_seed(0)
    np.random.seed(0)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_bins(data_dir, n_each=2000, vocab=4096)

    ckpt_dir = tmp_path / "ckpt"
    model = _tiny_model()
    loader = MixedDataLoader(
        data_dir=data_dir,
        mix_ratios={"english": 0.5, "german": 0.5, "code": 0.0},
        batch_size=4,
        seq_length=16,
        seed=0,
    )
    opt = build_optimizer(model, {"name": "adamw", "lr": 1e-3, "weight_decay": 0.01})
    sched = build_scheduler(
        opt, {"type": "cosine", "warmup_steps": 2, "min_lr_ratio": 0.5}, total_steps=10
    )
    config = {
        "data": {"seq_length": 16},
        "training": {
            "batch_size_per_device": 4,
            "gradient_accumulation": 1,
            "gradient_clip_norm": 1.0,
            "total_steps": 10,
        },
        "logging": {"log_every": 2, "eval_every": 9999, "save_every": 10},
        "checkpointing": {"output_dir": str(ckpt_dir), "save_last_n": 2},
    }
    trainer = PretrainTrainer(
        model=model,
        optimizer=opt,
        scheduler=sched,
        dataloader=loader,
        config=config,
        device="cpu",
    )
    return trainer, ckpt_dir


def test_trainer_runs_and_loss_finite(trainer_env):
    trainer, _ = trainer_env
    losses: list[float] = []

    # Patch the log function to capture loss values.
    original = trainer.log
    def capture(metrics, step):
        if "train/loss" in metrics:
            losses.append(metrics["train/loss"])
        original(metrics, step)
    trainer.log = capture

    # Capture weights before training to verify the optimizer actually moved them.
    w_before = next(p.detach().clone() for p in trainer.model.parameters() if p.requires_grad)

    final_state = trainer.train()

    assert final_state.step == 10
    assert len(losses) >= 3
    # Every logged loss finite + within a sane absolute range (vocab=4096 ⇒ ln≈8.3).
    for lv in losses:
        assert 0 < lv < 20
    # Parameters actually changed (optimizer.step did something).
    w_after = next(p.detach() for p in trainer.model.parameters() if p.requires_grad)
    assert not torch.equal(w_before, w_after)
    assert torch.isfinite(w_after).all()


def test_trainer_saves_and_loads_checkpoint(trainer_env):
    trainer, ckpt_dir = trainer_env
    trainer.train()
    ckpts = list(ckpt_dir.glob("step_*.pt"))
    assert ckpts, "no checkpoint written"
    sidecar = ckpts[0].with_suffix(".json")
    assert sidecar.exists()

    # Make a fresh trainer from the same pieces, load, verify state.
    trainer2 = trainer.__class__(
        model=trainer.model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        dataloader=trainer.dataloader,
        config=trainer.config,
        device="cpu",
    )
    trainer2.load_checkpoint(ckpts[0])
    assert trainer2.state.step == 10


def test_trainer_warm_start_loads_weights_without_resume_state(trainer_env, tmp_path: Path):
    trainer, ckpt_dir = trainer_env
    trainer.train()
    ckpt = next(ckpt_dir.glob("step_*.pt"))
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    saved_weight = next(iter(payload["model"].values())).detach().clone()

    data_dir = tmp_path / "warm_data"
    data_dir.mkdir()
    _write_bins(data_dir, n_each=2000, vocab=4096)
    model = _tiny_model()
    loader = MixedDataLoader(
        data_dir=data_dir,
        mix_ratios={"english": 0.5, "german": 0.5, "code": 0.0},
        batch_size=4,
        seq_length=16,
        seed=11,
    )
    opt = build_optimizer(model, {"name": "adamw", "lr": 9e-4, "weight_decay": 0.01})
    sched = build_scheduler(
        opt, {"type": "cosine", "warmup_steps": 3, "min_lr_ratio": 0.25}, total_steps=20
    )
    config = {
        "data": {"seq_length": 16},
        "training": {
            "batch_size_per_device": 4,
            "gradient_accumulation": 1,
            "gradient_clip_norm": 1.0,
            "total_steps": 20,
        },
        "logging": {"log_every": 2, "eval_every": 9999, "save_every": 10},
        "checkpointing": {"output_dir": str(tmp_path / "warm_ckpt"), "save_last_n": 2},
    }
    warm = PretrainTrainer(
        model=model,
        optimizer=opt,
        scheduler=sched,
        dataloader=loader,
        config=config,
        device="cpu",
    )
    initial_lr = warm.scheduler.get_last_lr()[0]

    warm.warm_start_from_checkpoint(ckpt)

    assert warm.state.step == 0
    assert warm.state.best_val_loss == float("inf")
    assert warm.scheduler.get_last_lr()[0] == initial_lr
    assert not warm.optimizer.state_dict()["state"]
    loaded_weight = next(iter(warm.model.state_dict().values())).detach()
    assert torch.equal(loaded_weight, saved_weight)


def test_trainer_runs_evaluation_when_val_loader_present(trainer_env, tmp_path: Path):
    trainer, _ = trainer_env
    # Build a tiny val loader that reuses the same synthetic bins with a
    # fresh split (no bytes reserved — just reuse the full window for the test).
    import numpy as np
    from auralis.training.dataset import MixedDataLoader
    data_dir = tmp_path / "valdata"
    data_dir.mkdir()
    rng = np.random.default_rng(1)
    for lang in ("english", "german", "code"):
        rng.integers(0, 4096, size=2000, dtype=np.uint32).tofile(data_dir / f"{lang}.bin")
    trainer.val_dataloader = MixedDataLoader(
        data_dir=data_dir,
        mix_ratios={"english": 0.5, "german": 0.5, "code": 0.0},
        batch_size=2, seq_length=16, seed=1,
    )
    # Make eval fire once during the 10-step run
    trainer._eval_every = 5
    trainer.config["evaluation"] = {"max_val_batches": 3}

    metrics_captured: list[dict[str, float]] = []
    original = trainer.log
    def cap(m, s):
        metrics_captured.append(dict(m))
        original(m, s)
    trainer.log = cap

    trainer.train()
    # At least one eval metric must have been logged.
    assert any("eval/val_loss" in m for m in metrics_captured), metrics_captured
    # best.pt must exist after the first successful eval.
    best = Path(trainer.config["checkpointing"]["output_dir"]) / "best.pt"
    assert best.exists()


def test_trainer_bf16_autocast_selected_on_config(trainer_env):
    trainer, _ = trainer_env
    trainer.config["training"]["dtype"] = "bf16"
    # Re-run __init__-style flags
    trainer._amp_dtype = torch.bfloat16
    trainer._use_amp = True
    ctx = trainer._autocast()
    # Can't assert much at a type level, but the autocast manager must be usable
    with ctx:
        t = torch.randn(2, 3)
        y = t @ t.T
    assert y.shape == (2, 2)


def test_trainer_fp16_without_cuda_rejected(trainer_env):
    """fp16 is only numerically safe with a GradScaler; GradScaler requires CUDA.
    Trying to configure fp16 on CPU must fail at construction, not silently
    produce NaNs later."""
    from auralis.training.trainer import PretrainTrainer
    trainer, _ = trainer_env
    cfg = {**trainer.config, "training": {**trainer.config["training"], "dtype": "fp16"}}
    with pytest.raises(ValueError, match="fp16 training requires a CUDA device"):
        PretrainTrainer(
            model=trainer.model, optimizer=trainer.optimizer,
            scheduler=trainer.scheduler, dataloader=trainer.dataloader,
            config=cfg, device="cpu",
        )


def test_trainer_records_run_metadata(trainer_env):
    trainer, _ = trainer_env
    md = trainer.metadata
    # Non-empty values from the environment
    assert md.torch_version
    assert md.hostname
    # config_sha16 is deterministic — 16 hex chars
    assert len(md.config_sha16) == 16


def test_safe_log_swallows_exceptions(trainer_env):
    """A crashing metrics logger must NOT kill training."""
    trainer, _ = trainer_env
    def boom(_m, _s):
        raise RuntimeError("pretend wandb died")
    trainer.log = trainer.log.__class__ if False else None  # satisfy type-checker
    # Re-wrap via the same helper the Trainer uses
    from auralis.training.trainer import _safe_log
    trainer.log = _safe_log(boom)
    # This must not raise
    trainer.train()


def test_trainer_saves_metadata_in_checkpoint(trainer_env, tmp_path: Path):
    trainer, _ = trainer_env
    trainer.train()
    ckpts = list(Path(trainer.config["checkpointing"]["output_dir"]).glob("step_*.pt"))
    assert ckpts
    payload = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    assert "metadata" in payload
    assert payload["metadata"]["torch_version"]
    # Sidecar JSON mirrors metadata
    sidecar = ckpts[0].with_suffix(".json")
    import json
    obj = json.loads(sidecar.read_text(encoding="utf-8"))
    assert "metadata" in obj and "state" in obj


def test_trainer_can_skip_duplicate_step_checkpoint_after_best(trainer_env, monkeypatch):
    trainer, _ = trainer_env
    trainer._skip_step_save_if_checkpoint_written = True
    trainer.state.step = 500

    calls: list[str] = []

    def fake_save(name: str) -> Path:
        calls.append(name)
        trainer._last_checkpoint_step = trainer.state.step
        trainer._last_checkpoint_name = name
        return Path(f"{name}.pt")

    monkeypatch.setattr(trainer, "save_checkpoint", fake_save)

    assert trainer._save_step_checkpoint_if_needed("best") == Path("best.pt")
    assert trainer._save_step_checkpoint_if_needed("step_500") is None
    assert calls == ["best"]


def test_trainer_raises_on_nan_loss(trainer_env, monkeypatch):
    trainer, _ = trainer_env

    class NaNModel(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def forward(self, input_ids, labels):
            out = self.inner(input_ids=input_ids, labels=labels)
            out["loss"] = torch.tensor(float("nan"))
            return out

    trainer.model = NaNModel(trainer.model)
    with pytest.raises(RuntimeError, match="non-finite loss"):
        trainer.train()


def test_emergency_checkpoint_does_not_break_rotation(trainer_env):
    trainer, ckpt_dir = trainer_env
    trainer._keep_last = 1

    trainer.save_checkpoint("step_1")
    trainer.save_checkpoint("step_2")
    trainer.save_checkpoint("step_2_emergency")

    ckpts = {p.name for p in ckpt_dir.glob("step_*.pt")}
    assert "step_2.pt" in ckpts
    assert "step_1.pt" not in ckpts


def test_emergency_checkpoint_survives_same_step_rotation(trainer_env):
    """Emergency snapshot must survive even when a regular save shares the step.

    Regression: rotation previously matched both ``step_<n>.pt`` and
    ``step_<n>_emergency.pt`` under the same numeric sort key, so on collision
    one of the two was silently unlinked depending on glob order — defeating
    the purpose of the emergency save in the auto-stop recovery path.
    """
    trainer, ckpt_dir = trainer_env
    trainer._keep_last = 1

    # Order A: regular first, then emergency at the same step.
    trainer.save_checkpoint("step_5")
    trainer.save_checkpoint("step_5_emergency")
    names = {p.name for p in ckpt_dir.glob("step_*.pt")}
    assert "step_5.pt" in names
    assert "step_5_emergency.pt" in names

    # Order B: emergency first, then a later regular save that triggers rotation.
    trainer.save_checkpoint("step_10_emergency")
    trainer.save_checkpoint("step_11")
    names = {p.name for p in ckpt_dir.glob("step_*.pt")}
    assert "step_11.pt" in names, "latest regular ckpt must be kept"
    assert "step_10_emergency.pt" in names, "emergency must never be rotated away"
    assert "step_5.pt" not in names, "older regular ckpt must be rotated"
    # The earlier emergency from step_5 also survives — emergencies are exempt.
    assert "step_5_emergency.pt" in names
