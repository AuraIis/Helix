"""HealthMonitor guard tests."""

from __future__ import annotations

from auralis.training.health import AlertLevel, HealthConfig, HealthMonitor


def test_grad_explosion_triggers_stop_after_k_windows():
    mon = HealthMonitor(HealthConfig(grad_explosion_threshold=10.0, grad_explosion_k=3))
    assert not mon.should_stop()
    for i in range(2):
        mon.observe({"train/grad_norm": 20.0}, step=10 + i * 10)
    assert not mon.should_stop(), "should not stop before K windows"
    mon.observe({"train/grad_norm": 20.0}, step=30)
    assert mon.should_stop()
    assert "grad_explosion" in mon.state.stop_reason


def test_grad_norm_resets_explosion_counter():
    mon = HealthMonitor(HealthConfig(grad_explosion_threshold=10.0, grad_explosion_k=3))
    mon.observe({"train/grad_norm": 20.0}, step=10)
    mon.observe({"train/grad_norm": 20.0}, step=20)
    mon.observe({"train/grad_norm": 2.0}, step=30)  # reset
    mon.observe({"train/grad_norm": 20.0}, step=40)
    mon.observe({"train/grad_norm": 20.0}, step=50)
    assert not mon.should_stop()


def test_loss_spike_warns_but_does_not_stop():
    mon = HealthMonitor(HealthConfig(loss_spike_factor=3.0, loss_spike_avg_window=10))
    # Build up a running-avg around 2.0
    for _ in range(10):
        mon.observe({"train/loss": 2.0}, step=1)
    fresh = mon.observe({"train/loss": 10.0}, step=100)  # spike
    assert any(lvl == AlertLevel.WARN for lvl, _ in fresh)
    assert not mon.should_stop()


def test_val_regression_triggers_stop():
    mon = HealthMonitor(HealthConfig(val_regression_stop_k=4))
    for i in range(3):
        fresh = mon.observe_val(val_loss=2.0, best_val_loss=1.0,
                                consecutive_increases=i + 1, step=100 + i)
        assert not mon.should_stop()
    fresh = mon.observe_val(val_loss=2.0, best_val_loss=1.0,
                            consecutive_increases=4, step=200)
    assert any(lvl == AlertLevel.STOP for lvl, _ in fresh)
    assert mon.should_stop()


def test_vram_warn_then_stop():
    mon = HealthMonitor(HealthConfig(vram_frac_warn=0.90, vram_frac_stop=0.95))
    fresh = mon.observe_vram(alloc_gb=9.0, total_gb=10.0, step=1)   # 90 % → WARN
    assert any(lvl == AlertLevel.WARN for lvl, _ in fresh)
    assert not mon.should_stop()
    fresh = mon.observe_vram(alloc_gb=9.6, total_gb=10.0, step=2)   # 96 % → STOP
    assert any(lvl == AlertLevel.STOP for lvl, _ in fresh)
    assert mon.should_stop()


def test_ckpt_write_anomaly_triggers_warn():
    mon = HealthMonitor(HealthConfig(ckpt_time_factor=3.0, ckpt_time_median_window=5))
    # Establish a median around 2 s
    for i in range(4):
        fresh = mon.observe_checkpoint_write(seconds=2.0 + 0.1 * i, step=100 + i)
        assert not fresh
    # Spike: 10 s > 3 × ~2.15 s median
    fresh = mon.observe_checkpoint_write(seconds=10.0, step=200)
    assert any(lvl == AlertLevel.WARN for lvl, _ in fresh)


def test_summary_serialisable():
    mon = HealthMonitor(HealthConfig())
    mon.observe({"train/grad_norm": 5.0, "train/loss": 2.0, "train/tokens_per_second": 1000}, step=1)
    s = mon.summary()
    assert "stop_requested" in s and "n_alerts" in s
    # Must be JSON-serialisable (all keys primitive)
    import json
    json.dumps(s)
