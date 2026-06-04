from auralis.training.health import AlertLevel, HealthConfig, HealthMonitor


def test_bpb_guard_disabled_by_default():
    health = HealthMonitor()
    metrics = {"eval/bpb/english": 1.0}

    assert health.observe_bpb(metrics, step=100) == []
    assert not health.should_stop()


def test_bpb_guard_stops_after_consecutive_regressions():
    health = HealthMonitor(HealthConfig(
        bpb_regression_enabled=True,
        bpb_regression_languages=["english"],
        bpb_regression_max_increase=0.10,
        bpb_regression_k=2,
        bpb_regression_lookback=4,
        bpb_regression_warmup_evals=1,
    ))

    assert health.observe_bpb({"eval/bpb/english": 1.00}, step=100) == []
    assert health.observe_bpb({"eval/bpb/english": 0.90}, step=200) == []
    assert health.observe_bpb({"eval/bpb/english": 1.05}, step=300) == []
    fresh = health.observe_bpb({"eval/bpb/english": 1.08}, step=400)

    assert fresh
    assert fresh[0][0] == AlertLevel.STOP
    assert "bpb/english regressed" in fresh[0][1]
    assert health.should_stop()
    assert health.state.stop_reason == "bpb_regression:english"


def test_bpb_guard_tracks_only_configured_languages():
    health = HealthMonitor(HealthConfig(
        bpb_regression_enabled=True,
        bpb_regression_languages=["german"],
        bpb_regression_max_increase=0.10,
        bpb_regression_k=1,
        bpb_regression_warmup_evals=1,
    ))

    health.observe_bpb({"eval/bpb/english": 1.00, "eval/bpb/german": 2.00}, step=100)
    fresh = health.observe_bpb({"eval/bpb/english": 9.00, "eval/bpb/german": 1.95}, step=200)

    assert fresh == []
    assert not health.should_stop()
    assert "english" not in health.state.bpb_series
    assert health.state.bpb_series["german"] == [2.00, 1.95]
