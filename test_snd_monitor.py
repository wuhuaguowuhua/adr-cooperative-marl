import pytest

from snd_monitor import SNDMonitor


def test_static_schedule_bypasses_warmup_and_stays_constant():
    monitor = SNDMonitor(trigger="static", eta_max=0.2, warmup_ratio=0.9)

    assert monitor.compute_eta(0.0) == pytest.approx(0.2)
    assert monitor.compute_eta(0.5) == pytest.approx(0.2)
    assert monitor.compute_eta(1.0) == pytest.approx(0.2)


def test_linear_schedule_bypasses_warmup_and_reaches_zero():
    monitor = SNDMonitor(trigger="linear", eta_max=0.2, warmup_ratio=0.9)

    assert monitor.compute_eta(0.0) == pytest.approx(0.2)
    assert monitor.compute_eta(0.5) == pytest.approx(0.1)
    assert monitor.compute_eta(1.0) == pytest.approx(0.0)
    assert monitor.compute_eta(1.2) == pytest.approx(0.0)
