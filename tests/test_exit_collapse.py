"""退出坍缩检测测试。

纯 Python，不需要 torch —— 本机可直接跑：
    pytest tests/test_exit_collapse.py -v
"""

from __future__ import annotations

import json

from src.training.monitor import ExitCollapseMonitor


def test_no_alert_when_exit_step_healthy():
    m = ExitCollapseMonitor(alert_below=1.2, window=10)
    for step in range(50):
        m.record(step, 1.95)
    assert len(m.alerts) == 0


def test_alert_on_low_exit_step():
    """退出步数跌破阈值 → 必须告警。"""
    m = ExitCollapseMonitor(alert_below=1.2, window=100)
    m.record(1, 1.05)
    assert len(m.alerts) >= 1
    assert "跳过循环层" in m.alerts[0] or "退出步数" in m.alerts[0]


def test_alert_on_sudden_drop():
    """从 2.0 骤降到 1.3 → 判定为坍缩前兆。"""
    m = ExitCollapseMonitor(alert_below=1.2, drop_ratio_threshold=0.25, window=10)
    for step in range(1, 11):
        m.record(step, 2.0)
    m.record(11, 1.3)     # 下降 35% > 25%
    assert any("骤降" in a for a in m.alerts)


def test_strict_mode_raises():
    m = ExitCollapseMonitor(alert_below=1.5, strict=True)
    import pytest

    with pytest.raises(RuntimeError):
        m.record(1, 1.0)


def test_history_and_save(tmp_path):
    m = ExitCollapseMonitor(output_dir=str(tmp_path / "monitor"))
    for step in range(1, 6):
        m.record(step, 1.8, exit_probs=[0.1, 0.6])
    p = m.save()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert len(data["history"]) == 5
    assert data["history"][-1]["exit_probs"] == [0.1, 0.6]


def test_exit_probability_table():
    m = ExitCollapseMonitor()
    m.record(1, 1.8, exit_probs=[0.2, 0.5])
    table = m.exit_probability_table()
    assert "第 1 遍" in table and "第 2 遍" in table


def test_summary_reports_collapse():
    m = ExitCollapseMonitor(alert_below=1.5)
    m.record(1, 1.0)
    assert "坍缩" in m.summary()
