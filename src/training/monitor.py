"""🚨 防「退出坍缩」雷达 —— 本项目最关键的监控脚本。

======================================================================
为什么单独抽一个文件
======================================================================
循环架构训练中，模型可能学会「偷懒」：在第 1 遍循环后就退出，
把 Looped Transformer 退化成普通网络。此时 **Loss 曲线看起来完全正常**，
甚至下降得更快——因为少算一遍，拟合更容易。

所以：**不要只看 Loss**。必须实时盯「平均退出步数（Expected Exit Step）」：
  - 训练中期该值突然骤降 → 模型在偷懒，立刻介入
  - 该值长期贴着 1.0 → 循环架构白搭了，等于训了个 9B 稠密模型

======================================================================
三级防线
======================================================================
  1. 监控（本文件）：实时打印 + 历史落盘 + 骤降告警
  2. 正则（SmoothExitLoss）：惩罚过早退出
  3. 结构（ExitRouter 上限）：硬限制最少遍数
"""

from __future__ import annotations

import json
from pathlib import Path

from ..utils.logger import get_logger

log = get_logger("looplite.monitor")


class ExitCollapseMonitor:
    """退出坍缩监控器。

    两种用法：
      A) 作为 HF TrainerCallback（推荐）：
             trainer.add_callback(ExitCollapseMonitor(...).as_callback())
      B) 手动喂数据：
             monitor.record(step=100, expected_exit_step=1.83)
    """

    def __init__(
        self,
        alert_below: float = 1.2,
        drop_ratio_threshold: float = 0.25,
        window: int = 50,
        output_dir: str = "outputs/monitor",
        strict: bool = False,
    ):
        """
        Args:
            alert_below: 期望退出步数低于此值即告警
            drop_ratio_threshold: 相对窗口内最高值下降超过该比例 → 判定为骤降
            window: 骤降检测的滑动窗口大小
            output_dir: 历史记录落盘位置（tensorboard 之外的第二份证据）
            strict: True 时检测到坍缩直接抛异常中断训练
        """
        self.alert_below = alert_below
        self.drop_ratio = drop_ratio_threshold
        self.window = window
        self.output_dir = Path(output_dir)
        self.strict = strict
        self.history: list[dict] = []
        self._alerts: list[str] = []

    # ---------- 数据接入 ----------

    def record(self, step: int, expected_exit_step: float, exit_probs: list[float] | None = None, loss: float | None = None) -> None:
        entry = {
            "step": step,
            "expected_exit_step": round(float(expected_exit_step), 4),
            "exit_probs": exit_probs or [],
            "loss": loss,
        }
        self.history.append(entry)

        if len(self.history) % 10 == 0:
            log.info(self.format_status())

        self._check(entry)

    def _check(self, entry: dict) -> None:
        v = entry["expected_exit_step"]

        if v < self.alert_below:
            self._alert(f"step {entry['step']}: 期望退出步数 {v:.3f} < {self.alert_below}，模型疑似跳过循环层")

        # 骤降检测：与窗口内峰值比较
        if len(self.history) >= self.window:
            win = [h["expected_exit_step"] for h in self.history[-self.window:]]
            peak = max(win)
            if peak > 0 and (peak - v) / peak > self.drop_ratio:
                self._alert(
                    f"step {entry['step']}: 退出步数骤降 {peak:.3f} → {v:.3f} "
                    f"(-{(peak - v) / peak * 100:.0f}%)，典型坍缩前兆"
                )

    def _alert(self, msg: str) -> None:
        self._alerts.append(msg)
        log.warning(f"[防坍缩] {msg}")
        log.warning("[防坍缩] 对策：调高 loop_control.exit_regularization_weight，或关闭自适应退出硬设 loop_pass")
        if self.strict:
            raise RuntimeError(f"退出坍缩中止训练：{msg}")

    # ---------- 输出 ----------

    def format_status(self) -> str:
        if not self.history:
            return "[防坍缩] 暂无数据"
        recent = [h["expected_exit_step"] for h in self.history[-20:]]
        avg = sum(recent) / len(recent)
        latest = self.history[-1]
        return (
            f"[防坍缩] step={latest['step']} 期望退出步数={latest['expected_exit_step']:.3f} "
            f"(近20均值 {avg:.3f}, 告警线 {self.alert_below}) 告警数={len(self._alerts)}"
        )

    def exit_probability_table(self) -> str:
        """打印最近一次的退出概率分布（每层/每遍）。"""
        last = self.history[-1] if self.history else None
        if not last or not last["exit_probs"]:
            return "[防坍缩] 无退出概率数据（模型未启用自适应退出）"
        lines = [f"[防坍缩] step={last['step']} 退出概率分布："]
        for i, p in enumerate(last["exit_probs"], start=1):
            bar = "█" * int(p * 40)
            lines.append(f"  第 {i} 遍后停止: {p:.3f} {bar}")
        return "\n".join(lines)

    def save(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        p = self.output_dir / "exit_history.json"
        p.write_text(
            json.dumps({"config": {"alert_below": self.alert_below, "window": self.window},
                        "alerts": self._alerts,
                        "history": self.history}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info(f"[防坍缩] 历史已保存: {p}")
        return p

    @property
    def alerts(self) -> list[str]:
        return list(self._alerts)

    def summary(self) -> str:
        if not self.history:
            return "[防坍缩] 未采集到数据"
        vals = [h["expected_exit_step"] for h in self.history]
        return (
            f"[防坍缩] 汇总: 步数范围 step {self.history[0]['step']}~{self.history[-1]['step']}\n"
            f"  期望退出步数: min={min(vals):.3f} max={max(vals):.3f} "
            f"均值={sum(vals) / len(vals):.3f}\n"
            f"  告警 {len(self._alerts)} 次"
            + ("\n  结论: 疑似发生退出坍缩，循环架构未发挥作用" if self._alerts else "\n  结论: 未发现坍缩迹象")
        )

    # ---------- 接入 HF Trainer ----------

    def as_callback(self):
        """包装成 transformers.TrainerCallback。"""
        from transformers import TrainerCallback

        monitor = self

        class _Cb(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                logs = logs or {}
                step = int(state.global_step)
                # 期望退出步数由模型侧写入 logs；键名兼容几种常见写法
                v = logs.get("expected_exit_step") or logs.get("exit_step") or logs.get("mean_exit_step")
                if v is not None:
                    monitor.record(step, float(v), loss=logs.get("loss"))

            def on_train_end(self, args, state, control, **kwargs):
                log.info(monitor.summary())
                monitor.save()

        return _Cb()


def extract_expected_exit_step(model) -> float | None:
    """从模型上取期望退出步数。

    官方自定义实现可能不暴露该字段，这时返回 None —— 监控会静默跳过，
    但请在作业日志里确认「无退出概率数据」这句话是否出现，出现了就意味着
    这一版实现没有自适应退出，防坍缩防线只剩 SmoothExitLoss 与硬设 loop_pass。
    """
    for attr in ("last_expected_exit_step", "expected_exit_step"):
        if hasattr(model, attr):
            v = getattr(model, attr)
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
    return None
