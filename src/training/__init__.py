"""训练管线：SFT / DPO / 防坍缩监控。

⚠️ 延迟导入设计（重要）：
`monitor`（防坍缩监控）是**纯 Python，不依赖 torch**，
目的是让本机（无 torch）也能导入它做数据复盘。

而 `sft` / `dpo` 会经 `src.core` 拉起 torch。
若在这里直接 `from .sft import ...`，那么 `from src.training.monitor import ...`
就会被 torch 连带拖起来 —— 本机没 torch 时直接 ImportError。

所以 sft/dpo 走 PEP 562 的模块级 __getattr__ 延迟导入。
"""

from .trainer import (
    build_trainer,
    build_training_args,
    load_dataset_from_jsonl,
    setup_lora,
    setup_quantization,
    build_monitor,
)
from .monitor import ExitCollapseMonitor, extract_expected_exit_step

__all__ = [
    "build_trainer",
    "build_training_args",
    "load_dataset_from_jsonl",
    "setup_lora",
    "setup_quantization",
    "build_monitor",
    "ExitCollapseMonitor",
    "extract_expected_exit_step",
    "run_sft",
    "run_dpo",
]

_LAZY = {
    "run_sft": (".sft", "run"),
    "run_dpo": (".dpo", "run"),
}


def __getattr__(name: str):
    """延迟导入需要 torch 的模块，保持 monitor 可独立使用。"""
    if name in _LAZY:
        import importlib

        module, attr = _LAZY[name]
        return getattr(importlib.import_module(module, __name__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
