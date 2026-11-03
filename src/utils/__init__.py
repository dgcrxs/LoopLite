"""LoopLite 工具层：硬件探测、配置加载、日志、量化辅助。"""

from .hardware import (
    HardwareInfo,
    detect_hardware,
    pick_profile,
    recommend_strategy,
)
from .config import load_config, deep_merge, resolve_path
from .logger import get_logger, setup_logging

__all__ = [
    "HardwareInfo",
    "detect_hardware",
    "pick_profile",
    "recommend_strategy",
    "load_config",
    "deep_merge",
    "resolve_path",
    "get_logger",
    "setup_logging",
]
