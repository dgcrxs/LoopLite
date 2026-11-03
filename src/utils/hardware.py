"""硬件探测与档位推荐。

设计要点：
  1. 本机（2G 显存）也能 import 本模块 —— torch 是可选依赖
  2. 档位选择完全数据驱动，来自 configs/inference_16g.yaml 的 profiles
  3. 超算上优先读 CUDA_VISIBLE_DEVICES / SLURM 环境变量
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .logger import get_logger

log = get_logger("looplite.hardware")


@dataclass
class HardwareInfo:
    cuda_available: bool = False
    device_count: int = 0
    total_mem_gb: float = 0.0
    per_device_mem_gb: list[float] = field(default_factory=list)
    gpu_names: list[str] = field(default_factory=list)
    torch_version: str = "not-installed"
    cuda_version: str = "n/a"
    cpu_count: int = 0
    ram_gb: float = 0.0

    @property
    def min_device_mem_gb(self) -> float:
        return min(self.per_device_mem_gb) if self.per_device_mem_gb else 0.0

    def summary(self) -> str:
        if not self.cuda_available:
            return f"CPU only (torch={self.torch_version}, cpu={self.cpu_count}, ram={self.ram_gb:.1f}G)"
        names = ",".join(sorted(set(self.gpu_names))) or "unknown"
        return (
            f"GPU x{self.device_count} [{names}] "
            f"mem={[round(m, 1) for m in self.per_device_mem_gb]}G "
            f"torch={self.torch_version} cuda={self.cuda_version}"
        )


def detect_hardware() -> HardwareInfo:
    """探测硬件。无 torch 或无 GPU 时安全降级，不抛异常。"""
    info = HardwareInfo()

    try:
        import psutil

        info.cpu_count = psutil.cpu_count(logical=True) or 0
        info.ram_gb = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass

    try:
        import torch

        info.torch_version = torch.__version__
        info.cuda_version = getattr(torch.version, "cuda", "n/a") or "n/a"
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            info.cuda_available = True
            info.device_count = torch.cuda.device_count()
            for i in range(info.device_count):
                free, total = torch.cuda.mem_get_info(i) if hasattr(torch.cuda, "mem_get_info") else (0, 0)
                gb = total / (1024**3) if total else 0.0
                if gb == 0.0:
                    prop = torch.cuda.get_device_properties(i)
                    gb = prop.total_memory / (1024**3)
                info.per_device_mem_gb.append(round(gb, 2))
                info.gpu_names.append(torch.cuda.get_device_name(i))
            info.total_mem_gb = round(sum(info.per_device_mem_gb), 2)
    except ImportError:
        log.warning("未安装 torch，硬件探测降级为 CPU 模式（本机准备阶段属正常）")
    except Exception as e:  # noqa: BLE001
        log.warning(f"GPU 探测失败: {e}")

    return info


# 档位 → 单卡显存下限(GB)。与 configs/inference_16g.yaml 的 profiles 对应
_PROFILE_MIN_MEM_GB = {
    "debug_2g": 0,
    "consumer_16g": 15,
    "a100_40g": 38,
    "a100_80g": 70,
}


def pick_profile(info: HardwareInfo, profiles: dict, explicit: str | None = None) -> str:
    """按显存自动选档。explicit 存在时直接返回（并校验合法性）。"""
    if explicit:
        if explicit not in profiles:
            raise KeyError(f"未知档位 {explicit}，可选: {list(profiles)}")
        return explicit

    if not info.cuda_available:
        return "debug_2g"

    mem = info.min_device_mem_gb
    # 从大到小匹配，保证「够用即最优」
    for name, need in sorted(_PROFILE_MIN_MEM_GB.items(), key=lambda kv: -kv[1]):
        if name in profiles and mem >= need:
            return name
    return "debug_2g"


def recommend_strategy(info: HardwareInfo) -> dict:
    """给出人话版的策略建议，供 CLI 首屏打印。"""
    mem = info.min_device_mem_gb if info.cuda_available else 0.0

    if not info.cuda_available:
        return {
            "mode": "cpu-only",
            "advice": "无可用 GPU。本机定位为准备机：下载权重、打离线包、静态检查。",
            "can_infer": False,
        }
    if mem < 15:
        return {
            "mode": "too-small",
            "advice": f"单卡仅 {mem:.1f}G，装不下 BF16 权重(~20GB)。请改用超算 A100 40G 档。",
            "can_infer": False,
        }
    if mem < 38:
        return {
            "mode": "consumer-16g",
            "advice": "16G 档：需 FP8 权重存储或 CPU offload。注意 GGUF/llama.cpp 目前不支持该架构。",
            "can_infer": True,
            "tensor_parallel": 1,
        }
    if mem < 70:
        return {
            "mode": "a100-40g",
            "advice": "40G 档：BF16 + tp2 最稳；QLoRA 微调单卡可行。",
            "can_infer": True,
            "tensor_parallel": 2,
        }
    return {
        "mode": "a100-80g",
        "advice": "80G 档：单卡 BF16，128K 长上下文可直接开。",
        "can_infer": True,
        "tensor_parallel": 1,
    }


def visible_devices() -> str | None:
    """超算上常被调度系统注入，用于日志溯源。"""
    return os.environ.get("CUDA_VISIBLE_DEVICES")
