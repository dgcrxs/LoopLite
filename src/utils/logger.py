"""统一日志。

超算作业里 stdout 会被调度系统收进日志文件，
因此这里额外按进程/作业号落一份本地文件，方便事后排查。
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_LOGGER_CACHE: dict[str, logging.Logger] = {}
_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"


def _log_dir() -> Path:
    # 超算上 SLURM_JOB_ID 存在时，按作业号分文件，避免多作业互相覆盖
    job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("PBS_JOBID") or "local"
    base = Path(os.environ.get("LOOPLITE_LOG_DIR", "logs"))
    d = base / f"job-{job_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def setup_logging(level: str | int = logging.INFO) -> None:
    """配置根 logger：控制台 + 文件。"""
    root = logging.getLogger()
    root.setLevel(level)
    if root.handlers:
        return

    fmt = logging.Formatter(_DEFAULT_FORMAT, datefmt="%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    try:
        fh = logging.FileHandler(_log_dir() / "run.log", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError:
        # 只读文件系统等边缘情况，仅保留控制台
        pass


def get_logger(name: str) -> logging.Logger:
    if name not in _LOGGER_CACHE:
        setup_logging()
        _LOGGER_CACHE[name] = logging.getLogger(name)
    return _LOGGER_CACHE[name]
