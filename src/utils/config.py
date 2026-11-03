"""YAML 配置加载与合并。

约定：
  - configs/ 下的配置可以用 `defaults:` 引用其他配置作为基底
  - 相对路径一律相对项目根目录解析，避免 cwd 不同导致路径失效
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


class ConfigError(RuntimeError):
    pass


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并，override 覆盖 base。列表整体替换，不做拼接。"""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_raw(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _looks_like_config_ref(s: str) -> bool:
    """判断一个值是否是「配置文件引用」。

    防御性设计：曾经踩过的坑——把 `defaults:` 当普通配置段用，
    里面的 `max_new_tokens: 1024` 被当成配置文件名去找 `configs/1024`。
    所以只接受看起来像路径的值。
    """
    return isinstance(s, str) and (s.endswith((".yaml", ".yml")) or "/" in s or "\\" in s)


def _resolve_defaults(data: dict[str, Any], seen: set[Path]) -> dict[str, Any]:
    """处理 `defaults:` 引用（支持相对路径）。"""
    defaults = data.pop("defaults", None)
    merged: dict[str, Any] = {}
    if defaults:
        items = defaults if isinstance(defaults, list) else [defaults]
        # defaults 可能是 dict（如 {model: ../configs/model_base.yaml}）或字符串列表
        refs: list[str] = []
        for it in items:
            if isinstance(it, dict):
                refs.extend(str(v) for v in it.values() if _looks_like_config_ref(v))
            elif _looks_like_config_ref(it):
                refs.append(str(it))
        for ref in refs:
            p = (CONFIG_DIR / ref).resolve()
            if p in seen:
                continue
            seen.add(p)
            merged = deep_merge(merged, _resolve_defaults(_load_raw(p), seen))
    return deep_merge(merged, data)


def load_config(name: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """按文件名加载配置。既接受 'inference_16g.yaml'，也接受绝对路径。"""
    path = Path(name)
    if not path.is_absolute():
        path = CONFIG_DIR / name
        if not path.suffix:
            path = path.with_suffix(".yaml")
    cfg = _resolve_defaults(_load_raw(path), {path.resolve()})
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def resolve_path(p: str | Path) -> Path:
    """把配置里的相对路径解析为项目根目录下的绝对路径。"""
    path = Path(p)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def expand_env(cfg: dict[str, Any]) -> dict[str, Any]:
    """展开 ${VAR} 形式的环境变量引用（用于 hpc.yaml 的账号/路径）。"""
    import os
    import re

    pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

    def _sub(v):
        if isinstance(v, str):
            return pattern.sub(lambda m: os.environ.get(m.group(1), ""), v)
        if isinstance(v, dict):
            return {k: _sub(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_sub(x) for x in v]
        return v

    return _sub(cfg)
