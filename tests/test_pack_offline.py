"""离线打包脚本的元数据测试（不联网、不下载，只校验映射与解析逻辑）。

为什么值得测：`TORCH_BY_CUDA` 一旦写错，打出来的包在超算上会
`No matching distribution found`，而那时已经排了几十分钟队。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pack_offline import (  # noqa: E402
    COMPILE_HARD,
    EXCLUDE_FROM_CODE,
    TORCH_BY_CUDA,
    TORCH_PACKAGES,
    parse_requirements,
    pypi_platforms,
    torch_index_platforms,
)


def test_platform_tags_cover_all_manylinux_generations():
    """PyPI 的 wheel 有 manylinux1/2010/2014/2_17/2_28 五种标签，
    只给一个会漏掉很多包（torch 2.6 就是 manylinux1 标签）。"""
    tags = pypi_platforms("x86_64")
    for need in ("manylinux1_x86_64", "manylinux2014_x86_64", "manylinux_2_28_x86_64"):
        assert need in tags


def test_torch_index_tag_is_not_manylinux():
    """PyTorch 官方索引的 wheel 用的是 linux_x86_64，不是 manylinux。
    传错标签 → No matching distribution，且表现为静默卡住。"""
    tags = torch_index_platforms("x86_64")
    assert tags == ["linux_x86_64"]
    assert not any("manylinux" in t for t in tags)


def test_platform_tags_follow_arch():
    assert "manylinux2014_aarch64" in pypi_platforms("aarch64")
    assert torch_index_platforms("aarch64") == ["linux_aarch64"]


def test_torch_cuda_mapping_is_monotonic():
    """CUDA 越新，允许的 torch 越新——映射顺序不能反。"""
    def key(v: str) -> list[int]:
        return [int(x) for x in v.split(".")]

    ordered = sorted(TORCH_BY_CUDA.items(), key=lambda kv: int(kv[0]))
    torch_versions = [key(v) for _, v in ordered]
    assert torch_versions == sorted(torch_versions), f"映射非单调: {ordered}"


def test_default_cuda_exists():
    """脚本默认用 cu124，映射里必须有。"""
    assert "124" in TORCH_BY_CUDA


def test_all_mapped_torch_satisfy_transformers_518():
    """transformers 5.18 要求 torch>=2.5，映射里不能出现更低版本。"""
    for cu, v in TORCH_BY_CUDA.items():
        major, minor = (int(x) for x in v.split(".")[:2])
        assert (major, minor) >= (2, 5), f"cu{cu} 映射到 torch {v}，低于 transformers 5.18 的下限 2.5"


def test_parse_requirements_splits_torch():
    normal, torch_pkgs, _ = parse_requirements(ROOT / "requirements.txt")
    assert normal, "普通依赖不应为空"
    assert torch_pkgs, "应识别出 torch 系依赖"
    names = {p.split(">=")[0].split("==")[0].strip().lower() for p in torch_pkgs}
    assert names <= TORCH_PACKAGES, f"误判为 torch 系的包: {names - TORCH_PACKAGES}"
    # transformers 必须钉版本，否则超算上可能装到 API 不兼容的新版
    joined = " ".join(normal)
    assert "transformers==5.18.0" in joined, "transformers 必须钉死 5.18.0"


def test_requirements_no_inline_comment_leak():
    normal, torch_pkgs, _ = parse_requirements(ROOT / "requirements.txt")
    for p in normal + torch_pkgs:
        assert "#" not in p, f"行内注释未清理: {p}"
        assert p.strip() == p, f"存在首尾空格: {p!r}"


def test_exclude_covers_heavy_dirs():
    """代码包里绝不能夹带 venv / 权重 / 临时目录。"""
    for d in (".venv", "models", "wheels", "dist", ".tmp"):
        assert d in EXCLUDE_FROM_CODE


def test_compile_hard_has_hints():
    """难装的包都要有人话解释，不然打包失败时无从下手。"""
    for pkg in ("flash-attn", "deepspeed", "bitsandbytes"):
        assert pkg in COMPILE_HARD
        assert len(COMPILE_HARD[pkg]) > 10
