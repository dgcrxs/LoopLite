"""仓库布局一致性测试。

防的是一类很容易复发的坑：`.gitignore` 把目录整目录忽略后，
`.gitkeep` 不会跟着 clone 过来，若脚本不显式建目录就会在运行期炸掉。

本测试锁住两件事：
1. 被忽略的运行时目录，setup.sh 必须显式 mkdir
2. 仓库里不能出现权重/venv 之类的大件或环境残留
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 这些目录是运行期必需的，但 .gitignore 会整目录忽略
IGNORED_RUNTIME_DIRS = [
    "models/base_weights",
    "models/quantized",
    "models/loras/office_lora",
    "models/loras/agent_lora",
    "models/tts_weights",
    "data/raw",
    "data/processed/tts_audio_text",
]


def _tracked() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return out.stdout.split()


def test_ignored_runtime_dirs_are_created_by_setup() -> None:
    setup = (ROOT / "scripts" / "setup.sh").read_text(encoding="utf-8")

    missing = [d for d in IGNORED_RUNTIME_DIRS if d not in setup]
    assert not missing, (
        f"这些目录被 .gitignore 忽略、不会被 clone 出来，"
        f"但 scripts/setup.sh 没有创建它们，运行期会「目录不存在」：{missing}"
    )


def test_models_dir_is_wholly_ignored() -> None:
    """models/ 必须整目录忽略——35GB 权重 + 中途 .part 文件，逐扩展名放行会漏。"""
    gi = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "models/" in gi, ".gitignore 缺少 models/ 整目录忽略规则"


def test_no_weights_or_env_in_repo() -> None:
    """入库文件里不该有权重、venv 环境、或本机临时产物。"""
    bad_suffixes = (".safetensors", ".gguf", ".bin", ".pt", ".pth", ".whl")
    bad_fragments = (".venv/", "venv/", "__pycache__/", ".pytest_cache/", ".tmp/")

    offenders: list[str] = []
    for f in _tracked():
        if f.endswith(bad_suffixes) or any(b in f for b in bad_fragments):
            offenders.append(f)
    assert not offenders, f"不应入库的文件: {offenders}"


def test_configs_and_hpc_env_are_tracked() -> None:
    """hpc/env/ 曾被无前缀的 `env/` 规则整个吃掉过，防止复发。"""
    tracked = set(_tracked())
    for expected in (
        "hpc/env/setup_offline.sh",
        "hpc/env/probe.sh",
        "hpc/env/modules.env",
        "configs/model_base.yaml",
        ".gitattributes",
    ):
        assert expected in tracked, f"{expected} 未被 git 跟踪（可能被 .gitignore 误伤）"


def test_gitattributes_forces_lf_for_shell_scripts() -> None:
    """Windows 检出 CRLF 会让 .sh 在 Linux 上报 `\\r: command not found`。"""
    ga = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "eol=lf" in ga
    assert "*.sh" in ga
