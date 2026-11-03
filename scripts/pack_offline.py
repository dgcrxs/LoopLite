#!/usr/bin/env python
"""离线打包：把超算需要的一切（代码 + 依赖 + 权重）装进一个目录。

======================================================================
⚠️ 最大的坑：跨平台
======================================================================
本机是 Windows，超算是 Linux x86_64 + CUDA。
直接 `pip download -r requirements.txt` 会拉到 **Windows 版 wheel**，
推到超算上装不了（或装上跑不了）。

本脚本强制指定目标平台：
    --platform manylinux2014_x86_64 --python-version 3.11 --only-binary=:all:

torch 另走 PyTorch 官方索引（PyPI 上的 torch 是 CPU 版）：
    --index-url https://download.pytorch.org/whl/cu124

======================================================================
产物
======================================================================
    wheels/                     全量 wheel（超算 pip install --no-index 用）
    requirements-hpc.lock.txt   锁定版本清单
    dist/looplite-code.tar.gz   代码包（不含权重）
    offline_bundle/MANIFEST.json 清单（含体积、校验、失败项）

用法：
    python scripts/pack_offline.py                    # 只打依赖 + 代码
    python scripts/pack_offline.py --include-weights  # 连权重一起打包
    python scripts/pack_offline.py --cuda 118         # A100 常见 CUDA 11.8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 需要从 PyTorch 官方索引下载的（PyPI 上是 CPU 版）
TORCH_PACKAGES = {"torch", "torchaudio", "torchvision"}

# CUDA 索引 → 该索引上「与 transformers 5.18 兼容、且驱动要求尽量低」的 torch 版本。
# 实测核对（2026-11）：
#   cu118 最高 2.7.1 / cu124 最高 2.6.0 / cu128 有 2.11.0 / cu130 有 2.14.1
# transformers 5.18 只要求 torch>=2.5，所以不必强上 CUDA 13。
# 默认取各索引里能满足下限的**最低**版本，兼容性最稳；cu128 取 2.11 与官方对齐。
TORCH_BY_CUDA = {
    "118": "2.6.0",     # 驱动 ≥450
    "121": "2.6.0",     # 驱动 ≥530
    "124": "2.6.0",     # 驱动 ≥550（默认）
    "126": "2.7.1",     # 驱动 ≥560
    "128": "2.11.0",    # 驱动 ≥570 ← 与官方验证环境一致
    "129": "2.11.0",    # 驱动 ≥575
    "130": "2.14.1",    # 驱动 ≥580
}

# 平台标签（关键坑！）
#
# pip 的 --platform 是**精确匹配 wheel 文件名里的平台段**，不做兼容推导。
# 两个来源的标签完全不同：
#   PyTorch 官方索引  → linux_x86_64        （不带 manylinux 前缀）
#   PyPI              → manylinux1_x86_64 / manylinux2010_x86_64 /
#                       manylinux2014_x86_64 / manylinux_2_17_x86_64 / manylinux_2_28_x86_64
# 只传 manylinux2014 去拉 torch，会「No matching distribution」静默失败，
# 表现为解析卡很久却没有文件落盘。
def pypi_platforms(arch: str = "x86_64") -> list[str]:
    return [
        f"manylinux1_{arch}",
        f"manylinux2010_{arch}",
        f"manylinux2014_{arch}",
        f"manylinux_2_17_{arch}",
        f"manylinux_2_28_{arch}",
    ]


def torch_index_platforms(arch: str = "x86_64") -> list[str]:
    return [f"linux_{arch}"]
COMPILE_HARD = {
    "flash-attn": "需源码编译（A100 上耗时 10~30 分钟），或改用 sdpa 注意力",
    "deepspeed": "需编译算子，若不做全参微调可去掉",
    "bitsandbytes": "需匹配 CUDA 版本，失败时改用 torch 原生量化",
    "vocos": "TTS 声码器，可选",
    "chromadb": "RAG 向量库，可选，不用可去掉",
    "librosa": "TTS 用，纯 Python 依赖，一般没问题",
}

EXCLUDE_FROM_CODE = {
    ".venv", "venv", "__pycache__", ".git", "models", "data",
    "dist", "offline_bundle", "wheels", "logs", ".pytest_cache",
    "outputs", "checkpoints", ".workbuddy", ".tmp",
}


def sh(cmd: list[str], cwd: Path | None = None, timeout: int = 3600) -> tuple[int, str]:
    print(f"  $ {' '.join(cmd)}")
    try:
        r = subprocess.run(cmd, cwd=str(cwd or ROOT), capture_output=True, text=True,
                           timeout=timeout, encoding="utf-8", errors="replace")
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return 1, "超时"


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.2f}{u}"
        n /= 1024
    return f"{n:.2f}PB"


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def parse_requirements(path: Path) -> tuple[list[str], list[str], list[str]]:
    """返回 (普通包, torch 包, 注释掉的可选包)"""
    normal, torch_pkgs, optional = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        # 去掉行内注释
        s = s.split("  #")[0].split(" #")[0].strip()
        if not s:
            continue
        name = s.split(">=")[0].split("==")[0].split("[")[0].strip().lower()
        if name in TORCH_PACKAGES:
            torch_pkgs.append(s)
        else:
            normal.append(s)
    return normal, torch_pkgs, optional


def download_wheels(packages: list[str], out_dir: Path, python_version: str,
                    platforms: list[str],
                    index_url: str = "https://pypi.org/simple",
                    extra_index: str | None = None) -> tuple[list[str], list[str]]:
    """下载 wheel 到 out_dir。返回 (成功, 失败)。

    platforms 可以给多个，pip 会把它们都当作可接受的目标平台。
    index_url 是主索引：torch 那一趟必须设成 PyTorch 官方索引，
    否则 pip 会去拉 PyPI 的 torch 索引页（上百 MB HTML），在慢网下表现为「卡住不动」。
    """
    if not packages:
        return [], []
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "pip", "download",
        "-d", str(out_dir),
        "--python-version", python_version,
        "--only-binary=:all:",
        "--no-cache-dir",
        "--index-url", index_url,
    ]
    for p in platforms:
        cmd += ["--platform", p]
    if extra_index:
        cmd += ["--extra-index-url", extra_index]
    cmd += packages

    code, out = sh(cmd)
    if code == 0:
        return packages, []
    # 整体失败时把 pip 的真实报错打出来，否则完全看不到为什么失败
    tail = "\n".join(out.strip().splitlines()[-6:])
    print(f"  ⚠️ 整体解析失败，逐个重试。pip 尾部输出:\n{tail}")

    # 整体失败时逐个重试，隔离出具体哪个包有问题
    ok, failed = [], []
    for pkg in packages:
        c2 = cmd[: -len(packages)] + [pkg]
        rc, c2_out = sh(c2)
        (ok if rc == 0 else failed).append(pkg)
        if rc != 0:
            err = "\n".join(c2_out.strip().splitlines()[-3:])
            print(f"  ✗ {pkg} 下载失败: {err}")
    return ok, failed


def pack_code(dist_dir: Path) -> Path:
    """把代码打成 tar.gz（排除权重/数据/venv）。"""
    dist_dir.mkdir(parents=True, exist_ok=True)
    tar_path = dist_dir / "looplite-code.tar.gz"

    def _filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(info.name).parts
        if any(p in EXCLUDE_FROM_CODE for p in parts):
            return None
        if "__pycache__" in info.name or info.name.endswith(".pyc"):
            return None
        return info

    with tarfile.open(tar_path, "w:gz") as tar:
        for item in sorted(ROOT.iterdir()):
            if item.name in EXCLUDE_FROM_CODE:
                continue
            tar.add(item, arcname=item.name, filter=_filter)
    print(f"  ✓ 代码包: {tar_path} ({human(tar_path.stat().st_size)})")
    return tar_path


def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="LoopLite 离线打包")
    ap.add_argument("--cuda", default="124", help="超算 CUDA 版本，如 118/121/124（决定 torch 索引）")
    ap.add_argument("--python-version", default="3.11", help="超算 Python 版本")
    ap.add_argument("--arch", default="x86_64", help="目标 CPU 架构（x86_64 / aarch64）")
    ap.add_argument("--include-weights", action="store_true", help="把基座权重也打进包")
    ap.add_argument("--skip-deps", action="store_true", help="跳过依赖下载（只打代码）")
    ap.add_argument("--skip-code", action="store_true")
    a = ap.parse_args()

    print("=" * 60)
    print("LoopLite 离线打包")
    print(f"  本机平台: {platform.system()} {platform.machine()}")
    print(f"  目标平台: {a.arch} manylinux / Python {a.python_version} / CUDA {a.cuda}")
    print("=" * 60)

    manifest: dict = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target": {"arch": a.arch, "python": a.python_version, "cuda": a.cuda},
        "wheels": {"ok": [], "failed": []},
        "notes": [],
    }

    # ---------- 1. 依赖 ----------
    if not a.skip_deps:
        print("\n===== 第 1 步：下载 wheel =====")
        # 按「Python + CUDA」分目录：换参数重打时不会把两套 torch 混在一起
        # （混在一起 pip 可能挑错版本，且 lock 清单会被污染）
        wheel_sub = f"py{a.python_version.replace('.', '')}-cu{a.cuda}"
        wheels_dir = ROOT / "wheels" / wheel_sub
        manifest["wheels_dir"] = f"wheels/{wheel_sub}"
        normal, torch_pkgs, _ = parse_requirements(ROOT / "requirements.txt")

        print(f"普通依赖 {len(normal)} 个，torch 系 {len(torch_pkgs)} 个")

        # 1a) torch 走 PyTorch 官方 CUDA 索引，版本按 CUDA 索引钉死
        torch_ok, torch_failed = [], []
        if torch_pkgs:
            url = f"https://download.pytorch.org/whl/cu{a.cuda}"
            pin = TORCH_BY_CUDA.get(a.cuda)
            if pin is None:
                print(f"  ⚠️ 未登记的 CUDA 索引 cu{a.cuda}，回退 2.6.0；"
                      f"已知: {sorted(TORCH_BY_CUDA)}")
                pin = "2.6.0"
            # 统一钉版本，避免 torch / torchaudio 版本错位导致 audio 算子缺失
            pinned = [f"torch=={pin}"]
            if any(p.lower().startswith("torchaudio") for p in torch_pkgs):
                pinned.append(f"torchaudio=={pin}")
            if any(p.lower().startswith("torchvision") for p in torch_pkgs):
                pinned.append(f"torchvision=={pin}")
            print(f"\n-- torch 系（索引: {url}，版本钉死 {pin}）--")
            # 关键：主索引直接指向 PyTorch 官方索引。
            # 若把 PyPI 当主索引，pip 会先拉 PyPI 的 /simple/torch/（上百 MB HTML），
            # 慢网下看起来就是「卡住不动」。
            # 官方索引自己托管了 nvidia-* / triton（manylinux 标签），deps 能就地解析完。
            torch_ok, torch_failed = download_wheels(
                pinned, wheels_dir, a.python_version,
                torch_index_platforms(a.arch) + pypi_platforms(a.arch),
                index_url=url,
            )
            if torch_failed:
                manifest["notes"].append(
                    f"torch {pin} 下载失败 {torch_failed}：确认 CUDA {a.cuda} 索引是否有该版本；"
                    f"可访问 {url} 手动确认可用版本后改 TORCH_BY_CUDA"
                )

        # 1b) 其余走 PyPI
        print("\n-- 其余依赖（PyPI）--")
        ok, failed = download_wheels(normal, wheels_dir, a.python_version, pypi_platforms(a.arch), None)

        manifest["wheels"]["ok"] = sorted(set(ok + torch_ok))
        manifest["wheels"]["failed"] = sorted(set(failed + torch_failed))

        # 生成锁定清单（同样按组合命名，避免不同版本互相覆盖）
        whls = sorted(wheels_dir.glob("*.whl"))
        lock_name = f"requirements-hpc.{wheel_sub}.lock.txt"
        lock = ROOT / lock_name
        # 用完整文件名（含版本）而不是包名，否则离线安装时无法钉版本
        lock_lines = []
        for w in whls:
            stem = w.name[: -len(".whl")]
            base = stem.split("-")[0]
            rest = stem[len(base) + 1:]
            ver = rest.split("-")[0]
            lock_lines.append(f"{base}=={ver}")
        lock.write_text(
            f"# 超算离线安装清单（{wheel_sub}，由 scripts/pack_offline.py 生成）\n"
            f"# 安装: pip install --no-index --find-links=wheels/{wheel_sub} -r {lock_name}\n"
            + "\n".join(sorted(set(lock_lines))) + "\n",
            encoding="utf-8",
        )
        print(f"\n  ✓ wheel 总数: {len(whls)}，体积 {human(dir_size(wheels_dir))}")
        print(f"  ✓ 锁定清单: {lock_name}")

        if manifest["wheels"]["failed"]:
            print("\n  ⚠️ 以下包下载失败，需人工处理：")
            for f in manifest["wheels"]["failed"]:
                base = f.split(">=")[0].split("==")[0].lower()
                tip = COMPILE_HARD.get(base, "检查平台是否有对应 wheel，或改用源码包在超算上现场编译")
                print(f"    - {f}: {tip}")

    # ---------- 2. 代码 ----------
    if not a.skip_code:
        print("\n===== 第 2 步：打包代码 =====")
        tar = pack_code(ROOT / "dist")
        manifest["code_tarball"] = {"name": tar.name, "size": human(tar.stat().st_size), "sha256": sha256_file(tar)}

    # ---------- 3. 权重 ----------
    weights_dir = ROOT / "models" / "base_weights" / "SenseNova-Loop-9B-Preview"
    # checkpoint 是 FP32 存的，完整体积 35.5GB（不是 BF16 的 19GB）
    WEIGHTS_MIN_GB = 34.0
    if weights_dir.exists() and any(weights_dir.glob("*.safetensors")):
        size = dir_size(weights_dir)
        complete = size > WEIGHTS_MIN_GB * 1024**3
        manifest["weights"] = {
            "path": str(weights_dir.relative_to(ROOT)),
            "size": human(size),
            "complete": complete,
            "dtype": "float32",
            "note": "FP32 存储；上超算后先跑 src/utils/quantize.py 转 BF16(17.8GB)",
        }
        print(f"\n===== 第 3 步：权重 =====\n  已存在: {human(size)} {'✓' if complete else f'⚠️ 不完整（<{WEIGHTS_MIN_GB}GB）'}")
        if a.include_weights:
            print("  权重将随目录整体推送（不额外压缩，35GB 压缩收益低且耗时）")
    else:
        manifest["weights"] = {"path": str(weights_dir.relative_to(ROOT)), "exists": False}
        print("\n===== 第 3 步：权重 =====\n  ⚠️ 未下载，请先运行 python scripts/download_models.py")

    # ---------- 4. 清单 ----------
    bundle_dir = ROOT / "offline_bundle"
    bundle_dir.mkdir(exist_ok=True)
    (bundle_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n===== 完成 =====\n  清单: {bundle_dir / 'MANIFEST.json'}")
    print("\n下一步：")
    print("  bash scripts/sync_hpc.sh --target user@hpc:/path/to/LoopLite")
    print("  然后在超算上: bash hpc/env/setup_offline.sh")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
