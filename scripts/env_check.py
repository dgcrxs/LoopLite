#!/usr/bin/env python
"""本机环境自检 —— 提交作业前的最后一道防线。

在超算上跑一次失败作业要排队等几十分钟，
而绝大多数失败都能在本机提前发现：配置写错、路径不对、权重没下全、
wheel 拉成了 Windows 版……这些都在这里一次性查出来。

用法：
    python scripts/env_check.py
    python scripts/env_check.py --strict     # 有问题则以非零码退出
"""

from __future__ import annotations

import os
import platform
import py_compile
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REQUIRED_DIRS = [
    "configs", "data/raw", "data/processed", "data/scripts",
    "models/base_weights", "models/quantized", "models/loras",
    "src/core", "src/inference", "src/training", "src/utils",
    "src/apps/office", "src/apps/agent", "src/apps/tts",
    "deployment", "hpc/slurm", "hpc/env", "scripts", "tests", "docs",
]

REQUIRED_FILES = [
    "README.md", "LICENSE", ".gitignore", "requirements.txt",
    "requirements-local.txt", "pyproject.toml",
    "configs/model_base.yaml", "configs/inference_16g.yaml",
    "configs/train_sft_office.yaml", "configs/train_dpo_agent.yaml",
    "configs/tts_flowmatch.yaml", "configs/hpc.yaml",
    "hpc/slurm/submit_infer.sbatch", "hpc/env/setup_offline.sh",
    "scripts/download_models.py", "scripts/pack_offline.py", "scripts/sync_hpc.sh",
]

OK, WARN, FAIL = "✓", "⚠️", "✗"


def check_dirs() -> list[tuple[str, str, str]]:
    out = []
    for d in REQUIRED_DIRS:
        p = ROOT / d
        out.append((OK if p.is_dir() else FAIL, d, "目录" if p.is_dir() else "缺失"))
    return out


def check_files() -> list[tuple[str, str, str]]:
    out = []
    for f in REQUIRED_FILES:
        p = ROOT / f
        out.append((OK if p.is_file() else FAIL, f, "文件" if p.is_file() else "缺失"))
    return out


def check_configs() -> list[tuple[str, str, str]]:
    out = []
    try:
        import yaml
    except ImportError:
        return [(FAIL, "PyYAML", "未安装，无法校验配置")]
    for name in ["model_base", "inference_16g", "train_sft_office", "train_dpo_agent", "tts_flowmatch", "hpc"]:
        p = ROOT / "configs" / f"{name}.yaml"
        try:
            yaml.safe_load(p.read_text(encoding="utf-8"))
            out.append((OK, f"configs/{name}.yaml", "YAML 合法"))
        except Exception as e:  # noqa: BLE001
            out.append((FAIL, f"configs/{name}.yaml", str(e)))
    return out


def check_syntax() -> list[tuple[str, str, str]]:
    """编译全部 Python 文件，能在没装 torch 的情况下抓语法错误。

    不用 cfile=os.devnull —— Windows 上 devnull 是 'nul'，
    py_compile 会拒绝写入（FileExistsError），改用临时文件。
    """
    import tempfile

    out = []
    files = [p for p in ROOT.rglob("*.py")
             if not any(x in p.parts for x in (".venv", "venv", "__pycache__", ".workbuddy"))]
    bad = []
    with tempfile.TemporaryDirectory() as td:
        tmp_pyc = Path(td) / "check.pyc"
        for f in files:
            try:
                py_compile.compile(str(f), doraise=True, cfile=str(tmp_pyc))
            except py_compile.PyCompileError as e:
                bad.append((FAIL, str(f.relative_to(ROOT)), str(e).splitlines()[-1][:80]))
            except Exception as e:  # noqa: BLE001
                bad.append((FAIL, str(f.relative_to(ROOT)), f"{type(e).__name__}: {e}"[:80]))
    if bad:
        out.extend(bad)
    out.append((OK if not bad else FAIL, f"Python 语法检查 ({len(files)} 个文件)",
                "全部通过" if not bad else f"{len(bad)} 个文件有语法错误"))
    return out


def check_weights() -> list[tuple[str, str, str]]:
    """校验基座权重。注意 checkpoint 是 FP32 存储，完整体积 35.5GB。"""
    out = []
    w = ROOT / "models" / "base_weights" / "SenseNova-Loop-9B-Preview"
    EXPECTED_GB, EXPECTED_SHARDS = 34.0, 8

    if not w.exists():
        out.append((WARN, "基座权重", "未下载（python scripts/download_models.py）"))
    else:
        st = list(w.glob("*.safetensors"))
        parts = list(w.glob("*.safetensors.part"))
        size = sum(f.stat().st_size for f in w.rglob("*") if f.is_file()) / 1024**3
        if parts and not st:
            out.append((WARN, "基座权重", f"下载中：{len(parts)} 个分片未完成，当前 {size:.1f}GB"))
        elif not st:
            out.append((WARN, "基座权重", "目录存在但无 safetensors"))
        elif len(st) < EXPECTED_SHARDS:
            out.append((FAIL, "基座权重",
                        f"仅 {len(st)}/{EXPECTED_SHARDS} 个分片，{size:.1f}GB，下载不完整"))
        elif size < EXPECTED_GB:
            out.append((FAIL, "基座权重",
                        f"仅 {size:.1f}GB，预期 {EXPECTED_GB}GB+（FP32 存储，不是 BF16 的 19GB）"))
        else:
            out.append((OK, "基座权重", f"{len(st)} 个分片，{size:.1f}GB（FP32 原始件）"))
            # 是否已转 BF16（上传这个，体积减半）
            bf16 = ROOT / "models" / "quantized" / "SenseNova-Loop-9B-BF16"
            if (bf16 / "config.json").exists():
                bsz = sum(f.stat().st_size for f in bf16.rglob("*") if f.is_file()) / 1024**3
                out.append((OK, "BF16 转换版", f"{bsz:.1f}GB（建议上传这个，而非 FP32 原始件）"))
            else:
                out.append((WARN, "BF16 转换版",
                            "未转换，上传会是 35.5GB 而非 17.8GB（见 docs/offline_prep.md 第 3 步）"))
    return out


def check_env() -> list[tuple[str, str, str]]:
    out = [
        (OK, "操作系统", f"{platform.system()} {platform.release()}"),
        (OK, "Python", sys.version.split()[0]),
    ]
    # 本机 GPU
    try:
        import subprocess

        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            out.append((WARN, "本机 GPU", r.stdout.strip().replace("\n", " | ") + " —— 本机不跑推理，仅作准备机"))
        else:
            out.append((OK, "本机 GPU", "无 NVIDIA GPU（符合预期，本机只做准备）"))
    except Exception:  # noqa: BLE001
        out.append((OK, "本机 GPU", "无 nvidia-smi（符合预期）"))

    # 磁盘
    total, used, free = shutil.disk_usage(ROOT)
    out.append((OK if free > 40 * 1024**3 else WARN, "磁盘剩余", f"{free / 1024**3:.0f}GB"))
    return out


def check_network() -> list[tuple[str, str, str]]:
    """检查下载通道是否可达（决定要不要走镜像）。"""
    out = []
    try:
        import urllib.request

        for name, url in [("HuggingFace", "https://huggingface.co"),
                          ("hf-mirror", "https://hf-mirror.com")]:
            try:
                req = urllib.request.Request(url, method="HEAD")
                with urllib.request.urlopen(req, timeout=8) as r:
                    out.append((OK, name, f"可达 (HTTP {r.status})"))
            except Exception as e:  # noqa: BLE001
                out.append((WARN, name, f"不可达: {type(e).__name__} —— 建议用镜像"))
    except Exception as e:  # noqa: BLE001
        out.append((WARN, "网络检查", str(e)))
    return out


def check_bundle() -> list[tuple[str, str, str]]:
    out = []
    wheels = ROOT / "wheels"
    if wheels.exists():
        # wheel 按组合分子目录（wheels/py311-cu124/），两种布局都要数
        whls = list(wheels.glob("*.whl")) + list(wheels.glob("*/*.whl"))
        n = len(whls)
        win = [f.name for f in whls if "win" in f.name][:3]
        if win:
            out.append((FAIL, "离线 wheel", f"发现 Windows 版 wheel {win} —— 超算装不上，重新打包"))
        elif n == 0:
            out.append((WARN, "离线 wheel", "目录存在但为空（打包进行中或未成功）"))
        else:
            subs = sorted({f.parent.name for f in whls if f.parent != wheels})
            out.append((OK, "离线 wheel", f"{n} 个" + (f"（{', '.join(subs)}）" if subs else "")))
    else:
        out.append((WARN, "离线 wheel", "未生成（python scripts/pack_offline.py）"))

    locks = sorted(ROOT.glob("requirements-hpc*.lock.txt"))
    out.append((OK if locks else WARN, "锁定清单",
                "已生成: " + ", ".join(p.name for p in locks) if locks else "未生成"))
    return out


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    sections = [
        ("目录结构", check_dirs),
        ("关键文件", check_files),
        ("配置文件", check_configs),
        ("代码语法", check_syntax),
        ("基座权重", check_weights),
        ("运行环境", check_env),
        ("网络通道", check_network),
        ("离线打包", check_bundle),
    ]

    n_fail = n_warn = 0
    blockers: list[str] = []
    for title, fn in sections:
        print(f"\n===== {title} =====")
        for status, name, msg in fn():
            print(f"  {status} {name}: {msg}")
            if status == FAIL:
                n_fail += 1
                blockers.append(name)
            elif status == WARN:
                n_warn += 1
                # 这几项未就绪时，超算上一定装不起来/跑不起来，属于硬阻塞
                if name in ("离线 wheel", "锁定清单", "基座权重"):
                    blockers.append(name)

    print("\n" + "=" * 50)
    print(f"自检完成：{n_fail} 项失败，{n_warn} 项提醒")
    if blockers:
        print(f"✗ 还有 {len(blockers)} 项阻塞未解决：{', '.join(dict.fromkeys(blockers))}")
        print("  这些不解决，推上去也跑不起来。详见 docs/offline_prep.md")
        return 1 if n_fail else 0
    print("✓ 可以提交作业了")
    return 1 if (n_fail and a.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
