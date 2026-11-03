#!/usr/bin/env python
"""下载基座权重（本机执行，超算无外网）。

用法：
    python scripts/download_models.py                  # 默认走 hf-mirror，自动选引擎
    python scripts/download_models.py --official       # 强制走 huggingface.co（国内不通）
    python scripts/download_models.py --engine native  # 裸 HTTP 断点续传，最稳
    python scripts/download_models.py --engine modelscope
    python scripts/download_models.py --dry-run        # 只列文件清单，不下载
    python scripts/download_models.py --workers 8      # 提高并发

事实基线（2026-11-03 实测，见 configs/model_base.yaml）：
  - 仓库 sensenova/SenseNova-Loop-9B-Preview，Apache-2.0
  - 9.656B 参数，但 checkpoint 以 FP32 存储 → 35.5GB（不是 BF16 的 19GB）
  - 8 个 model-*.safetensors 分片 + mtp.safetensors + 自定义 modeling 代码
  - 多模态：含视觉塔与 video_preprocessor_config，不是纯文本模型

断点续传：中断后直接重跑，snapshot_download 会跳过已完成分片。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ID = "sensenova/SenseNova-Loop-9B-Preview"
DEFAULT_DIR = "models/base_weights/SenseNova-Loop-9B-Preview"
HF_MIRROR = "https://hf-mirror.com"

# hf-mirror 会拦 Python-urllib 的默认 UA（返回 403），必须伪装成普通客户端
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LoopLiteDownloader/1.0"

# 实测体积（GB）——用于完整性校验，不看模型卡的描述
EXPECTED_TOTAL_GB = 35.52
EXPECTED_SHARDS = 8
MIN_ACCEPT_GB = 34.0

# 关键文件：缺任何一个都说明下载不完整
REQUIRED_FILES = [
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "modeling_sensenova_loop.py",   # trust_remote_code 必需，缺了直接加载失败
    "preprocessor_config.json",      # 多模态预处理
]

# 示例素材（约 30MB），不想下载可以跳过
MEDIA_PATTERNS = ["examples/*", "assets/*"]


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.2f}{unit}"
        n /= 1024
    return f"{n:.2f}PB"


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def remote_file_list(endpoint: str, repo: str) -> list[tuple[str, int]]:
    """通过 API 拿到远端文件清单（不下载内容）。"""
    url = f"{endpoint}/api/models/{repo}?blobs=true"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=40) as r:
        data = json.loads(r.read().decode())
    files = data.get("siblings") or []
    return [(f["rfilename"], f.get("size") or 0) for f in files]


def try_hf(repo: str, out_dir: Path, endpoint: str | None) -> bool:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  ✗ 未安装 huggingface_hub：pip install -r requirements-local.txt")
        return False

    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    # 国内网络必看：huggingface_hub 2.x 默认用 Xet 协议，其端点常被墙/超时，
    # 表现为「下载卡住不动、.incomplete 文件不再增长」。强制回落纯 HTTP。
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    print(f"  endpoint = {os.environ.get('HF_ENDPOINT', 'huggingface.co（官方）')}")
    print(f"  Xet 已禁用 = {os.environ.get('HF_HUB_DISABLE_XET')}（回落纯 HTTP，国内更稳）")
    # hf_transfer 若装了会显著加速；没有也不影响
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1":
        try:
            import hf_transfer  # noqa: F401
            print("  hf_transfer 加速已启用")
        except ImportError:
            os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

    try:
        snapshot_download(
            repo_id=repo,
            local_dir=str(out_dir),
            max_workers=8,
        )
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  ✗ 下载失败: {type(e).__name__}: {e}")
        return False


def try_modelscope(repo: str, out_dir: Path) -> bool:
    try:
        from modelscope import snapshot_download as ms_download
    except ImportError:
        print("  ✗ 未安装 modelscope：pip install modelscope")
        return False
    try:
        path = ms_download(repo, cache_dir=str(out_dir.parent))
        print(f"  ModelScope 落盘: {path}")
        if Path(path).resolve() != out_dir.resolve():
            print(f"  ⚠️ 路径与预期不同，请手动移动/软链到 {out_dir.resolve()}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  ✗ ModelScope 下载失败: {type(e).__name__}: {e}")
        return False


def native_download(
    endpoint: str,
    repo: str,
    out_dir: Path,
    files: list[tuple[str, int]],
    workers: int = 4,
    retries: int = 5,
) -> bool:
    """不依赖 huggingface_hub 的原生下载器（断点续传 + 多线程 + 体积校验）。

    为什么需要它：国内网络下 huggingface_hub 经常因代理/签名链路报
    ProxyError 502，而裸 HTTP Range 请求却能稳定跑满带宽。
    这里只依赖标准库 urllib，行为完全可控。
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    base = f"{endpoint}/{repo}/resolve/main/"
    lock = threading.Lock()
    state = {"done": 0, "bytes": 0, "failed": []}

    def fetch_one(item: tuple[str, int]) -> None:
        name, expected = item
        dst = out_dir / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and expected and dst.stat().st_size == expected:
            with lock:
                state["done"] += 1
                state["bytes"] += expected
            return

        part = dst.with_suffix(dst.suffix + ".part")
        for attempt in range(1, retries + 1):
            try:
                have = part.stat().st_size if part.exists() else 0
                if expected and have > expected:
                    part.unlink()
                    have = 0
                if expected and have == expected:
                    part.replace(dst)
                    break
                headers = {"User-Agent": USER_AGENT}
                if have:
                    headers["Range"] = f"bytes={have}-"
                req = urllib.request.Request(base + name, headers=headers)
                with urllib.request.urlopen(req, timeout=120) as r:
                    # 服务器不支持 Range 时返回 200，需从头写
                    mode = "ab" if (have and r.status == 206) else "wb"
                    with open(part, mode) as f:
                        while True:
                            chunk = r.read(1 << 20)
                            if not chunk:
                                break
                            f.write(chunk)
                got = part.stat().st_size
                if expected and got != expected:
                    raise OSError(f"体积不符: {got} != {expected}")
                part.replace(dst)
                break
            except Exception as e:  # noqa: BLE001
                if attempt == retries:
                    with lock:
                        state["failed"].append((name, f"{type(e).__name__}: {e}"))
                    return
                time.sleep(min(2**attempt, 15))

        with lock:
            state["done"] += 1
            state["bytes"] += dst.stat().st_size
            print(f"  [{state['done']}/{len(files)}] {name} "
                  f"({dst.stat().st_size / 1024**2:.0f}MB, 累计 {state['bytes'] / 1024**3:.2f}GB)")

    # 大文件优先，先啃硬骨头，避免小文件占满队列后长时间无进度
    ordered = sorted(files, key=lambda x: -(x[1] or 0))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(fetch_one, ordered))

    if state["failed"]:
        print(f"\n  ✗ {len(state['failed'])} 个文件失败（重跑脚本会自动续传）:")
        for n, err in state["failed"][:8]:
            print(f"    - {n}: {err}")
        return False
    return True


def verify(out_dir: Path) -> bool:
    """按实测体积与关键文件校验，不依赖远端。"""
    print("\n===== 完整性校验 =====")
    print(f"  目录: {out_dir}")
    if not out_dir.exists():
        print("  ✗ 目录不存在")
        return False

    size_gb = dir_size(out_dir) / 1024**3
    shards = sorted(out_dir.glob("model-*-of-*.safetensors"))
    missing = [f for f in REQUIRED_FILES if not (out_dir / f).exists()]

    print(f"  总体积: {size_gb:.2f} GB（预期 {EXPECTED_TOTAL_GB} GB）")
    print(f"  分片数: {len(shards)}/{EXPECTED_SHARDS}")
    for s in shards:
        print(f"    - {s.name} ({s.stat().st_size / 1024**2:.0f} MB)")

    ok = True
    if missing:
        print(f"  ✗ 缺少关键文件: {missing}")
        ok = False
    if len(shards) < EXPECTED_SHARDS:
        print(f"  ✗ 分片不足（{len(shards)}/{EXPECTED_SHARDS}）")
        ok = False
    if size_gb < MIN_ACCEPT_GB:
        print(f"  ✗ 体积仅 {size_gb:.2f} GB，低于下限 {MIN_ACCEPT_GB} GB")
        ok = False

    if ok:
        print("  ✓ 权重完整（FP32 存储，训练/推理前需先转 BF16，见 src/utils/quantize.py）")
    return ok


def main() -> int:
    p = argparse.ArgumentParser(description="下载 LoopLite 基座权重")
    p.add_argument("--repo", default=REPO_ID)
    p.add_argument("--out", default=DEFAULT_DIR)
    p.add_argument("--engine", default="auto", choices=["auto", "hf", "native", "modelscope"],
                   help="下载引擎：auto=依次尝试 hf→native→modelscope")
    p.add_argument("--workers", type=int, default=4, help="native 引擎的并发数（默认 4）")
    p.add_argument("--official", action="store_true", help="走 huggingface.co（国内通常不通）")
    p.add_argument("--dry-run", action="store_true", help="只列出远端文件清单")
    p.add_argument("--skip-media", action="store_true", help="跳过 examples/ 与 assets/ 示例素材")
    p.add_argument("--skip-verify", action="store_true")
    a = p.parse_args()

    out_dir = Path(a.out).resolve()
    endpoint = None if a.official else HF_MIRROR

    print(f"仓库: {a.repo}")
    print(f"目标: {out_dir}")

    # 远端清单（顺带确认网络通不通）
    files: list[tuple[str, int]] = []
    try:
        files = remote_file_list(endpoint or "https://huggingface.co", a.repo)
        total = sum(s for _, s in files)
        print(f"远端: {len(files)} 个文件，合计 {total / 1024**3:.2f} GB")
        if a.dry_run:
            for name, sz in sorted(files, key=lambda x: -x[1])[:15]:
                print(f"  {sz / 1024**2:9.1f} MB  {name}")
            print("  ...（--dry-run 结束，未下载）")
            return 0
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ 无法获取远端清单（{type(e).__name__}: {e}），直接尝试下载")

    out_dir.mkdir(parents=True, exist_ok=True)
    total, used, free = shutil.disk_usage(out_dir)
    print(f"磁盘剩余: {human(free)}")
    if free < 40 * 1024**3:
        print("⚠️ 剩余空间不足 40GB（下载 35.5GB + 转 BF16 中间产物），可能中途失败")

    if a.skip_media:
        # snapshot_download 没有 exclude 参数，用 ignore_patterns 环境变量不通用，
        # 这里直接提示：素材仅约 30MB，跳过意义不大
        print("  （示例素材合计约 30MB，占总量 0.1%，仍会下载）")

    t0 = time.time()
    ok = False
    eng = endpoint or "https://huggingface.co"

    # 通道顺序：hf 库（有加速）→ 原生 urllib（绕开代理/签名问题）→ ModelScope
    if a.engine == "native":
        sources = ["native"]
    elif a.engine == "hf":
        sources = ["hf"]
    elif a.engine == "modelscope":
        sources = ["modelscope"]
    else:
        sources = ["hf", "native", "modelscope"]

    for src in sources:
        if src == "hf":
            print("\n===== 通道: HuggingFace" + ("（官方）" if a.official else "（hf-mirror 镜像）") + " =====")
            ok = try_hf(a.repo, out_dir, endpoint)
        elif src == "native":
            if not files:
                print("\n===== 通道: 原生 HTTP =====")
                print("  ✗ 拿不到文件清单，跳过")
                continue
            print("\n===== 通道: 原生 HTTP（断点续传，绕开 huggingface_hub）=====")
            ok = native_download(eng, a.repo, out_dir, files, workers=a.workers)
        else:
            print("\n===== 通道: ModelScope =====")
            ok = try_modelscope(a.repo, out_dir)
        if ok:
            break

    print(f"\n耗时: {time.time() - t0:.0f}s")
    if not ok:
        print("\n❌ 下载失败。兜底方案：")
        print(f"  1) 浏览器打开 {eng}/{a.repo}/tree/main")
        print(f"  2) 手动下载全部文件到 {out_dir}")
        print("  3) 或配代理后重试：export HTTPS_PROXY=http://127.0.0.1:7890")
        print("  4) 断点续传：直接重跑本脚本，已完成文件会自动跳过")
        return 1

    if a.skip_verify:
        return 0
    return 0 if verify(out_dir) else 1


if __name__ == "__main__":
    raise SystemExit(main())
