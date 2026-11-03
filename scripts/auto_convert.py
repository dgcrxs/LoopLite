#!/usr/bin/env python
"""等待基座权重下载完成，校验完整性，然后自动转 BF16。

用途：权重下载要跑数小时，人工盯着不现实。本脚本后台运行，
      下载一结束就接着做 FP32→BF16 转换（上传量 35.5GB → 约 18GB）。

用法：
    python scripts/auto_convert.py                    # 等待 + 校验 + 转 BF16
    python scripts/auto_convert.py --no-wait          # 假定已下载完，直接转换
    python scripts/auto_convert.py --dtype float16    # 目标精度
    python scripts/auto_convert.py --timeout-min 300  # 最长等待时间

退出码：
    0 转换成功 / 1 超时或下载不完整 / 2 转换失败
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SRC = ROOT / "models" / "base_weights" / "SenseNova-Loop-9B-Preview"
DEFAULT_DST = ROOT / "models" / "quantized" / "SenseNova-Loop-9B-BF16"

# 官方仓库是 8 个分片 + mtp.safetensors。少一个都算不完整。
EXPECTED_SHARDS = 9
# FP32 落盘约 35.5GB，低于 30GB 一定是下载中断
MIN_SIZE_GB = 30.0


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def dir_bytes(p: Path) -> int:
    # 只统计分片与配置文件，跳过 .cache（内含失败尝试的残留）
    return sum(f.stat().st_size for f in p.rglob("*")
               if f.is_file() and ".cache" not in f.parts)


def snapshot(src: Path) -> tuple[int, int, int]:
    """返回 (已完成分片数, 进行中 .part 数, 总字节)"""
    done = len(list(src.glob("*.safetensors")))
    doing = len(list(src.glob("*.part")))
    return done, doing, dir_bytes(src)


def wait_done(src: Path, timeout_min: int, poll_sec: int) -> bool:
    print(f"等待下载完成：目标 {EXPECTED_SHARDS} 个分片，最长等 {timeout_min} 分钟")
    deadline = time.time() + timeout_min * 60
    last_report = 0.0

    while time.time() < deadline:
        done, doing, size = snapshot(src)
        now = time.time()

        if now - last_report >= 120:  # 每 2 分钟报一次进度
            print(f"  [{time.strftime('%H:%M:%S')}] "
                  f"已完成 {done}/{EXPECTED_SHARDS} 分片，{doing} 个下载中，{human(size)}")
            last_report = now

        # 主判据：所有 .part 消失 + 分片数达标
        if doing == 0 and done >= EXPECTED_SHARDS:
            print(f"  ✓ 下载完成：{done} 个分片，{human(size)}")
            return True

        time.sleep(poll_sec)

    done, doing, size = snapshot(src)
    print(f"  ✗ 超时：仍有 {doing} 个未完成，已完成 {done}/{EXPECTED_SHARDS}，{human(size)}")
    return False


def verify(src: Path) -> tuple[bool, str]:
    """校验下载完整性：分片数 + 体积 + 索引文件一致性。"""
    shards = sorted(src.glob("*.safetensors"))
    size_gb = dir_bytes(src) / (1024 ** 3)
    if len(shards) < EXPECTED_SHARDS:
        return False, f"分片仅 {len(shards)} 个，预期 {EXPECTED_SHARDS}"
    if size_gb < MIN_SIZE_GB:
        return False, f"体积仅 {size_gb:.1f}GB，低于 {MIN_SIZE_GB}GB，疑似不完整"

    idx = src / "model.safetensors.index.json"
    if idx.exists():
        try:
            data = json.loads(idx.read_text(encoding="utf-8"))
            need = set(data["weight_map"].values())
            have = {f.name for f in shards}
            missing = need - have
            if missing:
                return False, f"索引声明的分片缺失: {sorted(missing)}"
        except (json.JSONDecodeError, KeyError) as e:
            return False, f"索引文件解析失败: {e}"

    if not (src / "config.json").exists():
        return False, "缺少 config.json"
    return True, f"{len(shards)} 个分片，{size_gb:.2f}GB，索引一致"


def main() -> int:
    ap = argparse.ArgumentParser(description="下载完成后自动转 BF16")
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--dst", default=str(DEFAULT_DST))
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--no-wait", action="store_true", help="不等待，直接转换")
    ap.add_argument("--timeout-min", type=int, default=360, help="最长等待分钟数")
    ap.add_argument("--poll-sec", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    print(f"源: {src}")
    print(f"目标: {dst}  ({a.dtype})")

    if not src.exists():
        print(f"❌ 源目录不存在: {src}")
        return 1

    if not a.no_wait:
        if not wait_done(src, a.timeout_min, a.poll_sec):
            return 1

    ok, msg = verify(src)
    print(f"\n===== 完整性校验 =====\n  {'✓' if ok else '✗'} {msg}")
    if not ok:
        return 1

    if a.dry_run:
        print("  （--dry-run，未执行转换）")
        return 0

    print(f"\n===== 开始转换 {a.dtype} =====")
    cmd = [sys.executable, "-m", "src.utils.quantize", "convert",
           "--src", str(src), "--dst", str(dst), "--dtype", a.dtype]
    print("  " + " ".join(cmd[1:]))

    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        print(f"\n❌ 转换失败（退出码 {r.returncode}）")
        return 2

    out_gb = dir_bytes(dst) / (1024 ** 3) if dst.exists() else 0
    print(f"\n✓ 转换完成，耗时 {time.time() - t0:.0f}s，输出 {human(out_gb)}")
    print(f"  目录: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
