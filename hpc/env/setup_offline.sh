#!/bin/bash
# ============================================================
# 超算离线环境搭建
#
# 前提：本机已执行
#     python scripts/pack_offline.py --include-weights
#     bash scripts/sync_hpc.sh --target user@hpc:/path
# 之后本目录应有：
#     wheels/          ← 全量 wheel
#     requirements-hpc.lock.txt
#
# 用法（在超算上）：
#     bash hpc/env/setup_offline.sh
#
# ⚠️ 超算无外网，任何 pip install xxx 都会卡死在超时。
#    所有依赖必须来自本机打包的 wheels/
# ============================================================

set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$WORKDIR"
echo "工作目录: $WORKDIR"

# ---- 0. 加载模块 ----
if [ -f hpc/env/modules.env ]; then
    source hpc/env/modules.env
else
    echo "⚠️ 未找到 hpc/env/modules.env，跳过 module 加载"
fi

# ---- 1. 环境探测 ----
echo "===== 环境探测 ====="
echo "CUDA 驱动支持上限: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo '无 nvidia-smi')"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "⚠️ 看不到 GPU，检查是否在计算节点"

PYTHON_BIN="${PYTHON_BIN:-python3}"
$PYTHON_BIN --version

# ---- 2. 创建 venv ----
if [ ! -d venv ]; then
    echo "===== 创建 venv ====="
    $PYTHON_BIN -m venv venv
fi
source venv/bin/activate
python -m pip install --no-index --upgrade pip setuptools wheel 2>/dev/null || \
    echo "⚠️ pip 升级失败（离线环境常见，可忽略，继续用自带版本）"

# ---- 3. 离线安装依赖 ----
echo "===== 离线安装依赖 ====="
if [ ! -d wheels ]; then
    echo "❌ 缺少 wheels/ 目录。请先在本机运行 scripts/pack_offline.py 并推送"
    exit 1
fi

# wheel 按「Python+CUDA」分目录存放（wheels/py311-cu124），
# 自动挑出与当前解释器版本匹配的那一个；也可以 WHEELS_DIR=... 手动指定
if [ -n "${WHEELS_DIR:-}" ]; then
    WH="$WHEELS_DIR"
else
    CUR_PY="$(python -c 'import sys;print("py%d%d"%sys.version_info[:2])')"
    WH=""
    for d in wheels/"${CUR_PY}"-*; do
        [ -d "$d" ] && WH="$d" && break
    done
    # 兜底：只有一层没分目录时直接用 wheels/
    [ -z "$WH" ] && ls wheels/*.whl >/dev/null 2>&1 && WH="wheels"
fi

if [ -z "$WH" ] || [ ! -d "$WH" ]; then
    echo "❌ 找不到匹配当前 Python（$(python -V 2>&1)）的 wheel 目录"
    echo "   现有目录:"
    ls -d wheels/*/ 2>/dev/null || echo "   （无子目录）"
    echo "   → 在本机用对应的 --python-version 重新打包，或 WHEELS_DIR=... 手动指定"
    exit 1
fi
echo "使用 wheel 目录: $WH"

LOCK="requirements-hpc.$(basename "$WH").lock.txt"
if [ ! -f "$LOCK" ]; then
    LOCK="$(ls requirements-hpc.*.lock.txt 2>/dev/null | head -1)"
fi
if [ -z "$LOCK" ] || [ ! -f "$LOCK" ]; then
    echo "❌ 缺少锁定清单 requirements-hpc.*.lock.txt"
    exit 1
fi
echo "使用锁定清单: $LOCK"

# --no-index 强制只用本地 wheel，避免尝试联网（离线环境会卡很久）
python -m pip install --no-index --find-links="$WH" -r "$LOCK"

# ---- 4. torch 单独校验 ----
# torch 体积大且必须与 CUDA 匹配，装完必须实测能否用 GPU
echo "===== torch 校验 ====="
python - <<'PY'
import torch
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"可用 GPU: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  [{i}] {torch.cuda.get_device_name(i)} "
              f"{torch.cuda.get_device_properties(i).total_memory/1024**3:.1f}G")
    # 实测一次 CUDA 计算，确认不是「装上了但跑不了」
    x = torch.randn(1024, 1024, device="cuda")
    y = (x @ x).sum().item()
    print(f"CUDA 计算实测通过: {y:.2f}")
else:
    print("❌ torch 看不到 GPU —— 检查 CUDA 版本与 torch 版本是否匹配")
PY

# ---- 5. 关键依赖校验 ----
echo "===== 依赖校验 ====="
python - <<'PY'
mods = ["transformers", "peft", "trl", "accelerate", "datasets", "bitsandbytes"]
for m in mods:
    try:
        mod = __import__(m)
        print(f"  ✓ {m} {getattr(mod, '__version__', '?')}")
    except ImportError as e:
        print(f"  ✗ {m} 未安装: {e}")
PY

# ---- 6. 权重完整性 ----
echo "===== 权重检查 ====="
python -c "
from src.core.modeling_utils import check_local_weights
import json
r = check_local_weights()
print(json.dumps(r, ensure_ascii=False, indent=2))
if not r['exists']:
    print('❌ 本地权重不存在，请先同步 models/base_weights/')
"

echo "===== 完成 ====="
echo "下一步: sbatch hpc/slurm/submit_infer.sbatch"
