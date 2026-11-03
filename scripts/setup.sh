#!/usr/bin/env bash
# ============================================================
# 本机（Windows）准备环境一键安装
#
# 本机定位：只做下载 / 打包 / 静态检查，不装 torch（2G 显存跑不动也没意义）
# 用法（Git Bash）： bash scripts/setup.sh
# ============================================================

set -euo pipefail

cd "$(dirname "$0")/.."
echo "工作目录: $(pwd)"

# ---- 1. venv ----
if [ ! -d .venv ]; then
    echo "===== 创建 venv ====="
    python -m venv .venv
fi

# Windows 的 venv 脚本路径
if [ -f .venv/Scripts/activate ]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

# ---- 2. 本机最小依赖 ----
echo "===== 安装本机依赖 ====="
python -m pip install --upgrade pip
python -m pip install -r requirements-local.txt

# ---- 3. 目录完整性 ----
echo "===== 校验目录结构 ====="
python scripts/env_check.py

echo ""
echo "===== 完成 ====="
echo "下一步："
echo "  1) python scripts/download_models.py"
echo "  2) python scripts/pack_offline.py --include-weights"
echo "  3) bash scripts/sync_hpc.sh --target user@hpc:/path/to/LoopLite"
