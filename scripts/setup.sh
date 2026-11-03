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

# ---- 1. 目录结构 ----
# 注意：models/ 与 data/ 被 .gitignore 整目录忽略（35GB 权重绝不能进仓），
# 因此 .gitkeep 不会跟着 clone 过来，目录必须在这里显式建出来。
# 否则训练脚本保存 LoRA、转换脚本写 BF16 权重时会因目录不存在而失败。
echo "===== 初始化目录结构 ====="
mkdir -p \
    configs \
    data/raw \
    data/processed/tts_audio_text \
    models/base_weights \
    models/quantized \
    models/loras/office_lora \
    models/loras/agent_lora \
    models/tts_weights \
    logs
echo "  ✓ 目录就绪"

# ---- 2. venv ----
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

# ---- 3. 本机最小依赖 ----
echo "===== 安装本机依赖 ====="
python -m pip install --upgrade pip
python -m pip install -r requirements-local.txt

# ---- 4. 目录完整性 ----
echo "===== 校验目录结构 ====="
python scripts/env_check.py

echo ""
echo "===== 完成 ====="
echo "下一步："
echo "  1) python scripts/download_models.py"
echo "  2) python scripts/pack_offline.py --include-weights"
echo "  3) bash scripts/sync_hpc.sh --target user@hpc:/path/to/LoopLite"
