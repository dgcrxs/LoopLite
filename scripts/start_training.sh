#!/usr/bin/env bash
# ============================================================
# 一键启动微调
#
# 本机用法（只做数据/配置自检，不真跑）：
#   bash scripts/start_training.sh office --dry-run
#
# 超算用法：直接 sbatch 更合适，见 hpc/slurm/
#   bash scripts/start_training.sh office     # 办公 SFT
#   bash scripts/start_training.sh agent      # Agent DPO
# ============================================================

set -euo pipefail
cd "$(dirname "$0")/.."

SCENE="${1:-office}"
shift || true

case "$SCENE" in
    office)
        MODULE="src.training.sft"
        CONFIG="configs/train_sft_office.yaml"
        ;;
    agent)
        MODULE="src.training.dpo"
        CONFIG="configs/train_dpo_agent.yaml"
        ;;
    *)
        echo "未知场景: $SCENE（可选 office | agent）"
        exit 1
        ;;
esac

echo "训练场景: $SCENE  配置: $CONFIG"

if [ -d .venv ]; then
    source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate
fi

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m "$MODULE" --config "$CONFIG" "$@"

# ⚠️ 训练完必看：防坍缩记录
echo ""
echo "===== 训练结束 ====="
echo "请检查 outputs 下的 monitor/exit_history.json："
echo "  期望退出步数若塌到 1.0 附近 → 模型在跳过循环层偷懒，需调高正则权重"
