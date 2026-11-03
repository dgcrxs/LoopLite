#!/bin/bash
# ============================================================
# Slurm 作业：Agent 场景 DPO 偏好对齐（QLoRA，单卡 A100 40G）
#
# 显存注意：DPO 要同时跑策略模型与参考模型，显存约为 SFT 的 1.6 倍。
# 40G 卡上若 OOM：
#   1) 把 per_device_train_batch_size 降到 1
#   2) 开 precompute_ref_log_probs=true（预计算参考 logits，训练时卸载参考模型）
#   3) 降 max_seq_len 到 4096
# ============================================================
#SBATCH --job-name=looplite-dpo
#SBATCH --partition=gpu            # ★ 按集群实际修改
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/dpo-%j.out
#SBATCH --error=logs/slurm/dpo-%j.err
# #SBATCH --qos=normal
# #SBATCH --account=YOUR_ACCOUNT

set -euo pipefail

echo "JobID: $SLURM_JOB_ID 节点: $SLURM_NODELIST GPU: $CUDA_VISIBLE_DEVICES"

if [ -f hpc/env/modules.env ]; then
    source hpc/env/modules.env
fi

export WORKDIR="${HPC_WORKDIR:-$SLURM_SUBMIT_DIR}"
cd "$WORKDIR"
source venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "===== 偏好数据自检（会校验 chosen 是否为合法 JSON）====="
python -m src.training.dpo --config configs/train_dpo_agent.yaml --dry-run

echo "===== 开始 DPO ====="
python -m src.training.dpo --config configs/train_dpo_agent.yaml

echo "===== 完成 ====="
echo "⚠️ 下一步：跑真实工具调用请求，统计 JSON 解析率与调用准确率，<90% 不要上线"
echo "结束时间: $(date)"
