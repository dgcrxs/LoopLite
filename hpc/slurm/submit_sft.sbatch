#!/bin/bash
# ============================================================
# Slurm 作业：办公场景 SFT（QLoRA，单卡 A100 40G）
#
# 用法：
#   sbatch hpc/slurm/submit_sft.sbatch
#
# 显存账（40G 单卡，QLoRA + 梯度检查点）：
#   4bit 基座 ~6G + LoRA ~0.5G + 16K 长序列激活 ~8G + 优化器 ~2G ≈ 17G
#   → 40G 够用，若 OOM 先把 max_seq_len 降到 8192
# ============================================================
#SBATCH --job-name=looplite-sft
#SBATCH --partition=gpu            # ★ 按集群实际修改
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/sft-%j.out
#SBATCH --error=logs/slurm/sft-%j.err
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
export OMP_NUM_THREADS=16

# ---- 先做 dry-run，确认数据与配置没问题再烧机时 ----
echo "===== 数据/配置自检 ====="
python -m src.training.sft --config configs/train_sft_office.yaml --dry-run

echo "===== 开始 SFT ====="
python -m src.training.sft --config configs/train_sft_office.yaml

echo "===== 完成，检查防坍缩记录 ====="
# ⚠️ 训练完必须看这个文件：期望退出步数是否塌到 1.0 附近
if [ -f models/loras/office_lora/monitor/exit_history.json ]; then
    echo "退出历史已生成，请用 python -c \"import json;d=json.load(open('models/loras/office_lora/monitor/exit_history.json'));print(d['alerts'])\" 查看告警"
else
    echo "⚠️ 未找到 exit_history.json：说明模型未暴露 expected_exit_step，"
    echo "   防坍缩监控未生效，需人工确认循环层确实在起作用"
fi
echo "结束时间: $(date)"
