#!/bin/bash
# ============================================================
# Slurm 作业：推理冒烟测试（阶段一验收）
# 目标：让模型在超算上输出第一个字
#
# 用法：
#   sbatch hpc/slurm/submit_infer.sbatch
#   sbatch --gres=gpu:2 hpc/slurm/submit_infer.sbatch   # 覆盖资源
#
# 提交前必做：
#   1) 本机已执行 python scripts/pack_offline.py + scripts/sync_hpc.sh
#   2) 超算已执行 bash hpc/env/setup_offline.sh
#   3) 按集群实际修改 #SBATCH 的 partition / qos / account
# ============================================================
#SBATCH --job-name=looplite-infer
#SBATCH --partition=gpu            # ★ 按集群实际分区名修改
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2               # A100 40G × 2（BF16 tp2）
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/infer-%j.out
#SBATCH --error=logs/slurm/infer-%j.err
# #SBATCH --qos=normal             # ★ 需要时取消注释
# #SBATCH --account=YOUR_ACCOUNT   # ★ 需要时取消注释

set -euo pipefail

echo "===== 作业信息 ====="
echo "JobID: $SLURM_JOB_ID"
echo "节点: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "起始时间: $(date)"

# ---- 1. 环境 ----
# 超算用 module 管理 CUDA；如集群不用 module，注释掉这几行
# ★ 核对：基座官方环境为 CUDA 13.0，A100 常见 11.8/12.x，
#          torch 需与驱动兼容，装错直接报 CUDA driver version is insufficient
if [ -f hpc/env/modules.env ]; then
    source hpc/env/modules.env
fi

# ---- 2. Python 环境（离线 venv，由 setup_offline.sh 建好）----
export WORKDIR="${HPC_WORKDIR:-$SLURM_SUBMIT_DIR}"
cd "$WORKDIR"
source venv/bin/activate

# ---- 3. 离线环境变量（超算无外网，必须设）----
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=16

# ---- 4. 提交前自检（配置与权重完整性）----
echo "===== 环境自检 ====="
python -m src.inference.cli dry-run

# ---- 5. 跑推理 ----
echo "===== 开始推理 ====="
python -m src.inference.cli smoke \
    --profile a100_40g \
    --scene office \
    --loop-pass 2 \
    --max-new-tokens 64

echo "===== 完成 ====="
echo "结束时间: $(date)"
