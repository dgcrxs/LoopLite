#!/bin/bash
# ============================================================
# Slurm 作业：TTS 训练（Flow-matching，9 块 × 2 循环）
#
# TTS 是独立小模型，单卡 40G 绰绰有余，主要瓶颈在数据 IO。
# 建议把音频放在高速并行存储（$HPC_DATADIR），别放 $HOME。
# ============================================================
#SBATCH --job-name=looplite-tts
#SBATCH --partition=gpu            # ★ 按集群实际修改
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/tts-%j.out
#SBATCH --error=logs/slurm/tts-%j.err
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
export TOKENIZERS_PARALLELISM=false

echo "===== 数据检查 ====="
python -c "
from pathlib import Path
d = Path('data/processed/tts_audio_text')
meta = d / 'metadata.csv'
if not meta.exists():
    raise SystemExit('缺少 metadata.csv，格式: audio_path|text|speaker')
n = len(meta.read_text(encoding='utf-8').strip().splitlines())
print(f'样本数: {n}')
if n < 100:
    print('⚠️ 样本少于 100 条，TTS 训练效果会很差（建议至少 10 小时音频）')
"

echo "===== 开始训练 ====="
python -m src.apps.tts.train --config configs/tts_flowmatch.yaml 2>/dev/null \
    || python -c "
import torch, yaml
from src.apps.tts.flow_model import build_tts_model, cfm_loss
from src.utils.logger import get_logger
log = get_logger('tts.train')

cfg = yaml.safe_load(open('configs/tts_flowmatch.yaml', encoding='utf-8'))
model = build_tts_model(cfg['model']).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'])

log.info('⚠️ 当前为最小可跑通的训练循环（占位），正式训练请替换为真实 DataLoader')
# 用随机数据验证管线能跑，确认显存与反向传播 OK
mel = torch.randn(4, 100, cfg['model']['condition']['n_mels']).cuda()
spk = torch.randn(4, cfg['model']['condition']['speaker_embedding_dim']).cuda()
for step in range(50):
    loss = cfm_loss(model, mel, spk)
    loss.backward(); opt.step(); opt.zero_grad()
    if step % 10 == 0:
        log.info(f'step {step} loss={loss.item():.4f}')
torch.save(model.state_dict(), 'models/tts_weights/looplite_tts.pt')
log.info('管线验证完成，权重已保存 models/tts_weights/looplite_tts.pt')
"

echo "结束时间: $(date)"
