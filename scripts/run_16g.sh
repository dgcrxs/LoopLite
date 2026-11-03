#!/usr/bin/env bash
# ============================================================
# 消费级 16G 显卡启动推理（RTX 4060 Ti 16G / 4080）
#
# ⚠️ 当前本机是 GTX 950M 2G，跑不动。本脚本供后续换机或借卡时使用。
# ⚠️ 重要前提：llama.cpp / GGUF 目前【不支持】sensenova_loop 架构，
#    所以这里走的是 HF Transformers 原生推理，不是 llama.cpp。
# ============================================================

set -euo pipefail
cd "$(dirname "$0")/.."

SCENE="${1:-office}"
LOOP_PASS="${2:-2}"

echo "场景: $SCENE  循环遍数: $LOOP_PASS"
echo "档位: consumer_16g（BF16 装不下时自动 FP8/offload）"

source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m src.inference.cli chat \
    --profile consumer_16g \
    --scene "$SCENE" \
    --loop-pass "$LOOP_PASS"
