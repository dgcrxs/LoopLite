"""Agent 场景 DPO 偏好对齐脚本。

超算上执行：
    sbatch hpc/slurm/submit_dpo.sbatch

======================================================================
DPO 的三个关键旋钮（踩过的坑）
======================================================================
  beta:      0.1~0.3。太小 → 偏离参考模型太远，能力崩；太大 → 学不动
  lr:        比 SFT 低一个数量级（5e-6 级别）
  epochs:    1 轮起步，DPO 极易过拟合

======================================================================
评估口径
======================================================================
**不看 loss，看工具调用准确率与 JSON 解析率**。
DPO 的 loss 下降与「能不能正确调工具」相关性很弱，
必须在 eval 里直接跑一批真实工具调用请求来数成功率。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..core.modeling_utils import resolve_model_path
from ..utils.config import load_config, resolve_path
from ..utils.logger import get_logger
from .trainer import setup_lora, setup_quantization

log = get_logger("looplite.dpo")


def load_dpo_dataset(path: str, num_proc: int = 16):
    """加载偏好对数据。格式见 configs/train_dpo_agent.yaml 注释。"""
    from datasets import load_dataset

    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"DPO 数据不存在: {p}（超算无外网，需本机推送）")

    ds = load_dataset("json", data_files=str(p), split="train")

    required = {"prompt", "chosen", "rejected"}
    missing = required - set(ds.column_names)
    if missing:
        raise ValueError(f"DPO 数据缺少必需字段: {missing}，实际字段: {ds.column_names}")

    # 数据质量硬校验：chosen 必须是可解析 JSON（Agent 场景的底线）
    bad = 0
    for ex in ds:
        try:
            json.loads(ex["chosen"])
        except (json.JSONDecodeError, TypeError):
            bad += 1
    if bad:
        log.warning(f"chosen 中有 {bad} 条不是合法 JSON —— Agent 场景这会直接污染偏好信号，建议清洗后再训")
    return ds


def run(config_path: str = "configs/train_dpo_agent.yaml", dry_run: bool = False) -> None:
    cfg = load_config(config_path)
    exp = cfg["experiment"]
    out_dir = exp["output_dir"]
    log.info(f"启动 DPO: {exp['name']} → {out_dir}")

    ds = load_dpo_dataset(cfg["data"]["train_file"], cfg["data"].get("num_proc", 16))
    log.info(f"偏好对数量: {len(ds)}")

    if dry_run:
        log.info("[dry-run] 数据检查通过，不加载模型")
        return

    path = resolve_model_path(cfg["model"].get("local_path"), cfg["model"]["base"])

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=cfg["model"]["trust_remote_code"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = setup_quantization(cfg.get("quantization", {}))
    model = AutoModelForCausalLM.from_pretrained(
        path,
        quantization_config=bnb,
        torch_dtype=getattr(torch, cfg["model"].get("torch_dtype", "bfloat16")),
        attn_implementation=cfg["model"].get("attn_implementation", "flash_attention_2"),
        trust_remote_code=cfg["model"]["trust_remote_code"],
    )
    model.config.use_cache = False

    # DPO 需要参考模型；QLoRA 下与策略模型共享基座以省显存
    ref_model = None if cfg["training"].get("precompute_ref_log_probs") else model

    sft_adapter = cfg["model"].get("sft_adapter")
    if sft_adapter and Path(resolve_path(sft_adapter)).exists():
        # 从已训好的 Agent-SFT 权重继续：先挂 SFT adapter，再在其上叠 DPO 的 LoRA
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(resolve_path(sft_adapter)), is_trainable=True)
        log.info(f"已加载 SFT adapter 作为起点: {sft_adapter}")
    else:
        model = setup_lora(model, cfg.get("lora", {}))
        if sft_adapter:
            log.warning(f"sft_adapter 路径无效，改为从基座直接起 LoRA: {sft_adapter}")

    dpo_cfg = DPOConfig(
        beta=cfg["training"].get("beta", 0.1),
        loss_type=cfg["training"].get("loss_type", "sigmoid"),
        max_length=cfg["data"].get("max_seq_len", 8192),
        max_prompt_length=cfg["data"].get("max_prompt_len", 4096),
        output_dir=out_dir,
        num_train_epochs=cfg["training"].get("num_train_epochs", 1),
        per_device_train_batch_size=cfg["training"].get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=cfg["training"].get("gradient_accumulation_steps", 8),
        learning_rate=cfg["training"].get("learning_rate", 5e-6),
        lr_scheduler_type=cfg["training"].get("lr_scheduler_type", "cosine"),
        warmup_ratio=cfg["training"].get("warmup_ratio", 0.1),
        bf16=cfg["training"].get("bf16", True),
        logging_steps=cfg["training"].get("logging_steps", 10),
        save_steps=cfg["training"].get("save_steps", 100),
        save_total_limit=cfg["training"].get("save_total_limit", 2),
        report_to=cfg["training"].get("report_to", ["tensorboard"]) or [],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_cfg,
        train_dataset=ds,
        processing_class=tokenizer,
    )
    trainer.train()

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    log.info(f"DPO 完成: {out_dir}")
    log.info("⚠️ 下一步必须做：跑一批真实工具调用请求，统计 JSON 解析率与调用准确率，低于 90% 不要上线")


def main() -> int:
    p = argparse.ArgumentParser(description="LoopLite Agent 场景 DPO")
    p.add_argument("--config", default="configs/train_dpo_agent.yaml")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    run(a.config, a.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
