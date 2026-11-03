"""办公场景 SFT 训练脚本。

超算上执行：
    sbatch hpc/slurm/submit_sft.sbatch
或交互调试：
    srun --gpus=1 python -m src.training.sft --config configs/train_sft_office.yaml
"""

from __future__ import annotations

import argparse

from ..core.modeling_utils import load_base_model, load_tokenizer, resolve_model_path
from ..utils.config import load_config
from ..utils.logger import get_logger
from .trainer import build_trainer, load_dataset_from_jsonl, setup_lora, setup_quantization

log = get_logger("looplite.sft")


def run(config_path: str = "configs/train_sft_office.yaml", max_samples: int | None = None, dry_run: bool = False) -> None:
    cfg = load_config(config_path)
    exp = cfg["experiment"]
    out_dir = exp["output_dir"]
    log.info(f"启动 SFT: {exp['name']} → {out_dir}")

    # ---- 1. 数据与分词器 ----
    path = resolve_model_path(cfg["model"].get("local_path"), cfg["model"]["base"])
    tokenizer = load_tokenizer(path, trust_remote_code=cfg["model"]["trust_remote_code"])

    dataset = load_dataset_from_jsonl(
        cfg["data"]["train_file"],
        tokenizer,
        max_seq_len=cfg["data"].get("max_seq_len", 16384),
        fmt=cfg["data"].get("format", "alpaca"),
        num_proc=cfg["data"].get("num_proc", 16),
    )
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    if dry_run:
        log.info(f"[dry-run] 配置与数据检查通过：{len(dataset)} 条样本，不加载模型")
        return

    # ---- 2. 模型（QLoRA 或全参）----
    quant_cfg = cfg.get("quantization", {})
    bnb = setup_quantization(quant_cfg)

    import torch
    from transformers import AutoModelForCausalLM

    if bnb is not None:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            quantization_config=bnb,
            torch_dtype=getattr(torch, cfg["model"].get("torch_dtype", "bfloat16")),
            attn_implementation=cfg["model"].get("attn_implementation", "flash_attention_2"),
            trust_remote_code=cfg["model"]["trust_remote_code"],
        )
    else:
        model = load_base_model(
            path,
            torch_dtype=cfg["model"].get("torch_dtype", "bfloat16"),
            attn_implementation=cfg["model"].get("attn_implementation", "flash_attention_2"),
            trust_remote_code=cfg["model"]["trust_remote_code"],
        )

    model.config.use_cache = False     # 训练时必须关，否则与梯度检查点冲突

    # ---- 3. LoRA ----
    model = setup_lora(model, cfg.get("lora", {}))

    # ---- 4. 训练 ----
    trainer = build_trainer(
        model=model,
        train_cfg=cfg["training"],
        dataset=dataset,
        tokenizer=tokenizer,
        output_dir=out_dir,
        loop_ctl=cfg.get("loop_control"),
    )
    trainer.train()

    # ---- 5. 保存 ----
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    log.info(f"SFT 完成，权重已保存: {out_dir}")
    log.info("⚠️ 下一步：核对 monitor/exit_history.json 的期望退出步数，确认未发生坍缩")


def main() -> int:
    p = argparse.ArgumentParser(description="LoopLite 办公场景 SFT")
    p.add_argument("--config", default="configs/train_sft_office.yaml")
    p.add_argument("--max-samples", type=int, default=None, help="调试用：只取前 N 条")
    p.add_argument("--dry-run", action="store_true", help="只检查配置与数据，不加载模型")
    a = p.parse_args()
    run(a.config, a.max_samples, a.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
