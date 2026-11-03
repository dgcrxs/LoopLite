"""基础训练封装：QLoRA / 梯度检查点 / BPTT 控制 / 防坍缩回调。

======================================================================
循环架构训练的显存账（BPTT 展开 = 激活翻倍）
======================================================================
循环块跑 N 遍，反向传播要沿「时间」展开，激活值约为普通网络的 N 倍。
三板斧，按优先级：
  1. gradient_checkpointing（必开）—— 用重计算换显存，默认已强制
  2. QLoRA（4bit 基座）—— 权重从 20GB 压到 ~6GB
  3. per_device_batch_size=1 + 梯度累积 —— 长序列场景唯一选择

另外：训练时的 loop_pass 必须与推理时一致，否则训推分布偏移。
"""

from __future__ import annotations

from ..utils.config import load_config, resolve_path
from ..utils.logger import get_logger
from .monitor import ExitCollapseMonitor

log = get_logger("looplite.trainer")


def setup_lora(model, lora_cfg: dict):
    """按配置挂 LoRA。

    ⚠️ 循环架构注意点：中间 24 层是共享权重的一层，
    target_modules 打到它，就自动作用于全部循环遍数，不要重复指定。
    mHC 路由参数默认排除（保 FP32 稳定性）。
    """
    from peft import LoraConfig, get_peft_model

    exclude = tuple(p.lower() for p in lora_cfg.get("exclude_patterns", []))
    wanted = list(lora_cfg.get("target_modules", []))

    # 必须遍历 named_modules 而非 named_parameters：
    # 参数名形如 ...self_attn.q_proj.weight，取最后一段会得到 "weight"
    found = set()
    for name, _mod in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in wanted and not any(e in name.lower() for e in exclude):
            found.add(leaf)
    target_names = sorted(found)

    if not target_names:
        raise RuntimeError(
            "没有匹配到任何 LoRA 目标模块。请核对 target_modules 与基座参数命名，"
            "自定义架构（sensenova_loop）的命名可能与标准 Llama 不同。"
        )
    log.info(f"LoRA 目标模块: {target_names}")

    cfg = LoraConfig(
        r=lora_cfg.get("r", 64),
        lora_alpha=lora_cfg.get("alpha", 128),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type="CAUSAL_LM",
        target_modules=target_names,
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


def setup_quantization(quant_cfg: dict):
    """返回 BitsAndBytesConfig（mode=qlora 时）。"""
    from transformers import BitsAndBytesConfig

    if not quant_cfg.get("load_in_4bit", False):
        return None
    import torch

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
    )


def build_training_args(train_cfg: dict, output_dir: str, hpc_cfg: dict | None = None):
    """构造 TrainingArguments。循环架构强制开启梯度检查点。"""
    from transformers import TrainingArguments

    gc = train_cfg.get("gradient_checkpointing", True)
    if not gc:
        log.warning("配置未开启 gradient_checkpointing：循环架构 BPTT 极易 OOM，已强制开启")
        gc = True

    kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 16),
        learning_rate=train_cfg.get("learning_rate", 1e-4),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        bf16=train_cfg.get("bf16", True),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 200),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        report_to=train_cfg.get("report_to", ["tensorboard"]) or [],
        gradient_checkpointing=gc,
        gradient_checkpointing_kwargs=train_cfg.get("gradient_checkpointing_kwargs", {"use_reentrant": False}),
        optim=train_cfg.get("optim", "adamw_torch_fused"),
        group_by_length=train_cfg.get("group_by_length", False),
        seed=train_cfg.get("seed", 42),
    )
    # DPO 等场景可能额外传参
    kwargs.update(train_cfg.get("extra_args", {}))
    return TrainingArguments(**kwargs)


def build_monitor(loop_ctl: dict, output_dir: str) -> ExitCollapseMonitor:
    """按 loop_control 配置构造防坍缩监控。"""
    return ExitCollapseMonitor(
        alert_below=loop_ctl.get("min_expected_exit_step", 1.2),
        output_dir=f"{output_dir}/monitor",
    )


def build_trainer(model, train_cfg: dict, dataset, tokenizer, output_dir: str, data_collator=None, loop_ctl: dict | None = None):
    """一站式构造 Trainer（含防坍缩回调）。"""
    from transformers import Trainer

    args = build_training_args(train_cfg, output_dir)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    if loop_ctl:
        trainer.add_callback(build_monitor(loop_ctl, output_dir).as_callback())
        log.info("防坍缩监控已挂载（指标: expected_exit_step）")

    return trainer


def load_dataset_from_jsonl(path: str, tokenizer, max_seq_len: int = 16384, fmt: str = "alpaca", num_proc: int = 16):
    """加载并 tokenize JSONL 数据集。

    fmt=alpaca: {"instruction","input","output"}
    """
    from datasets import load_dataset

    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"训练数据不存在: {p}（超算无外网，必须先由本机推送）")

    ds = load_dataset("json", data_files=str(p), split="train")
    log.info(f"载入 {len(ds)} 条样本: {p}")

    def tokenize(example):
        if fmt == "alpaca":
            inp = example.get("input", "") or ""
            prompt = f"{example['instruction']}\n{inp}".strip() if inp else example["instruction"]
            full = prompt + "\n" + example["output"]
        else:
            raise ValueError(f"暂不支持的格式: {fmt}")

        enc = tokenizer(full, truncation=True, max_length=max_seq_len)
        labels = list(enc["input_ids"])
        # prompt 部分不参与 loss：只让模型学「输出」，不学「复述问题」
        prompt_len = len(tokenizer(prompt, truncation=True, max_length=max_seq_len)["input_ids"])
        labels[:prompt_len] = [-100] * prompt_len
        enc["labels"] = labels
        return enc

    return ds.map(tokenize, num_proc=num_proc, remove_columns=ds.column_names, desc="tokenize")


def load_config_with_defaults(name: str) -> dict:
    return load_config(name)
