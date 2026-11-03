"""推理引擎 —— 阶段一的全部赌注都在这里。

目标只有一个：在超算上让模型输出第一个字。

设计：
  - 档位自适应：不指定 --profile 时按显存自动选（2G 本机 / 16G 消费卡 / 40G / 80G）
  - 场景切换：office / agent 走同一引擎，只换 LoRA 与采样参数（「一个底座，多个专家」）
  - 循环次数可调：loop_pass 是 Looped 架构的核心旋钮，运行时可改
  - 退出监控：推理侧也记录期望退出步数，用来发现「自训权重偷懒」
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from ..core.modeling_utils import (
    attach_lora,
    check_local_weights,
    estimate_memory,
    load_base_model,
    load_tokenizer,
    resolve_model_path,
)
from ..utils.config import load_config, resolve_path
from ..utils.hardware import detect_hardware, pick_profile, recommend_strategy
from ..utils.logger import get_logger

log = get_logger("looplite.engine")


@dataclass
class EngineConfig:
    model_path: str | None = None
    profile: str | None = None
    scene: str = "office"          # office | agent | raw
    loop_pass: int | None = None
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    lora: str | None = None
    device: str = "auto"
    extra: dict = field(default_factory=dict)


class LoopLiteEngine:
    """核心推理引擎。"""

    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.base_cfg = load_config("model_base")
        self.infer_cfg = load_config("inference_16g.yaml")

        self.hw = detect_hardware()
        self.profile_name = pick_profile(self.hw, self.infer_cfg.get("profiles", {}), cfg.profile)
        self.profile = self.infer_cfg["profiles"][self.profile_name]
        log.info(f"硬件: {self.hw.summary()}")
        log.info(f"选定档位: {self.profile_name}")

        strat = recommend_strategy(self.hw)
        log.info(f"策略建议: {strat['advice']}")
        if not strat.get("can_infer") and cfg.profile not in (None, "debug_2g"):
            raise RuntimeError(f"当前硬件跑不动推理: {strat['advice']}")

        self.scene_cfg = self.infer_cfg.get("scene_overrides", {}).get(cfg.scene, {})
        gen_cfg = self.infer_cfg["generation"]
        self.loop_pass = cfg.loop_pass or self.scene_cfg.get("loop_pass") or gen_cfg["loop_pass"]
        self.model = None
        self.tokenizer = None

    # ---------- 加载 ----------

    def load(self) -> None:
        if self.profile_name == "debug_2g" and self.cfg.device == "auto":
            log.warning("debug_2g 档位不加载真实权重，仅做结构自检")

        # 不指定 --model-path 时自动挑：优先 BF16 转换版（17.8GB），否则 FP32 原始件
        path = resolve_model_path(
            self.cfg.model_path,
            self.base_cfg["model"]["repo_id"],
        )

        report = check_local_weights(self.cfg.model_path)
        if report["exists"] and report.get("warning"):
            log.warning(report["warning"])
        if report["exists"] and report.get("complete"):
            log.info(f"权重就绪: {report['size_gb']}GB, dtype={report['dtype']}, "
                     f"{len(report['safetensors'])} 个分片")

        mem = estimate_memory(
            self.base_cfg["architecture"]["stored_params_b"],
            dtype_bytes=2,
            context_len=self.profile.get("max_context", 32768),
            tensor_parallel=self.profile.get("tensor_parallel", 1),
        )
        log.info(f"显存预估: {mem}")

        self.model = load_base_model(
            path,
            torch_dtype=self.profile.get("dtype", "bfloat16"),
            attn_implementation=self.infer_cfg["generation"].get("attn_implementation", "flash_attention_2"),
            trust_remote_code=self.base_cfg["model"]["trust_remote_code"],
        )

        lora = self.cfg.lora or self.scene_cfg.get("lora")
        self.model = attach_lora(self.model, lora)

        self.tokenizer = load_tokenizer(path, trust_remote_code=self.base_cfg["model"]["trust_remote_code"])

        self._apply_loop_pass(self.loop_pass)
        self.model.eval()

    def _apply_loop_pass(self, n: int) -> None:
        """动态设置循环遍数。

        官方自定义实现暴露的字段名可能不同，这里做尽力而为的适配：
        命中则改，没命中只警告——不要静默失败，否则训推不一致很难查。
        """
        applied = False
        for attr in ("loop_pass", "num_loop_pass", "n_loop"):
            if hasattr(self.model, "config") and hasattr(self.model.config, attr):
                setattr(self.model.config, attr, n)
                applied = True
                break
            if hasattr(self.model, attr):
                setattr(self.model, attr, n)
                applied = True
                break
        if applied:
            log.info(f"循环遍数设为 {n}（一次前向 = 4 + 24×{n} + 4 = {4 + 24 * n + 4} 次层计算）")
        else:
            log.warning(
                f"未能动态设置 loop_pass={n}：模型未暴露该字段。"
                f"请检查官方实现的配置名，或在加载前写入 config.json。"
            )

    # ---------- 生成 ----------

    def build_prompt(self, user_text: str, system: str | None = None) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user_text})
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return user_text

    def generate(self, prompt: str, stream: bool = False, **kwargs) -> str | Iterator[str]:
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load()")

        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.cfg.max_new_tokens),
            "temperature": kwargs.get("temperature", self.cfg.temperature),
            "top_p": kwargs.get("top_p", self.cfg.top_p),
            "do_sample": kwargs.get("temperature", self.cfg.temperature) > 0,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }

        # 推理侧退出监控（流式模式下在流结束后检查）
        self._check_exit()

        if not stream:
            with torch.no_grad():
                out = self.model.generate(**inputs, **gen_kwargs)
            return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        return self._stream_generate(inputs, gen_kwargs)

    def _stream_generate(self, inputs: dict, gen_kwargs: dict) -> Iterator[str]:
        """流式输出：用后台线程跑 generate，主线程 yield 增量文本。"""
        import threading

        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = {**gen_kwargs, "streamer": streamer}

        err: list[BaseException] = []

        def _run():
            try:
                self.model.generate(**inputs, **gen_kwargs)
            except BaseException as e:  # noqa: BLE001
                err.append(e)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        for chunk in streamer:
            yield chunk
        t.join()
        if err:
            raise err[0]

    def _check_exit(self) -> None:
        """推理侧防坍缩观测。自训权重跑长文本时尤其要看这个。"""
        mon = self.infer_cfg.get("exit_monitoring", {})
        if not mon.get("enabled"):
            return
        step = getattr(self.model, "last_expected_exit_step", None)
        if step is not None and step < mon.get("alert_if_expected_exit_step_below", 1.2):
            log.warning(
                f"[防坍缩] 期望退出步数仅 {step:.2f}，模型疑似跳过循环层偷懒。"
                f"检查训练配置 loop_control.min_expected_exit_step"
            )

    # ---------- 便捷入口 ----------

    def smoke_test(self) -> str:
        """冒烟测试：跑一个极短输入，确认能出字。阶段一的验收标准。"""
        if self.model is None:
            raise RuntimeError("模型未加载")
        prompt = self.build_prompt("用一句话说明什么是循环神经网络。")
        return self.generate(prompt, max_new_tokens=32, temperature=0.0)


def build_engine_from_cli(
    profile: str | None = None,
    scene: str = "office",
    loop_pass: int | None = None,
    model_path: str | None = None,
    lora: str | None = None,
    **kwargs,
) -> LoopLiteEngine:
    cfg = EngineConfig(
        profile=profile,
        scene=scene,
        loop_pass=loop_pass,
        model_path=model_path,
        lora=lora,
        extra=kwargs,
    )
    eng = LoopLiteEngine(cfg)
    eng.load()
    return eng


def dry_run_report() -> dict:
    """不加载模型，只输出环境与档位判断。本机准备阶段用这个。"""
    hw = detect_hardware()
    infer_cfg = load_config("inference_16g.yaml")
    profile_name = pick_profile(hw, infer_cfg.get("profiles", {}))
    strat = recommend_strategy(hw)
    weights = check_local_weights()
    mem = estimate_memory(tensor_parallel=infer_cfg["profiles"].get("a100_40g", {}).get("tensor_parallel", 2))
    return {
        "hardware": hw.summary(),
        "profile": profile_name,
        "strategy": strat,
        "local_weights": weights,
        "a100_40g_memory_estimate": mem,
    }
