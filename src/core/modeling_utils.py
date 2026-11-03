"""权重加载与显存优化。

基座 SenseNova-Loop-9B 是自定义架构（sensenova_loop），
必须 trust_remote_code=True 加载 HF 仓库里的实现，本模块不重复造轮子。
"""

from __future__ import annotations

from pathlib import Path

from ..utils.config import resolve_path
from ..utils.logger import get_logger
from ..utils.quantize import estimate_weight_gb, protect_fp32_modules

log = get_logger("looplite.modeling")

# 原始 checkpoint（FP32，35.5GB）与转换后 BF16 版（17.8GB）的默认位置
RAW_WEIGHTS_DIR = "models/base_weights/SenseNova-Loop-9B-Preview"
BF16_WEIGHTS_DIR = "models/quantized/SenseNova-Loop-9B-BF16"


def default_model_dir() -> str:
    """自动挑一个可用的权重目录：优先 BF16 转换版，其次 FP32 原始件。

    为什么优先 BF16：同一张卡上 FP32 占 35.5GB，40G 卡几乎没余量给 KV cache；
    BF16 只要 17.8GB，能开出更长的上下文。
    """
    bf16 = resolve_path(BF16_WEIGHTS_DIR)
    if bf16.exists() and any(bf16.glob("*.safetensors")):
        return BF16_WEIGHTS_DIR
    return RAW_WEIGHTS_DIR


def resolve_model_path(local_path: str | None = None, repo_id: str | None = None) -> str:
    """优先用本地已下载的权重（超算无外网），否则回退 repo_id。"""
    if local_path:
        p = resolve_path(local_path)
        if p.exists() and any(p.glob("*.safetensors")):
            log.info(f"使用本地权重: {p}")
            return str(p)
        log.warning(f"本地权重不存在或不完整: {p}，回退在线 repo")
    else:
        auto = default_model_dir()
        p = resolve_path(auto)
        if p.exists() and any(p.glob("*.safetensors")):
            log.info(f"自动选用权重目录: {p}")
            return str(p)
    if repo_id:
        log.warning(f"回退到在线加载: {repo_id}（超算无外网时会失败）")
        return repo_id
    raise ValueError("既无可用本地权重，也未指定 repo_id")


def load_base_model(
    path_or_repo: str,
    torch_dtype: str = "bfloat16",
    attn_implementation: str = "flash_attention_2",
    device_map: str | dict | None = "auto",
    trust_remote_code: bool = True,
):
    """加载基座模型。已处理三类常见失败。"""
    import torch
    from transformers import AutoModelForCausalLM

    dtype = getattr(torch, torch_dtype)

    # flash-attn 在部分环境不可用（A100 需源码编译），失败自动降级
    try:
        model = AutoModelForCausalLM.from_pretrained(
            path_or_repo,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
    except (ImportError, ValueError) as e:
        log.warning(f"attn_implementation={attn_implementation} 不可用，降级 sdpa: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            path_or_repo,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )

    protect_fp32_modules(model)
    return model


def load_tokenizer(path_or_repo: str, trust_remote_code: bool = True):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path_or_repo, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def estimate_memory(
    stored_params_b: float = 9.0,
    dtype_bytes: int = 2,
    context_len: int = 32768,
    batch: int = 1,
    tensor_parallel: int = 1,
) -> dict:
    """粗估显存占用，用于提交作业前确认档位够不够。

    注意：循环架构的 KV cache 按【实际执行的层计算次数】算，
    中间 24 层跑 2 遍 = 48 层的 KV，不是 32 层。这是最容易估错的地方。
    """
    weight_gb = estimate_weight_gb(stored_params_b, dtype_bytes) / tensor_parallel

    # KV cache：4(KV头) × 128(head_dim) × 2(K/V) × 2(bytes) = 每层每 token 2KB
    kv_bytes_per_token = 4 * 128 * 2 * dtype_bytes
    effective_layers = 4 + 24 * 2 + 4  # 按官方结构；如需参数化从 config 读
    kv_gb = kv_bytes_per_token * context_len * batch * effective_layers / (1024**3) / tensor_parallel

    activation_gb = 4.0 / tensor_parallel  # 粗估

    return {
        "weight_gb": round(weight_gb, 2),
        "kv_cache_gb": round(kv_gb, 2),
        "activation_gb": round(activation_gb, 2),
        "total_gb": round(weight_gb + kv_gb + activation_gb, 2),
        "note": f"KV 按 {effective_layers} 次层计算估算（含循环 2 遍），非 32 层",
    }


def attach_lora(model, lora_path: str | None):
    """挂载 LoRA 权重（推理时按场景切换 office/agent）。"""
    if not lora_path:
        return model
    from peft import PeftModel

    p = resolve_path(lora_path)
    if not p.exists():
        log.warning(f"LoRA 路径不存在，跳过挂载: {p}")
        return model
    log.info(f"挂载 LoRA: {p}")
    return PeftModel.from_pretrained(model, str(p))


def check_local_weights(local_path: str | None = None) -> dict:
    """检查本地权重是否完整（提交作业前的必做自检）。

    要能同时认两种形态：
      - FP32 原始件：约 35.5GB，8 个分片
      - BF16 转换版：约 17.8GB，8 个分片
    阈值取 16GB（BF16 的下限），低于此值基本是下载/转换中断。
    """
    p = resolve_path(local_path or default_model_dir())
    result = {
        "path": str(p),
        "exists": p.exists(),
        "safetensors": [],
        "config_ok": False,
        "size_gb": 0.0,
        "dtype": None,
        "complete": False,
    }

    if not p.exists():
        return result

    st = sorted(p.glob("*.safetensors"))
    result["safetensors"] = [f.name for f in st]
    result["size_gb"] = round(sum(f.stat().st_size for f in st) / (1024**3), 2)
    result["config_ok"] = (p / "config.json").exists()

    if result["config_ok"]:
        try:
            import json

            cfg = json.loads((p / "config.json").read_text(encoding="utf-8"))
            result["dtype"] = cfg.get("dtype") or cfg.get("torch_dtype")
        except Exception:  # noqa: BLE001
            pass

    # 分片数与体积双校验：8 个分片 ≈ 35.5GB(FP32) / 17.8GB(BF16)
    if len(st) < 8:
        result["warning"] = f"只有 {len(st)}/8 个分片，权重不完整"
    elif result["size_gb"] < 16:
        result["warning"] = (
            f"权重仅 {result['size_gb']}GB（BF16 应 ~17.8GB，FP32 应 ~35.5GB），疑似不完整"
        )
    else:
        result["complete"] = True
    return result


def safetensors_index(local_path: str = "models/base_weights/SenseNova-Loop-9B-Preview") -> Path | None:
    p = resolve_path(local_path) / "model.safetensors.index.json"
    return p if p.exists() else None
