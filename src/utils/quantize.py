"""权重量化与导出辅助。

⚠️ 重要前提（2026-11 现状）：
  基座 `sensenova_loop` 架构含线性注意力 + mHC 双流，
  llama.cpp / GGUF 生态目前【不识别】该架构，社区也没有现成 Q4_K_M / Q8_0。
  因此本模块当前只做两件稳妥的事：
    1. mHC 路由参数的 FP32 白名单保护（任何时候都必需）
    2. FP8 权重量化（A100/Ada/Hopper 可直接加载，无需生态适配）
  GGUF 相关函数保留为占位实现，等上游支持后再启用。
"""

from __future__ import annotations

from pathlib import Path

from .config import load_config
from .logger import get_logger

log = get_logger("looplite.quantize")

# 默认必须保留高精度的参数名片段（来自 configs/model_base.yaml）
DEFAULT_FP32_PATTERNS = ("mhc", "hyper_connection", "route", "mixer", "sinkhorn")


def get_fp32_patterns() -> tuple[str, ...]:
    try:
        cfg = load_config("model_base")
        pats = cfg.get("precision", {}).get("keep_fp32_patterns")
        if pats:
            return tuple(pats)
    except Exception as e:  # noqa: BLE001
        log.warning(f"读取 model_base 配置失败，回退默认白名单: {e}")
    return DEFAULT_FP32_PATTERNS


def should_keep_fp32(name: str, patterns: tuple[str, ...] | None = None) -> bool:
    """判断某参数是否必须保持 FP32。

    这是循环架构最容易踩的坑：把 mHC 的双随机混合矩阵压成低精度后，
    多轮循环间的信号传播会失稳，表现为「长文本后半段突然胡说」。
    """
    pats = patterns or get_fp32_patterns()
    lowered = name.lower()
    return any(p.lower() in lowered for p in pats)


def protect_fp32_modules(model, patterns: tuple[str, ...] | None = None) -> list[str]:
    """把命中白名单的参数原地转回 FP32 并返回名单。"""
    import torch

    protected = []
    for name, param in model.named_parameters():
        if should_keep_fp32(name, patterns) and param.dtype != torch.float32:
            param.data = param.data.float()
            protected.append(name)
    if protected:
        log.info(f"已保留 FP32 参数 {len(protected)} 个：" f"{protected[:5]}{' ...' if len(protected) > 5 else ''}")
    else:
        log.warning("未命中任何 FP32 白名单参数，请核对参数命名（架构若改名需同步更新白名单）")
    return protected


def quantize_fp8(model, patterns: tuple[str, ...] | None = None):
    """FP8 权重量化（E4M3），白名单参数跳过。

    A100 无 FP8 计算单元，但可用于【存储】降显存；Ada/Hopper 可进一步提速。
    """
    import torch

    try:
        fp8 = torch.float8_e4m3fn
    except AttributeError:
        raise RuntimeError("当前 torch 不支持 float8_e4m3fn，需 torch>=2.1")

    converted, skipped = 0, 0
    for name, param in model.named_parameters():
        if should_keep_fp32(name, patterns):
            skipped += 1
            continue
        if param.dtype in (torch.float32, torch.bfloat16, torch.float16):
            param.data = param.data.to(fp8)
            converted += 1
    log.info(f"FP8 量化完成：转换 {converted} 个张量，白名单跳过 {skipped} 个")
    return model


def estimate_weight_gb(num_params_b: float, dtype_bytes: int = 2) -> float:
    """粗估权重体积（GB）。bf16/fp16=2, fp8/int8=1, fp32=4。"""
    return round(num_params_b * 1e9 * dtype_bytes / (1024**3), 2)


# ======================================================================
# 流式分片转换（超算上跑的第一道工序）
# ======================================================================
# 为什么需要它：
#   官方 checkpoint 以 FP32 存储 9.4B 参数 → 落盘 35.5GB。
#   直接 torch.load 进内存要 35GB+，24G/40G 卡都会先被内存拖死。
#   这里逐分片流式转换，峰值内存 ≈ 单个分片（~4.7GB）。
#
# 为什么不用 accelerate/官方脚本：
#   自定义架构 + mHC 白名单，需要按参数名挑着转，通用脚本做不到。
# ======================================================================

_DTYPE_MAP = {
    "bfloat16": "BF16",
    "float16": "F16",
    "fp16": "F16",
    "float32": "F32",
    "fp32": "F32",
    "float8": "F8_E4M3",
    "fp8": "F8_E4M3",
    "fp8_e4m3": "F8_E4M3",
}

# 这些文件直接原样拷贝，不做张量处理
_COPY_SUFFIXES = {
    ".json", ".jinja", ".py", ".txt", ".model", ".tiktoken",
}
_COPY_NAMES = {
    "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
    "generation_config.json", "preprocessor_config.json",
    "video_preprocessor_config.json", "processor_config.json",
    "modeling_sensenova_loop.py",
}


def convert_shards(
    src_dir: str | Path,
    dst_dir: str | Path,
    target: str = "bfloat16",
    patterns: tuple[str, ...] | None = None,
    overwrite: bool = False,
) -> dict:
    """把 safetensors 权重逐分片转换精度，输出到新目录。

    保留原分片文件名，因此 index.json 的 weight_map 可以原样复用——
    这一点很关键：prefix 一致才能让 transformers 正常分片加载。

    Args:
        src_dir: 原始权重目录（含 model.safetensors.index.json）
        dst_dir: 输出目录
        target: bfloat16 / float16 / float8 / float32
        patterns: 必须保留 FP32 的参数名片段；None 则读 model_base.yaml
        overwrite: 目标已存在时是否覆盖

    Returns:
        dict: 统计信息（分片数、参数量、输入/输出字节数、被保护张量名）
    """
    import json
    import shutil

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    src = Path(src_dir)
    dst = Path(dst_dir)
    if not (src / "model.safetensors.index.json").exists():
        raise FileNotFoundError(f"{src} 不是分片权重目录（缺 index.json）")
    if dst.exists() and any(dst.glob("*.safetensors")) and not overwrite:
        raise FileExistsError(f"{dst} 已有权重文件，加 overwrite=True 才覆盖")
    dst.mkdir(parents=True, exist_ok=True)

    tgt_name = _DTYPE_MAP.get(target.lower())
    if tgt_name is None:
        raise ValueError(f"不支持的目标精度 {target}，可选: {sorted(set(_DTYPE_MAP))}")
    tgt_dtype = {
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        "F32": torch.float32,
        "F8_E4M3": torch.float8_e4m3fn,
    }[tgt_name]

    # FP8 只做存储压缩：A100 无 FP8 计算单元，反序列化后仍需 upcast 回 BF16 计算
    if tgt_name == "F8_E4M3" and not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("当前 torch 不支持 float8_e4m3fn，需 torch>=2.1")

    pats = patterns or get_fp32_patterns()
    index = json.loads((src / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = sorted(set(index["weight_map"].values()))

    log.info(f"开始转换 {len(shards)} 个分片 → {tgt_name}（FP32 白名单: {list(pats)}）")

    stats = {
        "target": tgt_name,
        "shards": 0,
        "tensors": 0,
        "bytes_in": 0,
        "bytes_out": 0,
        "protected": [],
    }

    for i, shard in enumerate(shards, 1):
        spath = src / shard
        if not spath.exists():
            raise FileNotFoundError(f"缺分片 {shard}")
        out: dict[str, "torch.Tensor"] = {}
        with safe_open(str(spath), framework="pt", device="cpu") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                if t.is_floating_point() and not should_keep_fp32(name, pats):
                    out[name] = t.to(tgt_dtype)
                else:
                    if t.is_floating_point() and should_keep_fp32(name, pats):
                        out[name] = t.to(torch.float32)
                        if name not in stats["protected"]:
                            stats["protected"].append(name)
                    else:
                        # 整型张量原样保留
                        out[name] = t
                stats["tensors"] += 1
                stats["bytes_in"] += t.numel() * t.element_size()
        save_file(out, str(dst / shard), metadata={"format": "pt"})
        stats["shards"] += 1
        stats["bytes_out"] += (dst / shard).stat().st_size
        log.info(
            f"  [{i}/{len(shards)}] {shard}: {len(out)} 张量, "
            f"{stats['bytes_out'] / 1024**3:.2f}GB 累计输出"
        )
        del out

    # mtp 头（推理不需要，但保留以免 config 引用报错）
    for extra in ("mtp.safetensors",):
        ep = src / extra
        if ep.exists():
            od = {}
            with safe_open(str(ep), framework="pt", device="cpu") as f:
                for name in f.keys():
                    t = f.get_tensor(name)
                    od[name] = t.to(tgt_dtype) if t.is_floating_point() else t
            save_file(od, str(dst / extra), metadata={"format": "pt"})
            log.info(f"  {extra}: {len(od)} 张量（已转换）")

    # index.json 原样复制（分片名未变，weight_map 仍有效）
    shutil.copy2(src / "model.safetensors.index.json", dst / "model.safetensors.index.json")

    # 其余配置文件原样拷贝
    for p in src.iterdir():
        if p.is_file() and (p.suffix in _COPY_SUFFIXES or p.name in _COPY_NAMES):
            if p.name == "model.safetensors.index.json":
                continue
            shutil.copy2(p, dst / p.name)

    # 改写 config.json 的 dtype 声明，避免加载时按 FP32 反推
    cfg_path = dst / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        alias = {"BF16": "bfloat16", "F16": "float16", "F32": "float32", "F8_E4M3": "float8_e4m3fn"}[tgt_name]
        cfg["dtype"] = alias
        cfg["torch_dtype"] = alias
        if isinstance(cfg.get("text_config"), dict):
            cfg["text_config"]["dtype"] = alias
        cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info(
        f"转换完成: {stats['bytes_in'] / 1024**3:.2f}GB → {stats['bytes_out'] / 1024**3:.2f}GB "
        f"(压缩比 {stats['bytes_in'] / max(stats['bytes_out'], 1):.2f}x), "
        f"保护 FP32 张量 {len(stats['protected'])} 个"
    )
    return stats


def _cli() -> int:
    import argparse

    p = argparse.ArgumentParser(description="LoopLite 权重精度转换（流式分片）")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("convert", help="FP32 分片权重转 BF16/FP16/FP8")
    c.add_argument("--src", required=True, help="原始权重目录")
    c.add_argument("--dst", required=True, help="输出目录")
    c.add_argument("--dtype", default="bfloat16", help="bfloat16 / float16 / float8 / float32")
    c.add_argument("--overwrite", action="store_true")
    c.add_argument("--dry-run", action="store_true", help="只打印计划，不写文件")

    a = p.parse_args()
    if a.cmd == "convert":
        if a.dry_run:
            import json

            idx = json.loads((Path(a.src) / "model.safetensors.index.json").read_text(encoding="utf-8"))
            n = len(set(idx["weight_map"].values()))
            bytes_map = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1}
            tgt = _DTYPE_MAP.get(a.dtype.lower())
            if tgt is None:
                log.error(f"不支持的目标精度 {a.dtype}")
                return 2
            est = estimate_weight_gb(9.656, bytes_map[tgt])
            log.info(f"[dry-run] {n} 个分片 → {tgt}，输出到 {a.dst}")
            log.info(f"[预估] 输出体积 ≈ {est} GB")
            return 0
        stats = convert_shards(a.src, a.dst, a.dtype, overwrite=a.overwrite)
        print({k: v for k, v in stats.items() if k != "protected"})
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli())


def export_gguf(src_dir: str | Path, out_path: str | Path, quant: str = "q4_k_m") -> None:
    """GGUF 导出占位实现。

    当前生态不支持 sensenova_loop 架构，此处明确报错而不是静默产出错文件。
    等 llama.cpp 上游合并架构支持后，在此填充真实转换逻辑。
    """
    raise NotImplementedError(
        "GGUF 导出暂不可用：llama.cpp 尚未支持 sensenova_loop 架构"
        "（含线性注意力与 mHC 双流）。\n"
        "替代方案：\n"
        "  1) A100 40G：BF16 + tensor_parallel=2（见 configs/inference_16g.yaml）\n"
        "  2) 显存紧张：quantize_fp8() 做 FP8 权重存储\n"
        "  3) 服务化：LightLLM(tp4) 或 SGLang 指定分支\n"
        "进度跟踪见 deployment/patched_llamacpp/README.md"
    )
