"""核心层：循环架构实现、稠密转循环配方、权重与显存优化。"""

from .loop_transformer import (
    LoopConfig,
    LoopedBlock,
    LoopedTransformerStack,
    MHCMixer,
    ExitRouter,
    DecoderLayer,
    FullAttention,
    LinearAttention,
    FFN,
    sinkhorn_normalize,
    build_small_loop_model,
)
from .dense_to_loop import (
    ConversionPlan,
    plan_conversion,
    build_loop_state_dict,
    average_layer_weights,
    convert_dense_to_loop,
    SmoothExitLoss,
)
from .modeling_utils import (
    load_base_model,
    load_tokenizer,
    resolve_model_path,
    estimate_memory,
    attach_lora,
    check_local_weights,
)

__all__ = [
    "LoopConfig",
    "LoopedBlock",
    "LoopedTransformerStack",
    "MHCMixer",
    "ExitRouter",
    "DecoderLayer",
    "FullAttention",
    "LinearAttention",
    "FFN",
    "sinkhorn_normalize",
    "build_small_loop_model",
    "ConversionPlan",
    "plan_conversion",
    "build_loop_state_dict",
    "average_layer_weights",
    "convert_dense_to_loop",
    "SmoothExitLoss",
    "load_base_model",
    "load_tokenizer",
    "resolve_model_path",
    "estimate_memory",
    "attach_lora",
    "check_local_weights",
]
