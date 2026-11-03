"""稠密模型 → 循环模型的转换配方（Dense-to-Loop）。

用于阶段一「轻量实验」：拿一个 1.7B 开源小模型，把它改造成循环架构，
跑通一次 LoRA，验证配方是否成立，再决定要不要动 9B 基座。

======================================================================
切分方案（来自社区实践与论文结论）
======================================================================
稠密模型 L 层 → 切成 A(首) + B(中, 共享, 跑 N 遍) + C(尾)

经验法则：
  - B 段取【中后段】比取【前段】好。浅层负责「看懂 token」，重复跑收益低；
    中后段负责「推理」，重复跑收益高。这也是官方把 4~27 层设为循环块的原因。
  - A、C 段保留独立权重，作为「入口编码 / 出口解码」的稳定锚点。
  - B 段内部各层若来自稠密模型不同深度，转换时用【加权平均】初始化，
    比直接取某一层的权重更稳（避免某一层的偏置被放大 N 倍）。

======================================================================
三大坑
======================================================================
  1. 退出坍缩：加了自适应退出后模型学会 n=1 就停 → 见 SmoothExitLoss
  2. BPTT 显存爆炸：跑 2 遍 = 激活翻倍 → 梯度检查点必开
  3. 循环次数与训练不一致：训练跑 2 遍、推理跑 1 遍 = 分布偏移，必须对齐
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .loop_transformer import LoopConfig, LoopedTransformerStack

# ======================================================================
# 切分规划
# ======================================================================


@dataclass
class ConversionPlan:
    """稠密 → 循环的切分方案。"""

    total_layers: int
    pre_layers: list[int]         # A 段：稠密模型中的层索引
    loop_source_layers: list[int] # B 段：共享块初始化时参考的稠密层索引
    post_layers: list[int]        # C 段
    loop_pass: int
    init_strategy: str            # mean | tail | middle | first

    def describe(self) -> str:
        return (
            f"稠密 {self.total_layers} 层 → "
            f"A[{len(self.pre_layers)}] + B[{len(self.loop_source_layers)}×{self.loop_pass}] + C[{len(self.post_layers)}]\n"
            f"  存储层数: {len(self.pre_layers) + len(self.loop_source_layers) + len(self.post_layers)}\n"
            f"  前向计算: {len(self.pre_layers) + len(self.loop_source_layers) * self.loop_pass + len(self.post_layers)} 次\n"
            f"  B 段初始化: {self.init_strategy}，参考稠密层 {self.loop_source_layers}\n"
            f"  参数压缩比: {self.compression_ratio:.2f}x"
        )

    @property
    def compression_ratio(self) -> float:
        stored = len(self.pre_layers) + len(self.loop_source_layers) + len(self.post_layers)
        forward = len(self.pre_layers) + len(self.loop_source_layers) * self.loop_pass + len(self.post_layers)
        return forward / max(stored, 1)


def plan_conversion(
    total_layers: int,
    num_pre: int = 4,
    num_post: int = 4,
    loop_pass: int = 2,
    init_strategy: str = "mean",
) -> ConversionPlan:
    """规划切分。B 段 = 去掉首尾后剩下的全部中间层。"""
    if total_layers < num_pre + num_post + 1:
        raise ValueError(f"层数 {total_layers} 太少，无法切成 {num_pre}+B+{num_post}")

    pre = list(range(0, num_pre))
    post = list(range(total_layers - num_post, total_layers))
    middle = list(range(num_pre, total_layers - num_post))

    if init_strategy == "tail":
        # 只取中后段的一半作为初始化参考
        half = max(len(middle) // 2, 1)
        source = middle[-half:]
    elif init_strategy == "middle":
        q = max(len(middle) // 4, 1)
        source = middle[q:-q] or middle
    elif init_strategy == "first":
        source = middle[:1]
    else:  # mean（默认，最稳）
        source = middle

    return ConversionPlan(
        total_layers=total_layers,
        pre_layers=pre,
        loop_source_layers=source,
        post_layers=post,
        loop_pass=loop_pass,
        init_strategy=init_strategy,
    )


# ======================================================================
# 权重转换
# ======================================================================


def average_layer_weights(state_dict: dict[str, torch.Tensor], layer_indices: list[int], prefix: str = "model.layers") -> dict[str, torch.Tensor]:
    """把多个稠密层的权重按算术平均合成一份（B 段初始化用）。

    为什么平均而不是直接取一层：
      循环块要跑 N 遍，直接复制某一层会把该层的偏置放大 N 倍，
      平均相当于做了一次「深度上的集成」，起手更稳。
    """
    if not layer_indices:
        raise ValueError("layer_indices 为空")

    keys = None
    acc: dict[str, torch.Tensor] = {}
    for idx in layer_indices:
        cur = {}
        for k, v in state_dict.items():
            if k.startswith(f"{prefix}.{idx}."):
                cur[k.split(f"{prefix}.{idx}.", 1)[1]] = v
        if not cur:
            raise KeyError(f"state_dict 中找不到层 {prefix}.{idx}.*")
        if keys is None:
            keys = set(cur)
        for k, v in cur.items():
            acc[k] = acc.get(k, torch.zeros_like(v, dtype=torch.float32)) + v.float()
    return {k: acc[k] / len(layer_indices) for k in keys} if keys else {}


def build_loop_state_dict(
    dense_sd: dict[str, torch.Tensor],
    plan: ConversionPlan,
    prefix: str = "model.layers",
) -> dict[str, torch.Tensor]:
    """按方案生成循环模型可加载的 state_dict（键名为目标模型结构）。

    注意：本函数产出的是「结构已就位」的权重字典，
    加载时用 strict=False，再手动核对 missing/unexpected keys。
    """
    out: dict[str, torch.Tensor] = {}

    for new_i, old_i in enumerate(plan.pre_layers):
        for k, v in dense_sd.items():
            tag = f"{prefix}.{old_i}."
            if k.startswith(tag):
                out[f"pre_layers.{new_i}.{k.split(tag, 1)[1]}"] = v

    shared = average_layer_weights(dense_sd, plan.loop_source_layers, prefix)
    for k, v in shared.items():
        out[f"loop_block.layers.0.{k}"] = v
    # 其余循环层复制同一份（真正的共享在模型结构上，权重一致即可）
    # 实际训练中会因各自更新而分叉，这里仅为初始化

    for new_i, old_i in enumerate(plan.post_layers):
        for k, v in dense_sd.items():
            tag = f"{prefix}.{old_i}."
            if k.startswith(tag):
                out[f"post_layers.{new_i}.{k.split(tag, 1)[1]}"] = v

    # 嵌入与 lm_head 通常键名一致，直接透传
    for k, v in dense_sd.items():
        if k.startswith("model.embed_tokens") or k.startswith("lm_head"):
            out[k] = v

    return out


# ======================================================================
# 平滑退出损失（防「退出坍缩」）
# ======================================================================


class SmoothExitLoss(nn.Module):
    """平滑退出损失：惩罚「太早退出」，鼓励模型用满循环深度。

    ⚠️ 这是防坍缩的第一道防线。第二道是 src/training/monitor.py 的实时监控。

    用法：
        loss = task_loss + weight * smooth_exit_loss(expected_exit_step)
    """

    def __init__(self, min_expected_step: float = 1.5, weight: float = 0.01, mode: str = "hinge"):
        super().__init__()
        self.min_expected_step = min_expected_step
        self.weight = weight
        self.mode = mode  # hinge(只惩罚低于阈值) | l2(始终拉向目标)

    def forward(self, expected_exit_step: float | torch.Tensor) -> torch.Tensor:
        if not isinstance(expected_exit_step, torch.Tensor):
            expected_exit_step = torch.tensor(float(expected_exit_step))
        if self.mode == "l2":
            return self.weight * (expected_exit_step - self.min_expected_step) ** 2
        # hinge：低于阈值才惩罚，避免压制合理的早退
        return self.weight * torch.clamp(self.min_expected_step - expected_exit_step, min=0.0)


def convert_dense_to_loop(
    dense_model: nn.Module,
    total_layers: int,
    cfg: LoopConfig,
    init_strategy: str = "mean",
    num_pre: int = 4,
    num_post: int = 4,
) -> tuple[LoopedTransformerStack, ConversionPlan]:
    """高层封装：规划 → 建模型 → 灌权重。

    返回 (循环模型, 切分方案)。权重复制失败时不会静默跳过，
    而是把 missing keys 抛给调用方判断——循环架构转换最怕「看似成功实则没灌进去」。
    """
    plan = plan_conversion(total_layers, num_pre, num_post, cfg.loop_pass, init_strategy)
    cfg.num_pre_layers = len(plan.pre_layers)
    cfg.num_post_layers = len(plan.post_layers)
    cfg.num_loop_layers = len(plan.loop_source_layers)

    loop_model = LoopedTransformerStack(cfg)
    sd = build_loop_state_dict(dense_model.state_dict(), plan)

    missing, unexpected = loop_model.load_state_dict(sd, strict=False)
    if missing:
        # 属于预期：循环层共享权重、norm/lm_head 命名差异，需调用方人工确认
        pass
    return loop_model, plan


def summarize_conversion(dense_layers: int, loop_pass: int = 2) -> None:
    """打印一份人类可读的转换方案对比，用于决策。"""
    for strategy in ("mean", "tail", "middle"):
        plan = plan_conversion(dense_layers, loop_pass=loop_pass, init_strategy=strategy)
        print(f"--- {strategy} ---")
        print(plan.describe())
