"""Looped Transformer 核心实现（架构魔改区）。

======================================================================
边界说明（务必先读）
======================================================================
本文件是 LoopLite **自己维护**的循环架构实现，用途有三：
  1. 理解与复现 Looped Transformer 的运行机制（教学 / 消融实验）
  2. 阶段一「轻量实验」：把 1.7B 开源小模型做 dense→loop 转换并 LoRA 训练
  3. 阶段三 TTS：把循环块思想平移到声学模型

而**基座 SenseNova-Loop-9B 的推理并不走这里** —— 官方权重是自定义架构，
必须用 `trust_remote_code=True` 加载 HF 仓库里的实现（见 modeling_utils.py）。
两者共用的是同一套「循环语义」与「防坍缩监控」约定。

======================================================================
架构速览（与 configs/model_base.yaml 对齐）
======================================================================
    Input
      │
      ├─ Pre-loop  : 首 4 层，独立权重
      │
      ├─ Looped    : 中间 24 层【共享权重】，跑 N 遍（默认 2）
      │    └─ 每遍之间用 mHC 做双流信息交换
      │
      └─ Post-loop : 尾 4 层，独立权重 ──► LM Head

    存储 32 层权重（约 9B），一次前向执行 4 + 24*N + 4 = 56 次层计算（N=2）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# 配置
# ======================================================================


@dataclass
class LoopConfig:
    """循环架构的结构参数。"""

    hidden_size: int = 4096
    num_pre_layers: int = 4        # 首 A 层
    num_loop_layers: int = 24      # 中间 B 层（共享权重）
    num_post_layers: int = 4       # 尾 C 层
    loop_pass: int = 2             # 中间块循环遍数（推理时可调）
    num_attention_heads: int = 16
    num_key_value_heads: int = 4   # GQA
    intermediate_size: int = 12288
    vocab_size: int = 248320
    max_position_embeddings: int = 262144

    # mHC 配置
    num_streams: int = 2           # 双流
    sinkhorn_iters: int = 20       # Sinkhorn 迭代次数，对齐官方 mhc.iterations=20
    mhc_temperature: float = 1.0

    # 退出机制
    enable_exit: bool = False      # 自适应提前退出（训练 dense→loop 时开启）
    max_exit_pass: int = 4

    # 混合注意力：每 full_every 层里有 1 层全注意力，其余为线性注意力
    full_attention_every: int = 4

    extra: dict = field(default_factory=dict)

    @property
    def stored_layers(self) -> int:
        return self.num_pre_layers + self.num_loop_layers + self.num_post_layers

    @property
    def forward_computations(self) -> int:
        """一次前向执行的层计算次数。"""
        return self.num_pre_layers + self.num_loop_layers * self.loop_pass + self.num_post_layers

    def validate(self) -> None:
        assert self.loop_pass >= 1, "loop_pass 至少为 1"
        assert self.num_loop_layers > 0, "必须存在循环层"
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads({self.num_attention_heads}) 必须能被 "
                f"num_key_value_heads({self.num_key_value_heads}) 整除 (GQA)"
            )


# ======================================================================
# mHC：流形约束超连接（双流 + 双随机混合）
# ======================================================================


def sinkhorn_normalize(logits: torch.Tensor, iters: int = 20, eps: float = 1e-8) -> torch.Tensor:
    """Sinkhorn 迭代，把任意正矩阵压成双随机矩阵（行和=列和=1）。

    这是 mHC 的灵魂：不做约束的话，多轮循环后残差流会互相「串味」，
    信号要么爆炸要么消失；双随机约束把信息混合限制在一个良态的流形上。

    Args:
        logits: [*, S, S] 未归一化的混合权重（需为正）
    """
    x = torch.exp(logits - logits.max(dim=-1, keepdim=True).values) + eps
    for _ in range(iters):
        x = x / (x.sum(dim=-1, keepdim=True) + eps)   # 行归一
        x = x / (x.sum(dim=-2, keepdim=True) + eps)   # 列归一
    return x


class MHCMixer(nn.Module):
    """mHC 双流混合器。

    输入两条残差流 x = [x^(1), x^(2)]，用双随机矩阵做信息交换：

        x^(1)' = a·x^(1) + (1-a)·x^(2)
        x^(2)' = (1-a)·x^(1) + a·x^(2)

    其中 a 由输入动态生成并 Sinkhorn 归一（官方示意图里约 0.6/0.4）。
    """

    def __init__(self, hidden_size: int, num_streams: int = 2, sinkhorn_iters: int = 20):
        super().__init__()
        self.num_streams = num_streams
        self.sinkhorn_iters = sinkhorn_iters

        # H_pre: 由输入动态生成路由 logits
        self.h_pre = nn.Linear(hidden_size, num_streams, bias=False)
        # 可学习的静态混合矩阵（Sinkhorn 后趋近官方示意中的 0.6/0.4 形态）
        self.h_res = nn.Parameter(torch.zeros(num_streams, num_streams))
        # 混合强度门控：0 = 完全不混合，1 = 完全按双随机矩阵混合
        self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        """streams: [B, T, S, D] → [B, T, S, D]（混合后）"""
        b, t, s, d = streams.shape

        # 动态部分：双流平均后投影出每个位置的路由 logits [B, T, S]
        pooled = streams.mean(dim=2)
        dyn = self.h_pre(pooled).unsqueeze(-1)                       # [B, T, S, 1]

        # 静态 + 动态 → [B, T, S, S]
        logits = self.h_res.unsqueeze(0).unsqueeze(0) + dyn

        # Sinkhorn → 双随机（行和 = 列和 = 1）
        mix = sinkhorn_normalize(logits, iters=self.sinkhorn_iters)  # [B, T, S, S]

        mixed = torch.einsum("btsr,btrd->btsd", mix, streams)        # [B, T, S, D]
        # 残差式混合，避免破坏信号传播
        return streams + self.gate * (mixed - streams)


# ======================================================================
# 基础层：线性注意力 / 全注意力 / FFN
# ======================================================================


class LinearAttention(nn.Module):
    """线性注意力（O(T) 复杂度），用于混合注意力中的 3/4 层。

    用特征映射 φ(x)=elu(x)+1 把 softmax 注意力线性化。
    """

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        self.repeat = num_heads // num_kv_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(b, t, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim)

        def phi(z):
            return F.elu(z) + 1.0

        q, k = phi(q), phi(k)
        if self.repeat > 1:
            # GQA：每个 KV 头复制给 repeat 个 Q 头，使头数对齐
            k = k.repeat_interleave(self.repeat, dim=2)
            v = v.repeat_interleave(self.repeat, dim=2)

        # 因果线性注意力：对时间维做累和（而非全局求和）
        #   kv_t    = Σ_{s<=t} k_s ⊗ v_s        → [B, T, H, D, D]
        #   k_sum_t = Σ_{s<=t} k_s              → [B, T, H, D]
        kv = torch.cumsum(k.unsqueeze(-1) * v.unsqueeze(-2), dim=1)
        k_sum = torch.cumsum(k, dim=1)

        out = torch.einsum("bthd,bthde->bthe", q, kv)
        denom = torch.einsum("bthd,bthd->bth", q, k_sum).clamp(min=1e-6)
        out = out / denom.unsqueeze(-1)

        out = out.reshape(b, t, self.num_heads * self.head_dim)
        return self.o_proj(out)


class FullAttention(nn.Module):
    """标准全注意力（带 GQA），混合注意力中的 1/4 层。"""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, max_pos: int = 262144):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        self.repeat = num_heads // num_kv_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.repeat > 1:
            k = k.repeat_interleave(self.repeat, dim=1)
            v = v.repeat_interleave(self.repeat, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
        out = out.transpose(1, 2).reshape(b, t, -1)
        return self.o_proj(out)


class FFN(nn.Module):
    """SwiGLU 前馈网络。"""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    """单解码器层，按层索引决定用线性注意力还是全注意力。"""

    def __init__(self, cfg: LoopConfig, layer_idx: int, use_full_attention: bool):
        super().__init__()
        self.layer_idx = layer_idx
        if use_full_attention:
            self.attn = FullAttention(
                cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.max_position_embeddings
            )
        else:
            self.attn = LinearAttention(cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads)
        self.ffn = FFN(cfg.hidden_size, cfg.intermediate_size)
        self.norm1 = nn.RMSNorm(cfg.hidden_size)
        self.norm2 = nn.RMSNorm(cfg.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ======================================================================
# 退出机制（防「偷懒」的关键部件）
# ======================================================================


class ExitRouter(nn.Module):
    """自适应退出路由：第 n 遍循环后，判断是否还需继续。

    输出 halt 概率 p_n；累积 halting 达到 1 即停止。
    ⚠️ 这是「退出坍缩」风险的发生地：模型可能学会在 n=1 就 halt，
    等价于把循环架构退化成普通网络。必须配合 src/training/monitor.py 监控。
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] → 取序列池化，输出 [B] 的 halt 概率
        pooled = x.mean(dim=1)
        return torch.sigmoid(self.proj(pooled)).squeeze(-1)


# ======================================================================
# 循环块与整体堆叠
# ======================================================================


class LoopedBlock(nn.Module):
    """共享权重的循环块：同一组层被反复执行 loop_pass 遍。

    每一遍之间用 mHC 做双流信息交换，保证梯度不消失、信息不混淆。
    """

    def __init__(self, cfg: LoopConfig, start_layer_idx: int = 0):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    cfg,
                    layer_idx=start_layer_idx + i,
                    use_full_attention=((i + 1) % cfg.full_attention_every == 0),
                )
                for i in range(cfg.num_loop_layers)
            ]
        )
        self.mhc = MHCMixer(cfg.hidden_size, cfg.num_streams, cfg.sinkhorn_iters)
        self.exit_router = ExitRouter(cfg.hidden_size) if cfg.enable_exit else None

    def forward(
        self,
        streams: torch.Tensor,
        loop_pass: int | None = None,
        collect_exit: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """streams: [B, T, S, D] 双流输入。

        Returns:
            (输出 streams, 统计信息 dict)
        """
        n = loop_pass or self.cfg.loop_pass
        stats: dict = {"passes_executed": 0, "exit_probs": [], "expected_exit_step": float(n)}

        # 双流时只对主stream做层计算，副流通过 mHC 与之交换
        main = streams[:, :, 0, :]
        halted = torch.zeros(main.shape[0], device=main.device, dtype=torch.bool)
        cum_halt = torch.zeros(main.shape[0], device=main.device)

        for step in range(n):
            for layer in self.layers:
                main = layer(main)

            # 同步回双流并做 mHC 交换
            streams = streams.clone()
            streams[:, :, 0, :] = main
            streams = self.mhc(streams)
            main = streams[:, :, 0, :]

            stats["passes_executed"] = step + 1

            if self.exit_router is not None and step < n - 1:
                p = self.exit_router(main)
                stats["exit_probs"].append(p.detach())
                if not collect_exit:
                    cum_halt = cum_halt + p * (~halted).float()
                    halted = halted | (cum_halt >= 1.0)
                    if halted.all():
                        break

        if stats["exit_probs"]:
            probs = torch.stack(stats["exit_probs"], dim=0)          # [steps, B]
            steps = torch.arange(1, probs.shape[0] + 1, device=probs.device).float()
            stats["expected_exit_step"] = float((probs.mean(dim=1) * steps).sum() / probs.mean().clamp(min=1e-6))

        return streams, stats


class LoopedTransformerStack(nn.Module):
    """完整堆叠：Pre-loop → Looped(×N) → Post-loop。"""

    def __init__(self, cfg: LoopConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        h = cfg.hidden_size

        self.embed = nn.Embedding(cfg.vocab_size, h)
        self.pre_layers = nn.ModuleList(
            [DecoderLayer(cfg, i, use_full_attention=False) for i in range(cfg.num_pre_layers)]
        )
        self.loop_block = LoopedBlock(cfg, start_layer_idx=cfg.num_pre_layers)
        self.post_layers = nn.ModuleList(
            [
                DecoderLayer(cfg, cfg.num_pre_layers + cfg.num_loop_layers + i, use_full_attention=False)
                for i in range(cfg.num_post_layers)
            ]
        )
        # 双流初始化：主流用嵌入，副流为零
        self.stream_init = nn.Parameter(torch.zeros(cfg.num_streams, h))
        self.norm = nn.RMSNorm(h)
        self.lm_head = nn.Linear(h, cfg.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        loop_pass: int | None = None,
        collect_exit: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        b, t = input_ids.shape
        x = self.embed(input_ids)                                   # [B, T, D]

        for layer in self.pre_layers:
            x = layer(x)

        # 构造双流：[B, T, S, D]
        streams = self.stream_init.view(1, 1, self.cfg.num_streams, -1).expand(b, t, -1, -1).clone()
        streams[:, :, 0, :] = x

        streams, stats = self.loop_block(streams, loop_pass=loop_pass, collect_exit=collect_exit)
        x = streams[:, :, 0, :]

        for layer in self.post_layers:
            x = layer(x)

        logits = self.lm_head(self.norm(x))
        return logits, stats

    def count_parameters(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)


# ======================================================================
# 便捷构造
# ======================================================================


def build_small_loop_model(
    hidden_size: int = 256,
    num_pre_layers: int = 2,
    num_loop_layers: int = 4,
    num_post_layers: int = 2,
    vocab_size: int = 1000,
    loop_pass: int = 2,
    enable_exit: bool = True,
) -> LoopedTransformerStack:
    """构造一个可在 CPU 上秒级跑通的小模型（供 tests/ 与阶段一轻量实验使用）。"""
    cfg = LoopConfig(
        hidden_size=hidden_size,
        num_pre_layers=num_pre_layers,
        num_loop_layers=num_loop_layers,
        num_post_layers=num_post_layers,
        vocab_size=vocab_size,
        intermediate_size=hidden_size * 3,
        num_attention_heads=4,
        num_key_value_heads=2,
        loop_pass=loop_pass,
        enable_exit=enable_exit,
        max_position_embeddings=2048,
    )
    return LoopedTransformerStack(cfg)


def make_layer_factory(cfg: LoopConfig) -> Callable[[int, bool], DecoderLayer]:
    """返回层工厂，供 dense_to_loop 复用。"""
    return lambda idx, use_full: DecoderLayer(cfg, idx, use_full)
