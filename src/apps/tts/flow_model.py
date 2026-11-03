"""基于 Flow-matching 的轻量 TTS 网络。

======================================================================
设计（阶段三）
======================================================================
**不复用 LLM 的循环层**，单独设计：
    9 个物理块 × 每块循环 2 次 = 等效 18 层深度

理由：语音任务是「局部细节 + 全局韵律」的混合，
直接搬 LLM 的 24 层循环块太重；9×2 在参数量与音质之间更平衡。
目标：比同音质的非循环模型少约 47% 参数。

======================================================================
Flow-matching 速览
======================================================================
训练：随机时刻 t，插值 x_t = (1-t)·noise + t·mel，网络预测速度场 v，
      损失 = MSE(v_pred, x_1 - x_0)
推理：从噪声出发按 ODE 积分 t: 0→1，步数越多越准也越慢
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.logger import get_logger

log = get_logger("looplite.tts.flow")


@dataclass
class TTSConfig:
    num_blocks: int = 9
    loop_pass: int = 2
    hidden_dim: int = 512
    num_heads: int = 8
    ffn_dim: int = 2048
    n_mels: int = 100
    speaker_dim: int = 192
    dropout: float = 0.1
    sigma_min: float = 1e-4


class AdaLNBlock(nn.Module):
    """带自适应层归一化的 DiT 风格块（条件注入用 adaLN）。"""

    def __init__(self, cfg: TTSConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden_dim)
        self.attn = nn.MultiheadAttention(cfg.hidden_dim, cfg.num_heads, batch_first=True, dropout=cfg.dropout)
        self.norm2 = nn.LayerNorm(cfg.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.ffn_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn_dim, cfg.hidden_dim),
        )
        # 条件（时刻 t + 说话人）→ 6 个调制参数
        self.adaLN = nn.Linear(cfg.hidden_dim, 6 * cfg.hidden_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D], cond: [B, D]
        m = self.adaLN(F.silu(cond)).unsqueeze(1)
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = m.chunk(6, dim=-1)

        h = self.norm1(x) * (1 + scale_a) + shift_a
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_a * h

        h = self.norm2(x) * (1 + scale_f) + shift_f
        x = x + gate_f * self.ffn(h)
        return x


class LoopFlowModel(nn.Module):
    """循环式 flow-matching 主干。

    结构上只有 num_blocks 个物理块，但每块在推理时跑 loop_pass 遍。
    """

    def __init__(self, cfg: TTSConfig):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([AdaLNBlock(cfg) for _ in range(cfg.num_blocks)])

        self.mel_in = nn.Linear(cfg.n_mels, cfg.hidden_dim)
        self.mel_out = nn.Linear(cfg.hidden_dim, cfg.n_mels)
        self.time_emb = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.spk_proj = nn.Linear(cfg.speaker_dim, cfg.hidden_dim)

        self._log_param_saving()

    def _log_param_saving(self) -> None:
        """量化「省了多少参数」——这是阶段三的核心交付指标。"""
        stored = sum(p.numel() for p in self.parameters())
        # 对照：不循环、直接堆等效深度的等同模型
        equivalent_layers = self.cfg.num_blocks * self.cfg.loop_pass
        per_block = sum(p.numel() for p in self.blocks[0].parameters())
        fixed = stored - per_block * self.cfg.num_blocks
        equivalent = per_block * equivalent_layers + fixed
        saving = 1 - stored / equivalent
        log.info(
            f"循环 TTS：{self.cfg.num_blocks} 块 × {self.cfg.loop_pass} 遍 "
            f"= 等效 {equivalent_layers} 层深度；参数 {stored / 1e6:.1f}M，"
            f"相比非循环等效模型省 {saving:.1%}"
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, spk: torch.Tensor, text_emb: torch.Tensor | None = None) -> torch.Tensor:
        """预测速度场。x: [B, T, n_mels], t: [B], spk: [B, speaker_dim]"""
        h = self.mel_in(x)
        if text_emb is not None:
            h = h + text_emb

        # 时刻编码（正弦 + MLP）
        half = self.cfg.hidden_dim // 2
        freqs = torch.exp(-torch.arange(half, device=x.device).float() * (2 * 3.14159 / half))
        t_emb = torch.cat([torch.sin(t[:, None] * freqs), torch.cos(t[:, None] * freqs)], dim=-1)
        cond = self.time_emb(t_emb) + self.spk_proj(spk)

        for block in self.blocks:
            for _pass in range(self.cfg.loop_pass):   # 循环遍数：参数共享，深度翻倍
                h = block(h, cond)

        return self.mel_out(h)


# ======================================================================
# Conditional Flow Matching
# ======================================================================


def cfm_loss(model: LoopFlowModel, mel: torch.Tensor, spk: torch.Tensor, text_emb: torch.Tensor | None = None) -> torch.Tensor:
    """条件流匹配损失。

    x_0 = 噪声，x_1 = 真实 mel，目标速度 = x_1 - x_0
    """
    b = mel.shape[0]
    device = mel.device
    t = torch.rand(b, device=device)
    noise = torch.randn_like(mel)

    t_exp = t[:, None, None]
    x_t = (1 - t_exp) * noise + t_exp * mel
    target = mel - noise

    pred = model(x_t, t, spk, text_emb)
    return F.mse_loss(pred, target)


@torch.no_grad()
def sample(
    model: LoopFlowModel,
    spk: torch.Tensor,
    shape: tuple[int, int],
    text_emb: torch.Tensor | None = None,
    steps: int = 16,
    solver: str = "euler",
    device: str = "cuda",
) -> torch.Tensor:
    """从噪声积分到 mel。

    ⚠️ 步数少于 8 时务必用 midpoint：欧拉法在少步数下误差累积明显，
    4-step 实时场景用 euler 会出现明显的金属音/模糊。
    """
    model.eval()
    x = torch.randn(shape, device=device)
    dt = 1.0 / steps

    for i in range(steps):
        t = torch.full((shape[0],), i * dt, device=device)
        v = model(x, t, spk, text_emb)
        if solver == "midpoint":
            x_mid = x + 0.5 * dt * v
            t_mid = t + 0.5 * dt
            v_mid = model(x_mid, t_mid, spk, text_emb)
            x = x + dt * v_mid
        else:  # euler
            x = x + dt * v

    return x


def build_tts_model(cfg_dict: dict | None = None) -> LoopFlowModel:
    """按 configs/tts_flowmatch.yaml 构造模型。"""
    if cfg_dict is None:
        from ...utils.config import load_config

        cfg_dict = load_config("tts_flowmatch")["model"]

    cfg = TTSConfig(
        num_blocks=cfg_dict.get("num_blocks", 9),
        loop_pass=cfg_dict.get("loop_pass_per_block", 2),
        hidden_dim=cfg_dict.get("hidden_dim", 512),
        num_heads=cfg_dict.get("num_heads", 8),
        ffn_dim=cfg_dict.get("ffn_dim", 2048),
        n_mels=cfg_dict.get("condition", {}).get("n_mels", 100),
        speaker_dim=cfg_dict.get("condition", {}).get("speaker_embedding_dim", 192),
        dropout=cfg_dict.get("dropout", 0.1),
    )
    return LoopFlowModel(cfg)
