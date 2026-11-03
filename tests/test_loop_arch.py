"""循环架构前向传播正确性测试。

需要 torch。本机 2G 显存无所谓——用小模型在 CPU 上跑，验证的是结构与语义。

    pytest tests/test_loop_arch.py -v
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="需要 torch（超算端或本机装 CPU 版）")

from src.core.loop_transformer import (  # noqa: E402
    LoopConfig,
    build_small_loop_model,
    sinkhorn_normalize,
)
from src.core.dense_to_loop import (  # noqa: E402
    SmoothExitLoss,
    average_layer_weights,
    plan_conversion,
)


def test_config_layer_count():
    """层计算次数必须等于 4 + 24*N + 4。"""
    cfg = LoopConfig(loop_pass=2)
    assert cfg.stored_layers == 32
    assert cfg.forward_computations == 56

    cfg3 = LoopConfig(loop_pass=3)
    assert cfg3.stored_layers == 32          # 存储层数不变
    assert cfg3.forward_computations == 80   # 计算次数增加


def test_config_validates_gqa():
    cfg = LoopConfig(num_attention_heads=16, num_key_value_heads=5)
    with pytest.raises(ValueError):
        cfg.validate()


def test_sinkhorn_is_doubly_stochastic():
    """Sinkhorn 后行和与列和都应接近 1 —— mHC 稳定性的根基。"""
    logits = torch.randn(2, 4, 2, 2)
    m = sinkhorn_normalize(logits, iters=20)
    assert torch.allclose(m.sum(dim=-1), torch.ones_like(m.sum(dim=-1)), atol=1e-2)
    assert torch.allclose(m.sum(dim=-2), torch.ones_like(m.sum(dim=-2)), atol=1e-2)
    assert (m >= 0).all()


def test_small_model_forward():
    """小模型能前向，且输出形状正确。"""
    model = build_small_loop_model()
    x = torch.randint(0, 1000, (2, 16))
    logits, stats = model(x)
    assert logits.shape == (2, 16, 1000)
    assert stats["passes_executed"] >= 1


def test_loop_pass_affects_output():
    """跑 1 遍和跑 2 遍的输出必须不同 —— 否则说明循环层根本没起作用。"""
    model = build_small_loop_model(enable_exit=False)
    model.eval()
    x = torch.randint(0, 1000, (1, 16))
    with torch.no_grad():
        out1, s1 = model(x, loop_pass=1)
        out2, s2 = model(x, loop_pass=2)
    assert s1["passes_executed"] == 1
    assert s2["passes_executed"] == 2
    assert not torch.allclose(out1, out2), "循环遍数不影响输出，说明循环块是死的"


def test_weight_sharing():
    """循环块内部权重共享：参数量不应随 loop_pass 增长。"""
    m1 = build_small_loop_model(loop_pass=1)
    m2 = build_small_loop_model(loop_pass=2)
    assert m1.count_parameters() == m2.count_parameters()


def test_gradient_flows_through_loop():
    """梯度必须能穿过循环块回传到循环层（BPTT 通路）。"""
    model = build_small_loop_model(enable_exit=False)
    x = torch.randint(0, 1000, (1, 8))
    logits, _ = model(x)
    loss = logits.float().pow(2).mean()
    loss.backward()

    has_grad = [n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert any("loop_block.layers" in n for n in has_grad), "循环层没有梯度，BPTT 断了"


def test_plan_conversion():
    plan = plan_conversion(32, num_pre=4, num_post=4, loop_pass=2)
    assert len(plan.pre_layers) == 4
    assert len(plan.post_layers) == 4
    assert len(plan.loop_source_layers) == 24
    assert plan.compression_ratio > 1.5      # 确实省了参数


def test_average_layer_weights():
    sd = {
        "model.layers.0.self_attn.q_proj.weight": torch.ones(4, 4),
        "model.layers.1.self_attn.q_proj.weight": torch.ones(4, 4) * 3,
    }
    avg = average_layer_weights(sd, [0, 1])
    assert torch.allclose(avg["self_attn.q_proj.weight"], torch.ones(4, 4) * 2)


def test_smooth_exit_loss_penalizes_early_exit():
    loss_fn = SmoothExitLoss(min_expected_step=1.5, weight=0.1)
    early = loss_fn(1.0)     # 第 1 遍就退出 → 应被惩罚
    late = loss_fn(2.0)      # 用满深度 → 不应被惩罚
    assert early.item() > 0
    assert late.item() == 0
