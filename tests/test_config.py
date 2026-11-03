"""配置与路径自检。不需要 torch，本机可跑。

    pytest tests/test_config.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.config import ConfigError, deep_merge, load_config, resolve_path
from src.utils.hardware import HardwareInfo, pick_profile, recommend_strategy


def test_load_model_base():
    cfg = load_config("model_base")
    arch = cfg["architecture"]
    assert arch["total_stored_layers"] == 32
    assert arch["loop_block_layers"] == 24
    assert arch["default_loop_pass"] == 2
    # 实测值：safetensors 逐张量累加 871 个张量
    assert arch["stored_params_b"] == 9.656
    assert arch["effective_params_b"] == 14.6
    assert cfg["model"]["trust_remote_code"] is True


def test_checkpoint_is_fp32_and_big():
    """checkpoint 以 FP32 存储 → 35.5GB。低于这个数字的假设都会低估磁盘。"""
    cfg = load_config("model_base")
    prec = cfg["precision"]
    assert prec["checkpoint_dtype"] == "float32"
    assert prec["checkpoint_size_gb"] > 34
    # BF16 体积约为 FP32 的一半
    sizes = prec["size_gb_by_dtype"]
    assert sizes["bfloat16"] < sizes["fp32"] * 0.6


def test_base_is_multimodal_not_text_only():
    cfg = load_config("model_base")
    assert "image" in cfg["architecture"]["modality"]
    assert cfg["architecture"]["vision_config"]["depth"] == 27


def test_mhc_sinkhorn_aligned_with_official():
    """Sinkhorn 迭代次数必须与官方 config 的 mhc.iterations 一致（20）。"""
    import inspect

    from src.core import loop_transformer as lt

    cfg = load_config("model_base")
    assert cfg["architecture"]["mhc"]["sinkhorn_iterations"] == 20
    assert cfg["architecture"]["mhc"]["streams"] == 2
    assert inspect.signature(lt.sinkhorn_normalize).parameters["iters"].default == 20


def test_fp32_whitelist_exists():
    """mHC 路由参数白名单不能为空，否则量化会毁掉循环稳定性。"""
    cfg = load_config("model_base")
    pats = cfg["precision"]["keep_fp32_patterns"]
    assert len(pats) >= 3
    assert any("mhc" in p.lower() for p in pats)


def test_inference_profiles():
    cfg = load_config("inference_16g.yaml")
    profiles = cfg["profiles"]
    for name in ("debug_2g", "consumer_16g", "a100_40g", "a100_80g"):
        assert name in profiles


def test_scene_overrides():
    cfg = load_config("inference_16g.yaml")
    assert cfg["scene_overrides"]["agent"]["loop_pass"] == 1     # Agent 控延迟
    assert cfg["scene_overrides"]["office"]["loop_pass"] == 2


def test_training_configs_have_gradient_checkpointing():
    """循环架构 BPTT 必须开梯度检查点，两个训练配置都要有。"""
    for name in ("train_sft_office", "train_dpo_agent"):
        cfg = load_config(name)
        assert cfg["training"]["gradient_checkpointing"] is True, f"{name} 未开梯度检查点"


def test_tts_loop_layout():
    cfg = load_config("tts_flowmatch")
    assert cfg["model"]["num_blocks"] == 9
    assert cfg["model"]["loop_pass_per_block"] == 2
    assert cfg["flow_matching"]["inference_steps"]["realtime"] == 4


def test_hpc_config_offline():
    cfg = load_config("hpc")
    assert cfg["cluster"]["offline"] is True
    assert cfg["resource"]["gpu_mem_gb"] == 40
    assert cfg["sync"]["method"] == "scp"


def test_deep_merge():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    over = {"a": {"c": 9}, "e": 5}
    out = deep_merge(base, over)
    assert out == {"a": {"b": 1, "c": 9}, "d": 3, "e": 5}
    assert base["a"]["c"] == 2      # 原对象未被修改


def test_resolve_path():
    p = resolve_path("configs/model_base.yaml")
    assert p.is_absolute() and p.exists()


def test_load_missing_config():
    with pytest.raises(ConfigError):
        load_config("definitely_not_exists.yaml")


def test_pick_profile_by_memory():
    profiles = {"debug_2g": {}, "consumer_16g": {}, "a100_40g": {}, "a100_80g": {}}
    assert pick_profile(HardwareInfo(cuda_available=False), profiles) == "debug_2g"
    assert pick_profile(HardwareInfo(cuda_available=True, per_device_mem_gb=[40.0]), profiles) == "a100_40g"
    assert pick_profile(HardwareInfo(cuda_available=True, per_device_mem_gb=[80.0]), profiles) == "a100_80g"
    assert pick_profile(HardwareInfo(cuda_available=True, per_device_mem_gb=[16.0]), profiles) == "consumer_16g"
    # 显式指定优先
    assert pick_profile(HardwareInfo(), profiles, explicit="a100_40g") == "a100_40g"


def test_recommend_strategy_for_local_machine():
    """本机 2G 显存：必须明确告知不能推理，而不是静默尝试。"""
    info = HardwareInfo(cuda_available=True, per_device_mem_gb=[2.0])
    s = recommend_strategy(info)
    assert s["can_infer"] is False
