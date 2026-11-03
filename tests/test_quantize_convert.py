"""量化/转换模块测试：用微型假权重验证流式分片转换逻辑。

这套逻辑是超算上的第一道工序（35.5GB FP32 → 17.8GB BF16），
在本地用 3 个小分片跑通，才能在超算上放心跑大文件。
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors.torch import save_file  # noqa: E402

from src.utils.quantize import (  # noqa: E402
    convert_shards,
    estimate_weight_gb,
    should_keep_fp32,
)


def _make_fake_ckpt(root):
    """造一个 3 分片的小 checkpoint，含 mHC 参数与整型张量。"""
    root.mkdir(parents=True, exist_ok=True)
    shards = {
        "model-00001-of-00003.safetensors": {
            "model.language_model.embed_tokens.weight": torch.randn(64, 32, dtype=torch.float32),
            "model.language_model.layers.0.mlp.down_proj.weight": torch.randn(32, 48, dtype=torch.float32),
        },
        "model-00002-of-00003.safetensors": {
            # mHC 混合矩阵：必须保持 FP32
            "model.language_model.layers.4.mhc.h_res": torch.randn(2, 2, dtype=torch.float32),
            "model.language_model.layers.4.mlp.down_proj.weight": torch.randn(32, 48, dtype=torch.float32),
            # 整型张量：不该被转
            "model.language_model.layers.4.rotary.inv_freq": torch.arange(8, dtype=torch.int64),
        },
        "model-00003-of-00003.safetensors": {
            "lm_head.weight": torch.randn(64, 32, dtype=torch.float32),
        },
    }
    weight_map = {}
    for name, tensors in shards.items():
        save_file(tensors, str(root / name), metadata={"format": "pt"})
        for k in tensors:
            weight_map[k] = name

    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}), encoding="utf-8"
    )
    (root / "config.json").write_text(
        json.dumps(
            {
                "dtype": "float32",
                "torch_dtype": "float32",
                "text_config": {"dtype": "float32"},
                "model_type": "sensenova_loop",
            }
        ),
        encoding="utf-8",
    )
    (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (root / "modeling_sensenova_loop.py").write_text("# stub\n", encoding="utf-8")
    (root / "mtp.safetensors").__str__()
    save_file({"mtp.fc.weight": torch.randn(16, 16)}, str(root / "mtp.safetensors"))
    return root


def test_fp32_whitelist_matches_mhc():
    assert should_keep_fp32("model.language_model.layers.4.mhc.h_res")
    assert should_keep_fp32("layers.4.mixer.weight")
    assert not should_keep_fp32("model.language_model.layers.0.mlp.down_proj.weight")


def test_convert_to_bfloat16(tmp_path):
    src = _make_fake_ckpt(tmp_path / "src")
    dst = tmp_path / "dst"

    stats = convert_shards(src, dst, target="bfloat16")

    assert stats["target"] == "BF16"
    assert stats["shards"] == 3
    assert stats["bytes_out"] < stats["bytes_in"], "转换后体积必须变小"

    # index 与配置文件都到位
    assert (dst / "model.safetensors.index.json").exists()
    assert (dst / "modeling_sensenova_loop.py").exists()
    assert (dst / "mtp.safetensors").exists()

    # config dtype 被改写
    cfg = json.loads((dst / "config.json").read_text(encoding="utf-8"))
    assert cfg["dtype"] == "bfloat16"
    assert cfg["text_config"]["dtype"] == "bfloat16"

    # 普通浮点权重 → BF16
    from safetensors import safe_open

    with safe_open(str(dst / "model-00001-of-00003.safetensors"), framework="pt") as f:
        t = f.get_tensor("model.language_model.embed_tokens.weight")
        assert t.dtype == torch.bfloat16

    # mHC 参数 → 仍是 FP32
    with safe_open(str(dst / "model-00002-of-00003.safetensors"), framework="pt") as f:
        h = f.get_tensor("model.language_model.layers.4.mhc.h_res")
        assert h.dtype == torch.float32, "mHC 混合矩阵必须保持 FP32"
        # 整型张量原样
        inv = f.get_tensor("model.language_model.layers.4.rotary.inv_freq")
        assert inv.dtype == torch.int64


def test_convert_back_to_float32_is_lossless_for_values(tmp_path):
    src = _make_fake_ckpt(tmp_path / "src2")
    dst = tmp_path / "dst2"
    convert_shards(src, dst, target="float32")
    from safetensors import safe_open

    with safe_open(str(src / "model-00003-of-00003.safetensors"), framework="pt") as f:
        a = f.get_tensor("lm_head.weight")
    with safe_open(str(dst / "model-00003-of-00003.safetensors"), framework="pt") as f:
        b = f.get_tensor("lm_head.weight")
    assert torch.equal(a, b)


def test_refuse_overwrite(tmp_path):
    src = _make_fake_ckpt(tmp_path / "src3")
    dst = tmp_path / "dst3"
    convert_shards(src, dst, target="bfloat16")
    with pytest.raises(FileExistsError):
        convert_shards(src, dst, target="bfloat16")


def test_estimate_weight_gb():
    assert estimate_weight_gb(9.656, 2) == pytest.approx(17.99, abs=0.1)
    assert estimate_weight_gb(9.656, 4) == pytest.approx(35.98, abs=0.1)
