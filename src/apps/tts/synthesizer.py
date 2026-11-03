"""语音合成接口：对接 LLM 输出，完成「文字 → 语音」闭环。

设计要点：
  - **流式**：LLM 边出字边合成，不等整句结束。chunk 太小会破坏语调，
    太大则延迟高；默认 24 字（configs 里可调）
  - **采样步数按场景切换**：实时对话 4 步、日常 16 步、有声书 64 步
  - **声码器用预训练**：不自训，省一大笔算力
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...utils.config import load_config
from ...utils.logger import get_logger
from .flow_model import LoopFlowModel, sample

log = get_logger("looplite.tts.synth")


@dataclass
class SynthConfig:
    steps: int = 16
    solver: str = "euler"
    sample_rate: int = 24000
    chunk_chars: int = 24


class Synthesizer:
    """TTS 合成器。"""

    def __init__(self, model: LoopFlowModel, spk_emb: torch.Tensor | None = None, cfg: SynthConfig | None = None):
        self.model = model
        self.cfg = cfg or SynthConfig()
        self.device = next(model.parameters()).device
        if spk_emb is None:
            d = model.cfg.speaker_dim
            spk_emb = torch.zeros(1, d, device=self.device)
            log.warning("未提供音色 embedding，使用零向量（音色会是默认中性声）")
        self.spk = spk_emb.to(self.device)

    @classmethod
    def from_config(cls, weights_path: str | None = None, scene: str = "balanced") -> "Synthesizer":
        cfg = load_config("tts_flowmatch")
        steps = cfg["flow_matching"]["inference_steps"].get(scene, 16)
        solver = cfg["flow_matching"].get("solver", "euler")
        # 4-step 场景强制 midpoint：欧拉在极少步数下音质崩
        if steps <= 8:
            solver = "midpoint"
            log.info("少步数场景自动切换 solver=midpoint")

        from .flow_model import build_tts_model

        model = build_tts_model(cfg["model"])
        if weights_path:
            import os

            if os.path.exists(weights_path):
                model.load_state_dict(torch.load(weights_path, map_location="cpu"))
            else:
                log.warning(f"权重不存在，使用随机初始化: {weights_path}")

        return cls(model, cfg=SynthConfig(steps=steps, solver=solver,
                                          sample_rate=cfg["model"]["condition"]["sample_rate"]))

    def mel_to_wave(self, mel: torch.Tensor) -> torch.Tensor:
        """mel → 波形。优先 Vocos，未装则退回 Griffin-Lim。

        注意：Griffin-Lim 音质明显差，只用于管线联调，不能用于效果评估。
        """
        try:
            import vocos

            vocoder = vocos.Vocos.from_pretrained("charactr/vocos-mel-24khz")
            return vocoder.decode(mel.transpose(1, 2))
        except Exception as e:  # noqa: BLE001
            log.warning(f"Vocos 不可用（{e}），回退 Griffin-Lim（仅联调用，音质差）")
            import torchaudio

            return torchaudio.functional.griffinlim(
                mel.transpose(1, 2).exp(), n_fft=1024, hop_length=256, n_iter=32
            )

    def synth_chunk(self, text: str, text_emb: torch.Tensor | None = None) -> torch.Tensor:
        """合成一段文本。返回波形 [1, T]。"""
        # 时长粗估：中文约 5 字/秒
        duration_frames = max(int(len(text) / 5 * self.cfg.sample_rate / 256), 10)
        mel = sample(
            self.model,
            self.spk,
            shape=(1, duration_frames, self.model.cfg.n_mels),
            text_emb=text_emb,
            steps=self.cfg.steps,
            solver=self.cfg.solver,
            device=self.device,
        )
        return self.mel_to_wave(mel)

    def stream_speak(self, text_iter, text_emb: torch.Tensor | None = None):
        """流式合成：LLM 边出字，这里攒够 chunk 就吐一段音频。"""
        buf = ""
        for piece in text_iter:
            buf += piece
            while len(buf) >= self.cfg.chunk_chars:
                # 在标点处断句，避免把句子切碎
                cut = self._find_break(buf, self.cfg.chunk_chars)
                chunk, buf = buf[:cut], buf[cut:]
                yield self.synth_chunk(chunk, text_emb)
        if buf.strip():
            yield self.synth_chunk(buf, text_emb)

    @staticmethod
    def _find_break(text: str, min_len: int) -> int:
        for i in range(min_len, len(text)):
            if text[i] in "。！？；，、\n":
                return i + 1
        return min(len(text), min_len)

    def save_wav(self, wave: torch.Tensor, path: str) -> None:
        import soundfile as sf

        sf.write(path, wave.squeeze().detach().cpu().numpy(), self.cfg.sample_rate)
        log.info(f"音频已保存: {path}")
