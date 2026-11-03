"""轻量化 Web UI（Gradio）。

仅用于超算上临时验证效果，不作为生产交付：
  - 超算端口通常不对公网开放，需 ssh 端口转发：
      ssh -L 7860:localhost:7860 user@hpc
  - gradio 是可选依赖，未安装时给出明确提示而不是 ImportError 崩溃
"""

from __future__ import annotations

from ..utils.logger import get_logger
from .engine import EngineConfig, LoopLiteEngine

log = get_logger("looplite.webui")


def build_ui(engine: LoopLiteEngine):
    try:
        import gradio as gr
    except ImportError as e:
        raise RuntimeError(
            "未安装 gradio。超算离线安装：pip install gradio（需已打包进离线 wheel）"
        ) from e

    def respond(message: str, history, scene: str, loop_pass: int, temperature: float):
        engine.cfg.scene = scene
        engine.loop_pass = int(loop_pass)
        engine._apply_loop_pass(int(loop_pass))
        prompt = engine.build_prompt(message)
        out = engine.generate(prompt, temperature=float(temperature))
        return out

    with gr.Blocks(title="LoopLite 灵环") as demo:
        gr.Markdown("# 🌀 LoopLite 灵环\n循环生智，轻量无界")
        with gr.Row():
            scene = gr.Radio(["office", "agent", "raw"], value="office", label="场景")
            loop_pass = gr.Slider(1, 4, value=2, step=1, label="循环遍数（越大越聪明越慢）")
            temperature = gr.Slider(0.0, 1.5, value=0.7, label="温度")
        gr.ChatInterface(
            fn=respond,
            additional_inputs=[scene, loop_pass, temperature],
            title="对话",
        )
    return demo


def launch(profile: str | None = None, share: bool = False, server_port: int = 7860) -> None:
    cfg = EngineConfig(profile=profile)
    eng = LoopLiteEngine(cfg)
    eng.load()
    demo = build_ui(eng)
    # 超算环境禁止 share=True（会尝试连公网）
    demo.launch(server_name="127.0.0.1", server_port=server_port, share=False and share)


if __name__ == "__main__":
    launch()
