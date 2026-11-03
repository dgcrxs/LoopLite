"""OpenAI 兼容的 HTTP 服务。

启动：
    python deployment/api_server.py --profile a100_40g --scene office

接口：
    POST /v1/chat/completions   （兼容 OpenAI 格式，便于接入已有系统）
    GET  /health
    GET  /v1/models

⚠️ 现阶段定位：内部联调用。
官方推荐的生产部署是 LightLLM(tp4) 或 SGLang 指定分支 —— 那些有真正的
批处理、连续批处理与前缀缓存优化。本服务是单请求串行的，吞吐低。
"""

from __future__ import annotations

import argparse
import time
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="LoopLite API", version="0.1.0")

_ENGINE = None
_START = time.time()


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "looplite"
    messages: list[Message]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 1024
    stream: bool = False
    scene: str = Field(default="office", description="office | agent，决定 LoRA 与采样策略")


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[dict]
    usage: dict


@app.get("/health")
def health():
    return {
        "status": "ok" if _ENGINE is not None else "engine-not-loaded",
        "uptime_s": int(time.time() - _START),
        "scene": getattr(_ENGINE, "cfg", None) and _ENGINE.cfg.scene,
        "loop_pass": getattr(_ENGINE, "loop_pass", None),
    }


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": "looplite", "object": "model", "owned_by": "looplite"}]}


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    if _ENGINE is None:
        raise HTTPException(status_code=503, detail="引擎未加载")

    _ENGINE.cfg.scene = req.scene
    _ENGINE.cfg.temperature = req.temperature
    _ENGINE.cfg.max_new_tokens = req.max_tokens

    system = "\n".join(m.content for m in req.messages if m.role == "system") or None
    user = "\n".join(m.content for m in req.messages if m.role != "system")
    prompt = _ENGINE.build_prompt(user, system=system)

    text = _ENGINE.generate(prompt)

    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=req.model,
        choices=[{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        usage={"prompt_tokens": len(prompt), "completion_tokens": len(text), "total_tokens": len(prompt) + len(text)},
    )


def main():
    global _ENGINE

    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=None)
    ap.add_argument("--scene", default="office")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()

    from src.inference.engine import EngineConfig, LoopLiteEngine

    _ENGINE = LoopLiteEngine(EngineConfig(profile=a.profile, scene=a.scene, model_path=a.model_path))
    _ENGINE.load()

    import uvicorn

    # 只监听本机：超算节点不对公网开放，外部访问走 ssh 端口转发
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
