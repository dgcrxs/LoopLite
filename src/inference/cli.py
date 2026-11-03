"""命令行入口。

常用姿势：
    python -m src.inference.cli dry-run         # 不加载模型，看环境与档位判断（本机可用）
    python -m src.inference.cli smoke           # ★ 阶段一验收：输出第一个字
    python -m src.inference.cli run --scene office --prompt "..."
    python -m src.inference.cli chat --scene agent --loop-pass 1
"""

from __future__ import annotations

import argparse
import json
import sys

from .engine import EngineConfig, LoopLiteEngine, dry_run_report


def _build_engine(args) -> LoopLiteEngine:
    cfg = EngineConfig(
        model_path=args.model_path,
        profile=args.profile,
        scene=args.scene,
        loop_pass=args.loop_pass,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        lora=args.lora,
    )
    eng = LoopLiteEngine(cfg)
    eng.load()
    return eng


def cmd_dry_run(args) -> int:
    """本机也能跑：只探测环境，不碰权重。"""
    print(json.dumps(dry_run_report(), indent=2, ensure_ascii=False))
    return 0


def cmd_smoke(args) -> int:
    """阶段一验收命令：能出字就算赢。"""
    eng = _build_engine(args)
    print("\n----- 冒烟输出 -----")
    print(eng.smoke_test())
    print("--------------------")
    return 0


def cmd_run(args) -> int:
    eng = _build_engine(args)
    prompt = eng.build_prompt(args.prompt, system=args.system)
    if args.stream:
        for chunk in eng.generate(prompt, stream=True):
            print(chunk, end="", flush=True)
        print()
    else:
        print(eng.generate(prompt))
    return 0


def cmd_chat(args) -> int:
    eng = _build_engine(args)
    print(f"[LoopLite] 场景={args.scene} 循环遍数={eng.loop_pass} 档位={eng.profile_name}")
    print("输入 /exit 退出\n")
    while True:
        try:
            text = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text in ("/exit", "/quit"):
            break
        if not text:
            continue
        prompt = eng.build_prompt(text)
        for chunk in eng.generate(prompt, stream=True):
            print(chunk, end="", flush=True)
        print("\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="looplite", description="LoopLite 灵环推理引擎")
    sub = p.add_subparsers(dest="cmd", required=True)

    # dry-run
    sub.add_parser("dry-run", help="只探测硬件与档位，不加载模型").set_defaults(func=cmd_dry_run)

    # smoke / run / chat 共用参数
    def add_common(sp):
        sp.add_argument("--model-path", default=None, help="本地权重目录（默认 models/base_weights/...）")
        sp.add_argument("--profile", default=None, help="强制档位: debug_2g|consumer_16g|a100_40g|a100_80g")
        sp.add_argument("--scene", default="office", choices=["office", "agent", "raw"])
        sp.add_argument("--loop-pass", type=int, default=None, help="循环遍数，默认按场景取配置")
        sp.add_argument("--lora", default=None)
        sp.add_argument("--max-new-tokens", type=int, default=1024)
        sp.add_argument("--temperature", type=float, default=0.7)
        sp.add_argument("--top-p", type=float, default=0.9)

    s_smoke = sub.add_parser("smoke", help="冒烟测试：确认能输出第一个字")
    add_common(s_smoke)
    s_smoke.set_defaults(func=cmd_smoke)

    s_run = sub.add_parser("run", help="单次生成")
    add_common(s_run)
    s_run.add_argument("--prompt", required=True)
    s_run.add_argument("--system", default=None)
    s_run.add_argument("--stream", action="store_true")
    s_run.set_defaults(func=cmd_run)

    s_chat = sub.add_parser("chat", help="交互式对话")
    add_common(s_chat)
    s_chat.set_defaults(func=cmd_chat)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except RuntimeError as e:
        print(f"[错误] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
