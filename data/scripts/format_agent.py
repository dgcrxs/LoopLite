#!/usr/bin/env python
"""把 API 调用日志转成 DPO 偏好数据。

用法：
    python data/scripts/format_agent.py --input data/raw/api_logs.jsonl \
                                        --output data/processed/agent_dpo.jsonl

======================================================================
偏好对怎么构造，直接决定 DPO 效果
======================================================================
chosen（好样本）必须满足：
  ✓ 是合法 JSON
  ✓ tool 名在工具注册表里存在
  ✓ 参数与 schema 完全对得上
  ✓ 这一步确实推进了任务

rejected（坏样本）优先取这三类（按收益排序）：
  1. 格式错误：输出了自然语言而不是 JSON
  2. 参数幻觉：编造了不存在的参数值或工具名
  3. 空转：没调工具，直接编了一个答案

⚠️ 不要拿「长度差异」当偏好信号 —— 模型会学会啰嗦。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# 注册在册的工具名（与 src/apps/agent/tools.py 保持一致）
KNOWN_TOOLS = {"read_excel", "summarize_document", "search_web", "final_answer"}

SYSTEM_HINT = (
    "你可以调用工具。调用时必须且只能输出一个 JSON 对象："
    '{"tool": "<工具名>", "args": {<参数>}}。不要输出解释文字。'
)


def is_valid_call(obj: dict) -> tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "不是 JSON 对象"
    if "tool" not in obj:
        return False, "缺少 tool 字段"
    if obj["tool"] not in KNOWN_TOOLS:
        return False, f"未知工具: {obj['tool']}"
    if "args" in obj and not isinstance(obj["args"], dict):
        return False, "args 不是对象"
    return True, ""


def build_prompt(record: dict) -> str:
    """把一条日志组装成 DPO 的 prompt 字段。"""
    task = record.get("task") or record.get("instruction") or ""
    return f"{SYSTEM_HINT}\n\n任务：{task}"


def convert(records: list[dict]) -> tuple[list[dict], dict]:
    """产出偏好对，并给出质量统计。"""
    pairs = []
    stats = {"total": len(records), "kept": 0, "skipped": 0, "bad_chosen": 0, "no_rejected": 0}

    for r in records:
        chosen_raw = r.get("chosen") or r.get("good") or r.get("correct")
        rejected_raw = r.get("rejected") or r.get("bad") or r.get("wrong")
        if not chosen_raw or not rejected_raw:
            stats["no_rejected"] += 1
            stats["skipped"] += 1
            continue

        # chosen 必须是合法且注册过的工具调用
        try:
            obj = json.loads(chosen_raw) if isinstance(chosen_raw, str) else chosen_raw
        except json.JSONDecodeError:
            stats["bad_chosen"] += 1
            stats["skipped"] += 1
            continue

        ok, why = is_valid_call(obj)
        if not ok:
            stats["bad_chosen"] += 1
            stats["skipped"] += 1
            continue

        # chosen 统一序列化（去掉多余空白，保证格式一致）
        chosen = json.dumps(obj, ensure_ascii=False)
        rejected = rejected_raw if isinstance(rejected_raw, str) else json.dumps(rejected_raw, ensure_ascii=False)

        pairs.append({
            "prompt": build_prompt(r),
            "chosen": chosen,
            "rejected": rejected,
            "meta": {"source": r.get("source", "unknown"), "reason": r.get("reject_reason", "")},
        })
        stats["kept"] += 1

    return pairs, stats


def main() -> int:
    ap = argparse.ArgumentParser(description="API 日志 → DPO 偏好数据")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="data/processed/agent_dpo.jsonl")
    ap.add_argument("--report", action="store_true", help="只输出质量报告，不写文件")
    a = ap.parse_args()

    src = Path(a.input)
    if not src.exists():
        print(f"❌ 输入不存在: {src}")
        return 1

    records = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    pairs, stats = convert(records)

    print("===== 转换报告 =====")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    keep_rate = stats["kept"] / max(stats["total"], 1)
    print(f"  保留率: {keep_rate:.0%}")
    if keep_rate < 0.5:
        print("  ⚠️ 保留率低于 50%：检查日志里 chosen 字段是否为合法 JSON、工具名是否已注册")
    if stats["bad_chosen"] > 0:
        print(f"  ⚠️ {stats['bad_chosen']} 条的 chosen 不合格，这些样本会污染偏好信号，已丢弃")

    if a.report:
        return 0

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"✓ 已写入 {len(pairs)} 条偏好对 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
