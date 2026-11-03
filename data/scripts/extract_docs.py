#!/usr/bin/env python
"""从长文档提取文本，构造办公 SFT 数据。

用法：
    python data/scripts/extract_docs.py --input data/raw/docs \
                                        --output data/processed/office_sft.jsonl

产出的每条样本形如：
    {"instruction": "总结以下会议纪要", "input": "<长文本>", "output": "<摘要>"}

======================================================================
办公 SFT 数据的两个要点
======================================================================
1. input 要够长。办公场景的价值就在长上下文，
   短样本（<2000 字）让模型学不到「从长文里抓重点」的能力，
   反而会教会它「不看输入也能编」。
2. output 要结构化。摘要 / 要点 / 表格 / 大纲 —— 统一格式，
   模型才会稳定输出结构化结果，而不是每次风格都不一样。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SUPPORTED = {".txt", ".md", ".pdf", ".docx", ".xlsx", ".xlsm"}

# 输出结构化模板
TEMPLATES = {
    "summary": "请用不超过 {n} 字总结以下内容，输出三段式：核心结论 / 关键数据 / 待办事项。",
    "outline": "请为以下内容生成层级大纲，最多 {n} 个一级标题，每个标题下 2-4 个要点。",
    "action": "请从以下内容中提取所有待办事项，每条格式：- [负责人] 事项（截止时间）。没有就写「无」。",
    "table": "请把以下内容中的关键数据整理成 Markdown 表格，列名用中文。",
}


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        import pdfplumber

        with pdfplumber.open(str(path)) as pdf:
            return "\n\n".join(p.extract_text() or "" for p in pdf.pages)
    if suffix == ".docx":
        import docx

        d = docx.Document(str(path))
        return "\n\n".join(p.text for p in d.paragraphs if p.text.strip())
    if suffix in (".xlsx", ".xlsm"):
        import openpyxl

        wb = openpyxl.load_workbook(str(path), data_only=True)
        rows = []
        for ws in wb.worksheets:
            for row in ws.iter_rows(values_only=True):
                cells = [str(c) for c in row if c is not None]
                if cells:
                    rows.append(" | ".join(cells))
        return "\n".join(rows)
    raise ValueError(f"不支持的文件类型: {suffix}")


def build_samples(text: str, source: str, min_chars: int = 2000, kinds: tuple[str, ...] = ("summary", "outline")) -> list[dict]:
    """把一篇长文档转成若干条 SFT 样本。"""
    if len(text) < min_chars:
        return []

    samples = []
    for kind in kinds:
        instr = TEMPLATES[kind].format(n=300 if kind == "summary" else 8)
        samples.append({
            "instruction": instr,
            "input": text,
            "output": "",          # ★ 需人工或用强模型生成，见下方说明
            "meta": {"source": source, "kind": kind, "chars": len(text)},
        })
    return samples


def main() -> int:
    ap = argparse.ArgumentParser(description="长文档 → 办公 SFT 数据")
    ap.add_argument("--input", required=True, help="文件或目录")
    ap.add_argument("--output", default="data/processed/office_sft.jsonl")
    ap.add_argument("--min-chars", type=int, default=2000)
    ap.add_argument("--kinds", nargs="+", default=["summary", "outline"], choices=list(TEMPLATES))
    a = ap.parse_args()

    src = Path(a.input)
    files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.suffix.lower() in SUPPORTED)
    if not files:
        print(f"❌ 未找到支持的文件: {src}（支持 {sorted(SUPPORTED)}）")
        return 1

    all_samples = []
    skipped = []
    for f in files:
        try:
            text = extract_text(f)
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {f.name}: {e}")
            continue
        if len(text) < a.min_chars:
            skipped.append((f.name, len(text)))
            continue
        all_samples.extend(build_samples(text, f.name, a.min_chars, tuple(a.kinds)))
        print(f"  ✓ {f.name}: {len(text)} 字")

    if skipped:
        print(f"\n⚠️ {len(skipped)} 个文件短于 {a.min_chars} 字被跳过（办公场景需要长文本）")

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fp:
        for s in all_samples:
            fp.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n✓ 已生成 {len(all_samples)} 条样本 → {out}")
    print("=" * 50)
    print("⚠️ 重要：output 字段当前为空，需要填充。三种做法：")
    print("  1) 人工标注（质量最高，量小）")
    print("  2) 用强模型批量生成后人工抽检（推荐，性价比最高）")
    print("  3) 用超算上的基座自生成 + 规则过滤（Self-Instruct，需严格过滤）")
    print("  填充后务必跑一遍抽检：随机看 20 条，output 为空/跑题的必须删掉")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
