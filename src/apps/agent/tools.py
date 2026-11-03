"""工具注册与 JSON Schema 定义。

Agent 场景的成败就两件事：
  1. 模型能不能输出**可解析**的 JSON
  2. 输出的**工具名和参数**对不对得上

所以这里的设计原则是：schema 尽量简单、字段名尽量直白、枚举尽量明确。
不要让模型去猜参数格式。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict            # JSON Schema
    func: Callable | None = None
    examples: list[str] = field(default_factory=list)

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def validate_args(self, args: dict) -> list[str]:
        """按 JSON Schema 做最小校验（required + type）。"""
        errs = []
        required = self.parameters.get("required", [])
        for r in required:
            if r not in args:
                errs.append(f"缺少必填参数: {r}")
        props = self.parameters.get("properties", {})
        for k, v in args.items():
            if k not in props:
                errs.append(f"未知参数: {k}（可用: {list(props)}）")
                continue
            want = props[k].get("type")
            got = {
                "string": str, "integer": int, "number": (int, float),
                "boolean": bool, "array": list, "object": dict,
            }.get(want)
            if got and not isinstance(v, got):
                errs.append(f"参数 {k} 类型错误: 期望 {want}，实际 {type(v).__name__}")
            if "enum" in props[k] and v not in props[k]["enum"]:
                errs.append(f"参数 {k} 取值非法: {v}，可选 {props[k]['enum']}")
        return errs


class ToolRegistry:
    """工具注册表。"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, name: str, description: str, parameters: dict, examples: list[str] | None = None):
        def deco(fn: Callable):
            self._tools[name] = Tool(name, description, parameters, fn, examples or [])
            return fn

        return deco

    def add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    def build_system_prompt(self) -> str:
        """生成工具描述提示词。

        刻意用「紧凑 JSON」而不是自然语言描述：
        模型对结构化工具定义更敏感，且便于后续做格式校验。
        """
        lines = [
            "你可以调用以下工具。调用时必须且只能输出一个 JSON 对象，格式：",
            '{"tool": "<工具名>", "args": {<参数>}}',
            "不要输出解释文字，不要输出 markdown 围栏。",
            "",
            "可用工具：",
        ]
        for t in self._tools.values():
            lines.append(f"- {t.name}: {t.description}")
            lines.append(f"  参数: {json.dumps(t.parameters, ensure_ascii=False)}")
            for ex in t.examples:
                lines.append(f"  示例: {ex}")
        return "\n".join(lines)


# ======================================================================
# 内置示例工具（可直接用于联调）
# ======================================================================


def build_default_registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.register(
        name="read_excel",
        description="读取 Excel 文件指定工作表，返回前 N 行",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"},
                "sheet": {"type": "string", "description": "工作表名，默认第一个"},
                "n_rows": {"type": "integer", "description": "返回行数，默认 20"},
            },
            "required": ["path"],
        },
        examples=['{"tool": "read_excel", "args": {"path": "data/raw/sales.xlsx", "n_rows": 10}}'],
    )
    def read_excel(path: str, sheet: str | None = None, n_rows: int = 20):
        import openpyxl

        wb = openpyxl.load_workbook(path, data_only=True)
        ws = wb[sheet] if sheet else wb.worksheets[0]
        rows = []
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i >= n_rows:
                break
            rows.append([str(c) if c is not None else "" for c in row])
        return {"rows": rows, "total_returned": len(rows)}

    @reg.register(
        name="summarize_document",
        description="对给定文本做摘要，指定最大字数",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "max_chars": {"type": "integer", "description": "摘要最大字数，默认 500"},
            },
            "required": ["text"],
        },
    )
    def summarize_document(text: str, max_chars: int = 500):
        return {"summary": text[:max_chars], "truncated": len(text) > max_chars}

    @reg.register(
        name="search_web",
        description="联网搜索（⚠️ 超算无外网时该工具不可用，调用会直接报错）",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )
    def search_web(query: str):
        raise RuntimeError("超算环境无外网，search_web 不可用")

    return reg
