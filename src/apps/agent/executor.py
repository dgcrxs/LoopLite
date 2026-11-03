"""多步推理与工具调度循环。

======================================================================
循环架构在 Agent 场景的取舍
======================================================================
循环层的高阶推理能力是优势，但循环 = 单次 FLOPs 增加。
Agent 是「多轮调用」场景，延迟会被放大 N 倍。

对策：**限制 max_loop_pass**。
configs/inference_16g.yaml 里 agent 场景默认 loop_pass=1，
用一点点智力换响应速度。跑通后做 A/B：1 pass vs 2 pass 的
准确率差多少、延迟差多少，再定最终值。

======================================================================
为什么自己写 executor 而不直接用框架的 function calling
======================================================================
基座是自定义架构，vLLM/SGLang 常规版不支持，
OpenAI 兼容的 tool_calls 协议未必走得通。
自己解析 JSON 更可控，也方便统计「格式合格率」这个核心指标。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ...utils.logger import get_logger
from .tools import ToolRegistry

log = get_logger("looplite.agent.executor")

CALL_PATTERN = re.compile(r"\{\s*[\"']tool[\"'].*?\}", re.DOTALL)


@dataclass
class StepRecord:
    step: int
    raw_output: str
    parsed: dict | None = None
    parse_error: str | None = None
    tool_name: str | None = None
    result: dict | None = None
    error: str | None = None


@dataclass
class ExecutionTrace:
    task: str
    steps: list[StepRecord] = field(default_factory=list)
    final_answer: str = ""
    stopped_reason: str = ""

    @property
    def json_parse_rate(self) -> float:
        """格式合格率 —— Agent 场景唯一的硬指标。"""
        if not self.steps:
            return 0.0
        ok = sum(1 for s in self.steps if s.parsed is not None)
        return ok / len(self.steps)

    @property
    def tool_success_rate(self) -> float:
        called = [s for s in self.steps if s.tool_name]
        if not called:
            return 0.0
        return sum(1 for s in called if not s.error) / len(called)

    def summary(self) -> str:
        return (
            f"任务: {self.task}\n"
            f"步数: {len(self.steps)} | JSON 解析率: {self.json_parse_rate:.0%} | "
            f"工具成功率: {self.tool_success_rate:.0%}\n"
            f"终止原因: {self.stopped_reason}"
        )


def extract_json_call(raw: str) -> tuple[dict | None, str | None]:
    """从模型输出里抠出工具调用 JSON。

    三层容错：
      1) 去除 markdown 围栏
      2) 正则找第一个含 "tool" 键的 JSON 对象
      3) 直接尝试整体解析
    """
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()

    for m in CALL_PATTERN.finditer(text):
        frag = m.group(0)
        try:
            obj = json.loads(frag)
            if isinstance(obj, dict) and "tool" in obj:
                return obj, None
        except json.JSONDecodeError:
            continue

    # 整体尝试
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end + 1]), None
        except json.JSONDecodeError as e:
            return None, f"JSON 解析失败: {e}"
    return None, "输出中未找到 JSON 对象"


class AgentExecutor:
    """ReAct 风格执行器：思考 → 调工具 → 观察 → 再思考。"""

    def __init__(self, engine, registry: ToolRegistry | None = None, max_steps: int = 8, verbose: bool = True):
        self.engine = engine
        self.registry = registry or ToolRegistry()
        self.max_steps = max_steps
        self.verbose = verbose

    def run(self, task: str) -> ExecutionTrace:
        trace = ExecutionTrace(task=task)
        system = self.registry.build_system_prompt()
        history: list[str] = []

        for step in range(1, self.max_steps + 1):
            prompt = self._build_prompt(system, task, history)
            raw = self.engine.generate(
                self.engine.build_prompt(prompt, system=system),
                temperature=0.2,
                max_new_tokens=512,
            )
            rec = StepRecord(step=step, raw_output=raw)
            obj, err = extract_json_call(raw)

            if err:
                rec.parse_error = err
                history.append(f"第 {step} 步输出格式错误：{err}。请只输出 JSON。")
                trace.steps.append(rec)
                log.warning(f"[step {step}] JSON 解析失败: {err}")
                continue

            rec.parsed = obj
            tool_name = obj.get("tool")
            args = obj.get("args", {}) or {}
            rec.tool_name = tool_name

            if tool_name in ("final_answer", "answer", "finish"):
                trace.final_answer = args.get("answer") or obj.get("answer", "")
                trace.stopped_reason = "模型给出最终答案"
                trace.steps.append(rec)
                break

            tool = self.registry.get(tool_name)
            if tool is None:
                rec.error = f"未注册的工具: {tool_name}（可用: {self.registry.names}）"
                history.append(f"第 {step} 步调用失败：{rec.error}")
                trace.steps.append(rec)
                continue

            errs = tool.validate_args(args)
            if errs:
                rec.error = "参数校验失败: " + "; ".join(errs)
                history.append(f"第 {step} 步参数错误：{rec.error}。请按 schema 修正。")
                trace.steps.append(rec)
                continue

            try:
                rec.result = tool.func(**args)
                obs = json.dumps(rec.result, ensure_ascii=False)[:2000]
                history.append(f"第 {step} 步调用 {tool_name} 成功，结果：{obs}")
            except Exception as e:  # noqa: BLE001
                rec.error = f"{type(e).__name__}: {e}"
                history.append(f"第 {step} 步工具执行异常：{rec.error}")
            trace.steps.append(rec)

            if self.verbose:
                log.info(f"[step {step}] {tool_name} → {rec.error or 'ok'}")

        else:
            trace.stopped_reason = f"达到最大步数 {self.max_steps}"

        if not trace.final_answer:
            trace.final_answer = history[-1] if history else "(无有效输出)"
            trace.stopped_reason = trace.stopped_reason or "未产出最终答案"

        log.info(trace.summary())
        return trace

    def _build_prompt(self, system: str, task: str, history: list[str]) -> str:
        hist = "\n".join(history) if history else "（暂无，这是第一步）"
        return (
            f"{system}\n\n"
            f"任务：{task}\n\n"
            f"已完成步骤：\n{hist}\n\n"
            f"请输出下一步的 JSON 调用；若任务已完成，输出 "
            f'{{"tool": "final_answer", "args": {{"answer": "..."}}}}。'
        )
