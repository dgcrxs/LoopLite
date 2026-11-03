"""PPT 大纲生成与渲染。

链路：LLM 输出结构化 JSON 大纲 → 本模块渲染成 .pptx。
为什么要 JSON 而不是直接让模型写 pptx 代码：JSON 可校验、可重跑、可 diff。

⚠️ 提醒（来自官方模型页的说明）：
模型页上那些漂亮的 PPT 效果图是「商汤小浣熊客户端」的产品效果，
= 模型 + 提示词工程 + 工具调用 + 渲染引擎。
裸接模型接口不会自动产出同款成品，外面的工作流必须自己搭。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ...utils.logger import get_logger

log = get_logger("looplite.office.ppt")

OUTLINE_SCHEMA = {
    "title": "string",
    "subtitle": "string(可选)",
    "slides": [
        {
            "title": "string",
            "bullets": ["string"],
            "notes": "string(可选)",
            "layout": "bullets|title|section|two_column(可选，默认 bullets)",
        }
    ],
}


@dataclass
class Slide:
    title: str
    bullets: list[str] = field(default_factory=list)
    notes: str = ""
    layout: str = "bullets"


@dataclass
class Outline:
    title: str
    subtitle: str = ""
    slides: list[Slide] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "Outline":
        return cls(
            title=d.get("title", "未命名演示"),
            subtitle=d.get("subtitle", ""),
            slides=[
                Slide(
                    title=s.get("title", ""),
                    bullets=s.get("bullets", []),
                    notes=s.get("notes", ""),
                    layout=s.get("layout", "bullets"),
                )
                for s in d.get("slides", [])
            ],
        )

    def validate(self) -> list[str]:
        errs = []
        if not self.title.strip():
            errs.append("缺少 title")
        if not self.slides:
            errs.append("slides 为空")
        for i, s in enumerate(self.slides):
            if not s.title.strip():
                errs.append(f"第 {i + 1} 页缺少标题")
            if len(s.bullets) > 8:
                errs.append(f"第 {i + 1} 页要点 {len(s.bullets)} 条，超过 8 条会挤爆版面")
        return errs


def build_outline_prompt(topic: str, n_slides: int = 10, audience: str = "业务决策层") -> str:
    """构造让模型输出合法 JSON 大纲的提示词。"""
    schema = json.dumps(OUTLINE_SCHEMA, ensure_ascii=False, indent=2)
    return (
        f"请为主题「{topic}」生成一份 {n_slides} 页的演示文稿大纲，面向{audience}。\n\n"
        f"严格要求：\n"
        f"1. 只输出 JSON，不要任何解释、不要 markdown 代码块围栏\n"
        f"2. JSON 结构必须严格符合以下 schema：\n{schema}\n"
        f"3. 每页要点控制在 3~6 条，每条不超过 30 字\n"
        f"4. layout 只能取 bullets / title / section / two_column\n"
    )


def parse_outline(raw: str) -> Outline:
    """从模型输出里抠出 JSON。做了三层容错，因为模型偶尔会加围栏或前缀。"""
    text = raw.strip()
    # 1) 去掉 markdown 围栏
    if text.startswith("```"):
        text = text.split("```", 2)[1] if "```" in text[3:] else text
        text = text.removeprefix("json").strip()
    # 2) 截取第一个 { 到最后一个 }
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"模型输出不是合法 JSON: {e}\n原始输出前 300 字：\n{raw[:300]}") from e
    return Outline.from_dict(data)


def render_pptx(outline: Outline, out_path: str | Path, template: str | None = None) -> Path:
    """渲染成 .pptx。使用 python-pptx 内置模板。"""
    from pptx import Presentation
    from pptx.util import Pt

    errs = outline.validate()
    if errs:
        log.warning(f"大纲存在问题但仍继续渲染: {errs}")

    prs = Presentation(template) if template else Presentation()

    # 封面
    cover = prs.slides.add_slide(prs.slide_layouts[0])
    cover.shapes.title.text = outline.title
    if outline.subtitle and len(cover.placeholders) > 1:
        cover.placeholders[1].text = outline.subtitle

    for s in outline.slides:
        if s.layout == "section":
            slide = prs.slides.add_slide(prs.slide_layouts[5])
            slide.shapes.title.text = s.title
        else:
            slide = prs.slides.add_slide(prs.slide_layouts[1])
            slide.shapes.title.text = s.title
            body = slide.placeholders[1]
            tf = body.text_frame
            tf.word_wrap = True
            for i, b in enumerate(s.bullets):
                p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                p.text = b
                p.level = 0
                p.font.size = Pt(18)
        if s.notes:
            slide.notes_slide.notes_text_frame.text = s.notes

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))
    log.info(f"PPT 已生成: {out}（共 {len(outline.slides) + 1} 页）")
    return out


def generate_ppt(engine, topic: str, out_path: str | Path, n_slides: int = 10) -> Path:
    """端到端：调用引擎生成大纲 → 渲染。engine 为 LoopLiteEngine 实例。"""
    prompt = build_outline_prompt(topic, n_slides)
    raw = engine.generate(engine.build_prompt(prompt), temperature=0.3)
    outline = parse_outline(raw)
    errs = outline.validate()
    if errs:
        log.warning(f"大纲校验问题: {errs}")
    return render_pptx(outline, out_path)
