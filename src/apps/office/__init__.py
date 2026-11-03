"""办公助手应用：长文档 RAG + PPT 结构化生成。"""

from .rag_pipeline import OfficeRAG, SimpleChunker, KeywordIndex, Chunk
from .ppt_generator import (
    Outline,
    Slide,
    build_outline_prompt,
    parse_outline,
    render_pptx,
    generate_ppt,
)

__all__ = [
    "OfficeRAG",
    "SimpleChunker",
    "KeywordIndex",
    "Chunk",
    "Outline",
    "Slide",
    "build_outline_prompt",
    "parse_outline",
    "render_pptx",
    "generate_ppt",
]
