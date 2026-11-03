"""长文档检索增强（RAG）。

办公场景的核心不是「生成」，而是「从 200 页文档里找对那三段」。
循环架构的强项是长上下文理解，RAG 负责把长上下文收敛到有用范围。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ...utils.logger import get_logger

log = get_logger("looplite.office.rag")


@dataclass
class Chunk:
    text: str
    source: str
    index: int
    score: float = 0.0
    meta: dict = field(default_factory=dict)


class SimpleChunker:
    """按段落 + 长度约束切分。默认不依赖任何 NLP 库（超算离线友好）。"""

    def __init__(self, chunk_size: int = 1500, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str, source: str = "") -> list[Chunk]:
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks: list[Chunk] = []
        buf = ""
        idx = 0
        for p in paras:
            if len(buf) + len(p) <= self.chunk_size:
                buf = f"{buf}\n{p}".strip()
            else:
                if buf:
                    chunks.append(Chunk(buf, source, idx))
                    idx += 1
                # 保留尾部 overlap，避免切断语义
                buf = (buf[-self.overlap:] + "\n" + p).strip() if self.overlap else p
        if buf:
            chunks.append(Chunk(buf, source, idx))
        return chunks


class KeywordIndex:
    """极简 BM25 风格检索。

    为什么不用 embedding 模型：
      超算离线、无外网，额外拉一个 embedding 模型成本高；
      办公文档关键词区分度足够，BM25 往往比小 embedding 模型更准。
    若后续要换向量检索，实现同样的 .search() 接口即可。
    """

    def __init__(self):
        self.chunks: list[Chunk] = []
        self.df: dict[str, int] = {}
        self.avg_len: float = 0.0

    def _tokenize(self, text: str) -> list[str]:
        # 中英混合：英文按词，中文按 2-gram
        words = re.findall(r"[a-zA-Z0-9]+", text.lower())
        zh = re.findall(r"[\u4e00-\u9fff]+", text)
        grams = []
        for seg in zh:
            grams.extend(seg[i:i + 2] for i in range(len(seg) - 1))
            if len(seg) == 1:
                grams.append(seg)
        return words + grams

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.df = {}
        total = 0
        for c in chunks:
            toks = set(self._tokenize(c.text))
            total += len(toks)
            for t in toks:
                self.df[t] = self.df.get(t, 0) + 1
        self.avg_len = total / max(len(chunks), 1)
        log.info(f"索引构建完成：{len(chunks)} 个片段，词表 {len(self.df)}")

    def search(self, query: str, top_k: int = 5) -> list[Chunk]:
        import math

        q_tokens = self._tokenize(query)
        n = len(self.chunks)
        scored = []
        for c in self.chunks:
            toks = self._tokenize(c.text)
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            score = 0.0
            for t in set(q_tokens):
                if t not in tf:
                    continue
                idf = math.log(1 + (n - self.df.get(t, 0) + 0.5) / (self.df.get(t, 0) + 0.5))
                freq = tf[t]
                norm = freq / (freq + 1.5 + len(toks) / max(self.avg_len, 1))
                score += idf * norm
            scored.append((score, c))
        scored.sort(key=lambda x: -x[0])
        out = []
        for score, c in scored[:top_k]:
            c.score = round(score, 4)
            out.append(c)
        return out


class OfficeRAG:
    """对外接口：喂文档 → 检索 → 组装带引用的 prompt。"""

    def __init__(self, chunk_size: int = 1500, overlap: int = 200):
        self.chunker = SimpleChunker(chunk_size, overlap)
        self.index = KeywordIndex()

    def add_document(self, text: str, source: str = "") -> int:
        chunks = self.chunker.split(text, source)
        all_chunks = self.index.chunks + chunks
        self.index.build(all_chunks)
        return len(chunks)

    def add_file(self, path: str | Path) -> int:
        p = Path(path)
        suffix = p.suffix.lower()
        if suffix == ".txt" or suffix == ".md":
            text = p.read_text(encoding="utf-8", errors="ignore")
        elif suffix == ".pdf":
            import pdfplumber

            with pdfplumber.open(str(p)) as pdf:
                text = "\n\n".join(page.extract_text() or "" for page in pdf.pages)
        elif suffix == ".docx":
            import docx

            d = docx.Document(str(p))
            text = "\n\n".join(par.text for par in d.paragraphs)
        elif suffix in (".xlsx", ".xlsm"):
            import openpyxl

            wb = openpyxl.load_workbook(str(p), data_only=True)
            text = "\n\n".join(
                "\n".join(str(c.value) for c in row if c.value is not None)
                for ws in wb.worksheets
                for row in ws.iter_rows()
            )
        else:
            raise ValueError(f"暂不支持的文件类型: {suffix}")
        return self.add_document(text, p.name)

    def build_prompt(self, query: str, top_k: int = 5, system: str | None = None) -> str:
        hits = self.index.search(query, top_k)
        context = "\n\n".join(
            f"[{i + 1}] 来源: {c.source}#{c.index}\n{c.text}" for i, c in enumerate(hits)
        )
        sys_msg = system or "你是办公助手。只能依据下列材料回答，引用时用 [编号] 标注，材料中没有的就说不知道。"
        return (
            f"{sys_msg}\n\n===== 材料 =====\n{context}\n===== 材料结束 =====\n\n"
            f"问题：{query}\n请给出答案，并在句末标注引用编号。"
        )

    def stats(self) -> dict:
        return {"chunks": len(self.index.chunks), "vocab": len(self.index.df)}
