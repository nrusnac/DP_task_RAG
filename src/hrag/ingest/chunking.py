"""Structure-aware chunking: section-aligned, self-contained chunks with a deterministic header.

Rules
- A section (heading + its paragraphs) becomes one chunk if <= max_tokens; otherwise it is split at paragraph
  boundaries with a small overlap. Oversized single paragraphs are split by sentences.
- Small tables are inlined in their section; large tables become their own chunks, split by row blocks with the
  header row repeated in every block (so numbers keep their column names).
- Slides and e-mails are atomic unless oversized.
- Consecutive tiny chunks of the same document are merged (tiny chunks are noisy and waste rerank budget).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from functools import lru_cache

from ..config import CFG
from .parsers import rows_to_markdown


@lru_cache(maxsize=1)
def _tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(CFG.embed_model)


def n_tokens(text: str) -> int:
    return len(_tok()(text, add_special_tokens=False)["input_ids"]) if text else 0


@dataclass
class Chunk:
    chunk_id: str
    task_id: str
    doc_id: str
    doc_type: str
    doc_title: str
    section: str
    kind: str
    text: str
    n_tokens: int
    order: int

    @property
    def header(self) -> str:
        sec = f" › {self.section}" if self.section else ""
        return f"[{self.doc_title}] ({self.doc_type}){sec}"

    def to_dict(self):
        d = asdict(self)
        d["header"] = self.header
        return d


_SENT = re.compile(r"(?<=[.;:!?])\s+(?=[A-Z(\[\"'])")


def _split_long(text: str, max_tok: int) -> list[str]:
    """Split one oversized paragraph by sentences, falling back to token windows."""
    sents = _SENT.split(text)
    out, buf, buf_t = [], [], 0
    for s in sents:
        t = n_tokens(s)
        if t > max_tok:  # pathological sentence: hard token windows
            ids = _tok()(s, add_special_tokens=False)["input_ids"]
            if buf:
                out.append(" ".join(buf)); buf, buf_t = [], 0
            out += [_tok().decode(ids[i:i + max_tok]) for i in range(0, len(ids), max_tok)]
            continue
        if buf_t + t > max_tok and buf:
            out.append(" ".join(buf)); buf, buf_t = [], 0
        buf.append(s); buf_t += t
    if buf:
        out.append(" ".join(buf))
    return out


def _pack(paras: list[str], max_tok: int, overlap_tok: int) -> list[str]:
    """Greedy paragraph packing with overlap (the last paragraph(s) carried over, up to overlap_tok)."""
    units = []
    for p in paras:
        units += _split_long(p, max_tok) if n_tokens(p) > max_tok else [p]
    chunks, cur, cur_t = [], [], 0
    for u in units:
        t = n_tokens(u)
        if cur and cur_t + t > max_tok:
            chunks.append("\n".join(cur))
            carry, carry_t = [], 0
            for prev in reversed(cur):
                pt = n_tokens(prev)
                if carry_t + pt > overlap_tok:
                    break
                carry.insert(0, prev); carry_t += pt
            cur, cur_t = carry, carry_t
        cur.append(u); cur_t += t
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def _table_blocks(rows: list[list[str]], max_tok: int) -> list[str]:
    if not rows:
        return []
    header, body = rows[0], rows[1:]
    if not body:
        return [rows_to_markdown([header])]
    head_t = n_tokens(rows_to_markdown([header]))
    blocks, cur, cur_t = [], [], head_t
    for r in body:
        t = n_tokens(" | ".join(r)) + 2
        if cur and cur_t + t > max_tok:
            blocks.append(rows_to_markdown(cur, header=header)); cur, cur_t = [], head_t
        cur.append(r); cur_t += t
    if cur:
        blocks.append(rows_to_markdown(cur, header=header))
    return blocks


def chunk_document(task_id: str, doc_id: str, doc_type: str, title: str, elements: list[dict],
                   max_tok: int | None = None, overlap_tok: int | None = None) -> list[Chunk]:
    max_tok = max_tok or CFG.max_chunk_tokens
    overlap_tok = overlap_tok if overlap_tok is not None else CFG.chunk_overlap_tokens
    raw: list[tuple[str, str, str]] = []  # (section_path, kind, text)
    stack: list[tuple[int, str]] = []
    buf: list[str] = []

    def path() -> str:
        return " › ".join(t for _, t in stack)

    def flush():
        nonlocal buf
        if buf:
            for piece in _pack(buf, max_tok, overlap_tok):
                raw.append((path(), "text", piece))
        buf = []

    for el in elements:
        k = el["kind"]
        if k == "heading":
            flush()
            while stack and stack[-1][0] >= el["level"]:
                stack.pop()
            stack.append((el["level"], el["text"][:100]))
        elif k == "para":
            buf.append(el["text"])
        elif k == "table":
            md = rows_to_markdown(el["rows"])
            if n_tokens(md) <= 150:
                buf.append(md)
            else:
                flush()
                for block in _table_blocks(el["rows"], max_tok):
                    raw.append((path(), "table", block))
        elif k in ("slide", "email"):
            flush()
            sec = el.get("title", "")
            for piece in _pack([p for p in el["text"].split("\n") if p.strip()], max_tok, overlap_tok):
                raw.append((sec, k, piece))
    flush()

    # merge consecutive tiny text chunks (e.g. a heading with one short line)
    merged: list[list] = []
    for sec, kind, text in raw:
        t = n_tokens(text)
        if merged and kind == "text" and merged[-1][1] == "text" and (merged[-1][3] < 120 or t < 60) \
                and merged[-1][3] + t <= max_tok:
            prev = merged[-1]
            if sec and sec != prev[0]:
                text = f"{sec.split(' › ')[-1]}\n{text}"
            prev[2] += "\n" + text
            prev[3] += t
        else:
            merged.append([sec, kind, text, t])

    return [Chunk(chunk_id=f"{task_id}::{doc_id}::{i}", task_id=task_id, doc_id=doc_id, doc_type=doc_type,
                  doc_title=title, section=sec, kind=kind, text=text, n_tokens=t, order=i)
            for i, (sec, kind, text, t) in enumerate(merged)]
