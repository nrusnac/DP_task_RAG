"""A deliberately basic RAG baseline ("textbook" defaults), to show what the advanced stack and the agent add.

What it does NOT have, compared with `naive` / `single_shot` / the agent:
- no structure-aware chunking: each document is flattened to plain text and cut into fixed 1,000-character windows
  with 200 characters of overlap (LangChain's classic splitter defaults); no headings, no table header rows, no header line
- no BM25, no hybrid fusion, no reranking: dense similarity only, top 4 chunks (LangChain's default k)
- no grounded prompt: the classic "use the following pieces of context" QA prompt, no numbered evidence or citations
What it keeps, so the comparison stays about the RAG techniques: the same documents and text extraction, the same
embedding model (BGE-M3), the same generator (Sonnet), and the search filtered to the matter's own data room.

    python -m hrag.agent.basic_rag build      # parse + fixed-size chunks + embed -> artifacts/basic_chunks.parquet, data/index/dense_basic.npy
"""
from __future__ import annotations

import argparse
import json
import time
from functools import lru_cache

import numpy as np
import pandas as pd

from .. import llm
from ..config import CFG

CHUNK_CHARS, OVERLAP_CHARS, TOP_K = 1000, 200, 4

BASIC_PROMPT = """Use the following pieces of context to answer the question at the end. If you don't know the answer, \
just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""


def plain_text(elements: list[dict]) -> str:
    """Document elements -> one plain string, the way a generic text extractor would see it (tables as cell text)."""
    out = []
    for e in elements:
        if e["kind"] == "table":
            out += ["  ".join(r) for r in e["rows"]]
        else:
            out.append(" ".join(x for x in (e.get("title"), e.get("text")) if x))
    return "\n".join(out)


def fixed_chunks(text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """Fixed-size character windows; the last window may be shorter. No awareness of sentences, sections or tables."""
    step = size - overlap
    return [text[i:i + size] for i in range(0, max(len(text) - overlap, 1), step) if text[i:i + size].strip()]


def build(cfg=CFG) -> dict:
    from ..index import models
    from ..ingest.build import acquire, load_tasks
    from ..ingest.parsers import parse
    t0 = time.perf_counter()
    area = acquire(cfg)
    rows = []
    for t in load_tasks(area):
        for path in sorted((area / t["task_id"] / "documents").glob("*")):
            try:
                _, elements = parse(path)
            except Exception:  # noqa: BLE001  -- same rule as the main ingest: one broken file can't stop the build
                continue
            for i, text in enumerate(fixed_chunks(plain_text(elements))):
                rows.append(dict(chunk_id=f"basic::{t['task_id']}::{path.name}::{i}", task_id=t["task_id"],
                                 doc_id=path.name, text=text))
    df = pd.DataFrame(rows)
    df.to_parquet(cfg.artifacts_dir / "basic_chunks.parquet", index=False)
    E = models.embed(df.text.tolist(), show_progress=True)
    cfg.index_dir.mkdir(parents=True, exist_ok=True)
    np.save(cfg.index_dir / "dense_basic.npy", E.astype("float16"))
    return dict(n_chunks=len(df), n_docs=int(df.groupby(["task_id", "doc_id"]).ngroups),
                chars_median=int(df.text.str.len().median()), seconds=round(time.perf_counter() - t0, 1))


@lru_cache(maxsize=1)
def _index() -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_parquet(CFG.artifacts_dir / "basic_chunks.parquet")
    return df, np.load(CFG.index_dir / "dense_basic.npy").astype("float32")


def search(question: str, task_id: str | None, k: int = TOP_K) -> pd.DataFrame:
    from ..index import models
    df, E = _index()
    s = E @ models.embed([question])[0]
    if task_id is not None:
        s = np.where(df.task_id.to_numpy() == task_id, s, -np.inf)
    top = np.argsort(-s)[:k]
    return df.iloc[[i for i in top if np.isfinite(s[i])]]


async def basic(question: str, task_id: str | None) -> dict:
    t0, ctr = time.perf_counter(), llm.start_counter()
    hits = search(question, task_id)
    evidence = [dict(chunk_id=r.chunk_id, doc=r.doc_id, text=r.text) for r in hits.itertuples()]
    answer = await llm.complete(BASIC_PROMPT.format(context="\n\n".join(e["text"] for e in evidence),
                                                    question=question),
                                system="You are a helpful assistant.", model=CFG.strong_llm, purpose="generate_basic")
    return dict(answer=answer, evidence=evidence, route="basic", llm_calls=ctr["calls"],
                seconds=round(time.perf_counter() - t0, 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["build"])
    ap.parse_args()
    print(json.dumps(build(), indent=1))
