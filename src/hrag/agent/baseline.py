"""Non-agentic baselines used in the evaluation.

- single_shot: hybrid + rerank top-8 chunks -> one generation call (a strong "advanced but fixed" pipeline).
- naive:       dense-only top-8 on the raw-text index -> one generation call (classic naive RAG).
"""
from __future__ import annotations

import time

from .. import llm
from ..config import CFG
from . import prompts as P
from .graph import _matters, retriever


async def single_shot(question: str, task_id: str | None, variant: str = "ctx", mode: str = "hybrid",
                      rerank: bool = True) -> dict:
    t0, ctr = time.perf_counter(), llm.start_counter()
    hits = retriever(variant).search(question, task_id=task_id, mode=mode, rerank=rerank)
    evidence = [dict(chunk_id=h.chunk_id, doc=h.doc_label(), text=h.meta["chunk_text"], score=h.score) for h in hits]
    ev_txt = "\n\n".join(f"[{i}] ({e['doc']})\n{e['text']}" for i, e in enumerate(evidence, 1))
    t1 = time.perf_counter()
    answer = await llm.complete(P.GENERATE.format(insufficient="", matter=_matters().get(task_id, "corpus"),
                                                  question=question, evidence=ev_txt),
                                system=P.GENERATE_SYSTEM, model=CFG.strong_llm, purpose="generate")
    return dict(answer=answer, evidence=evidence, route="fixed", llm_calls=ctr["calls"],
                seconds=round(time.perf_counter() - t0, 1), gen_seconds=round(time.perf_counter() - t1, 1))


async def naive(question: str, task_id: str | None) -> dict:
    return await single_shot(question, task_id, variant="raw", mode="dense", rerank=False)
