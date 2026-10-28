"""B3-B4: scoped hybrid retrieval (BGE-M3 + BM25 -> RRF) and cross-encoder reranking.

Exact dense search is a numpy matmul: at ~9k x 1024 vectors it takes ~1 ms, so an ANN index (FAISS/HNSW) would add
approximation error and a dependency for no gain. Swap in FAISS-IVF/HNSW above ~10^6 chunks.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import CFG
from ..index.build import bm25_tokenize, load_chunks, variant_texts
from ..index import models
from ..obs import tracing


@dataclass
class Hit:
    idx: int
    chunk_id: str
    score: float  # rerank score if reranked, else RRF score
    text: str     # indexed text (header [+ context] + chunk)
    meta: dict = field(default_factory=dict)

    def doc_label(self) -> str:
        return f"{self.meta.get('doc_id')} § {self.meta.get('section') or self.meta.get('kind')}"


class HybridRetriever:
    def __init__(self, variant: str = CFG.default_variant, cfg=CFG):
        import bm25s
        self.cfg, self.variant = cfg, variant
        self.df: pd.DataFrame = load_chunks(cfg)
        self.texts = variant_texts(self.df, variant)
        self.E = np.load(cfg.index_dir / f"dense_{variant}.npy").astype("float32")
        self.bm25 = bm25s.BM25.load(str(cfg.index_dir / f"bm25_{variant}"))
        self.task_ids = self.df.task_id.to_numpy()
        self._masks: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ scoping (data-room filter)
    def mask(self, task_id: str | None) -> np.ndarray | None:
        if task_id is None:
            return None
        if task_id not in self._masks:
            m = self.task_ids == task_id
            if not m.any():
                raise ValueError(f"unknown task_id {task_id}")
            self._masks[task_id] = m
        return self._masks[task_id]

    # ------------------------------------------------------------------ retrievers
    def dense(self, query: str, task_id: str | None = None, k: int | None = None) -> list[tuple[int, float]]:
        k = k or self.cfg.dense_k
        q = models.embed([query])[0]
        s = self.E @ q
        m = self.mask(task_id)
        if m is not None:
            s = np.where(m, s, -np.inf)
        top = np.argpartition(-s, min(k, len(s) - 1))[:k]
        top = top[np.argsort(-s[top])]
        return [(int(i), float(s[i])) for i in top if np.isfinite(s[i])]

    def sparse(self, query: str, task_id: str | None = None, k: int | None = None) -> list[tuple[int, float]]:
        k = k or self.cfg.bm25_k
        toks = bm25_tokenize([query])
        m = self.mask(task_id)
        n_avail = int(m.sum()) if m is not None else len(self.df)
        k = min(k, n_avail)
        if not toks[0] or k == 0:
            return []
        idx, sc = self.bm25.retrieve(toks, k=k, show_progress=False, backend_selection="numpy",
                                     weight_mask=m.astype("float32") if m is not None else None)
        return [(int(i), float(s)) for i, s in zip(idx[0], sc[0]) if s > 0]

    @staticmethod
    def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
        """Reciprocal Rank Fusion: score(d) = sum_r 1 / (k + rank_r(d)). Uses ranks only: BM25 and cosine scales differ."""
        scores: dict[int, float] = {}
        for ranking in rankings:
            for r, i in enumerate(ranking, 1):
                scores[i] = scores.get(i, 0.0) + 1.0 / (k + r)
        return sorted(scores.items(), key=lambda kv: -kv[1])

    # ------------------------------------------------------------------ public API
    def hit(self, i: int, score: float) -> Hit:
        row = self.df.iloc[i]
        return Hit(i, row.chunk_id, score, self.texts[i],
                   dict(task_id=row.task_id, doc_id=row.doc_id, section=row.section, kind=row.kind,
                        doc_title=row.doc_title, chunk_text=row.text))

    @staticmethod
    def fuse_rerank(cands: list[Hit], scores, k: int = 60) -> list[Hit]:
        """Order candidates by RRF of their first-stage rank (list order) and their rerank rank. Each Hit keeps its
        rerank score: the CRAG thresholds are calibrated on that scale."""
        for h, s in zip(cands, scores):
            h.score = float(s)
        by_rerank = sorted(range(len(cands)), key=lambda i: -cands[i].score)
        fused = HybridRetriever.rrf([list(range(len(cands))), by_rerank], k)
        return [cands[i] for i, _ in fused]

    def search(self, query: str, task_id: str | None = None, mode: str = "hybrid", rerank: bool = True,
               dense_query: str | None = None, top: int | None = None, exclude: set[str] | None = None,
               question: str | None = None, fusion: str | None = None) -> list[Hit]:
        """mode: 'dense' | 'bm25' | 'hybrid'. After a CRAG rewrite, `query` is the BM25 keyword query, `dense_query`
        the HyDE passage and `question` the original sub-question: HyDE and the question each get a dense vote, and
        the reranker scores against the question, so rerank scores mean the same thing in every round.
        fusion: 'rrf' | 'replace' (default CFG.rerank_fusion), see config."""
        top = top or self.cfg.rerank_top
        question = question or query
        fusion = fusion or self.cfg.rerank_fusion
        with tracing.span("retrieve.hybrid", kind="RETRIEVER", **{"input.value": query, "retrieval.mode": mode,
                                                                  "retrieval.scope": task_id or "corpus",
                                                                  "retrieval.variant": self.variant}) as sp:
            t0 = time.perf_counter()
            rankings = []
            if mode in ("dense", "hybrid"):
                rankings.append([i for i, _ in self.dense(dense_query or question, task_id)])
                if dense_query and dense_query != question:  # HyDE + original question both vote
                    rankings.append([i for i, _ in self.dense(question, task_id)])
            if mode in ("bm25", "hybrid"):
                rankings.append([i for i, _ in self.sparse(query, task_id)])
            fused = self.rrf(rankings, self.cfg.rrf_k) if len(rankings) > 1 else \
                [(i, 1.0 / (self.cfg.rrf_k + r)) for r, i in enumerate(rankings[0], 1)]
            if exclude:
                fused = [(i, s) for i, s in fused if self.df.chunk_id.iat[i] not in exclude]
            cands = [self.hit(i, s) for i, s in fused[:self.cfg.rerank_candidates if rerank else top]]
            if rerank and cands:
                with tracing.span("rerank", kind="RERANKER", **{"input.value": question, "reranker.fusion": fusion,
                                                                "reranker.model_name": self.cfg.rerank_model}) as rs:
                    scores = models.rerank_scores(question, [h.text for h in cands])
                    if fusion == "rrf":
                        cands = self.fuse_rerank(cands, scores, self.cfg.rrf_k)
                    else:
                        for h, s in zip(cands, scores):
                            h.score = float(s)
                        cands.sort(key=lambda h: -h.score)
                    tracing.set_documents(rs, [dict(id=h.chunk_id, content=h.text, score=h.score)
                                               for h in cands[:top]], prefix="reranker.output_documents")
            hits = cands[:top]
            tracing.set_documents(sp, [dict(id=h.chunk_id, content=h.text, score=h.score) for h in hits])
            tracing.set_attrs(sp, **{"retrieval.ms": round((time.perf_counter() - t0) * 1000)})
            return hits
