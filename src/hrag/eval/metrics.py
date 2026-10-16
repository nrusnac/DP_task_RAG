"""Retrieval metrics over ranked chunk ids vs. a (silver) gold set."""
from __future__ import annotations

import math


def recall_at(ranked: list[str], gold: set[str], k: int) -> float:
    """Hit rate: 1 if any gold chunk is in the top k (one relevant chunk is enough to answer)."""
    return float(any(c in gold for c in ranked[:k]))


def mrr(ranked: list[str], gold: set[str]) -> float:
    for r, c in enumerate(ranked, 1):
        if c in gold:
            return 1.0 / r
    return 0.0


def ndcg_at(ranked: list[str], gold: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(r + 1) for r, c in enumerate(ranked[:k], 1) if c in gold)
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def summarize(rows: list[dict], ks=(1, 5, 10)) -> dict:
    n = len(rows)
    if not n:
        return {}
    out = {f"recall@{k}": round(sum(recall_at(r["ranked"], r["gold"], k) for r in rows) / n, 3) for k in ks}
    out["mrr"] = round(sum(mrr(r["ranked"], r["gold"]) for r in rows) / n, 3)
    out["ndcg@10"] = round(sum(ndcg_at(r["ranked"], r["gold"], 10) for r in rows) / n, 3)
    out["n"] = n
    return out
