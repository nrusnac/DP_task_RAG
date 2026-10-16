"""A5: build the dense (BGE-M3) and sparse (BM25) indexes for one or more text variants.

Variants (for the Contextual Retrieval ablation):
  raw     - chunk text only (naive baseline)
  header  - deterministic header + chunk text
  ctx     - header + LLM context (artifacts/contexts.jsonl) + chunk text

    python -m hrag.index.build --variants header ctx
"""
from __future__ import annotations

import argparse
import json
import re
import time

import numpy as np
import pandas as pd

from ..config import CFG
from ..ingest.contextualize import load_contexts

# Legal-aware BM25 tokens: keep figures and references whole ("3,552", "14.3", "50.0%", "$1.2"), drop 1-char words
TOKEN_PATTERN = r"(?u)\$?\d+(?:[.,]\d+)*%?|\b\w\w+\b"


def bm25_tokenize(texts: list[str]) -> list[list[str]]:
    import bm25s  # noqa: F401  (import after config set JAX_PLATFORMS=cpu)
    from bm25s.stopwords import STOPWORDS_EN
    stop = set(STOPWORDS_EN)
    rx = re.compile(TOKEN_PATTERN)
    return [[t for t in rx.findall(s.lower()) if t not in stop] for s in texts]


def load_chunks(cfg=CFG) -> pd.DataFrame:
    return pd.read_parquet(cfg.artifacts_dir / "chunks.parquet")


def variant_texts(df: pd.DataFrame, variant: str, contexts: dict[str, str] | None = None) -> list[str]:
    if variant == "raw":
        return df.text.tolist()
    if variant == "header":
        return (df.header + "\n" + df.text).tolist()
    if variant == "ctx":
        contexts = contexts if contexts is not None else load_contexts(CFG.artifacts_dir / "contexts.jsonl")
        return [f"{h}\nContext: {contexts[c]}\n{t}" if contexts.get(c) else f"{h}\n{t}"
                for h, c, t in zip(df.header, df.chunk_id, df.text)]
    raise ValueError(variant)


def build(variants=("header", "ctx"), cfg=CFG) -> dict:
    import bm25s
    from .models import embed

    df = load_chunks(cfg)
    cfg.index_dir.mkdir(parents=True, exist_ok=True)
    contexts = load_contexts(cfg.artifacts_dir / "contexts.jsonl")
    report = {}
    for v in variants:
        if v == "ctx" and not contexts:
            report[v] = "skipped: artifacts/contexts.jsonl missing"
            continue
        texts = variant_texts(df, v, contexts)
        t0 = time.perf_counter()
        E = embed(texts, show_progress=True)
        np.save(cfg.index_dir / f"dense_{v}.npy", E.astype("float16"))
        t_dense = time.perf_counter() - t0
        t0 = time.perf_counter()
        bm = bm25s.BM25()
        bm.index(bm25_tokenize(texts), show_progress=False)
        bm.save(str(cfg.index_dir / f"bm25_{v}"))
        report[v] = dict(n=len(texts), dense_s=round(t_dense, 1), bm25_s=round(time.perf_counter() - t0, 1),
                         ctx_coverage=round(sum(c in contexts for c in df.chunk_id) / len(df), 3) if v == "ctx" else None)
    (cfg.index_dir / "build_report.json").write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["header", "ctx"])
    print(json.dumps(build(ap.parse_args().variants), indent=1))
