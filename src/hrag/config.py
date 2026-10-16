"""Central configuration. Every value can be overridden by an HRAG_* environment variable."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Colab preinstalls GPU JAX and `import bm25s` runs jax.lax.top_k at import time; JAX would then pre-allocate
# 75% of the T4 outside torch (measured: 11.3 GB). Keep JAX on CPU. Must run before bm25s is imported.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default):
    raw = os.environ.get(f"HRAG_{name}")
    if raw is None:
        return default
    return type(default)(raw) if not isinstance(default, Path) else Path(raw)


@dataclass
class Config:
    # data
    harvey_repo: str = "https://github.com/harveyai/harvey-labs.git"
    practice_area: str = _env("PRACTICE_AREA", "antitrust-competition")
    raw_dir: Path = _env("RAW_DIR", REPO_ROOT / "data" / "raw")
    artifacts_dir: Path = _env("ARTIFACTS_DIR", REPO_ROOT / "artifacts")
    index_dir: Path = _env("INDEX_DIR", REPO_ROOT / "data" / "index")

    # chunking
    max_chunk_tokens: int = _env("MAX_CHUNK_TOKENS", 600)
    chunk_overlap_tokens: int = _env("CHUNK_OVERLAP_TOKENS", 60)

    # models
    embed_model: str = _env("EMBED_MODEL", "BAAI/bge-m3")
    rerank_model: str = _env("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    context_model: str = _env("CONTEXT_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    fast_llm: str = _env("FAST_LLM", "haiku")      # router, grader, rewrite, sufficiency
    strong_llm: str = _env("STRONG_LLM", "sonnet")  # planning, generation, judge

    # retrieval
    dense_k: int = 50
    bm25_k: int = 50
    rrf_k: int = 60
    rerank_candidates: int = 30
    rerank_top: int = 8

    # CRAG (reranker sigmoid scores are compressed high: median 0.76 on weak candidates, so not 0.5)
    crag_high: float = _env("CRAG_HIGH", 0.90)
    crag_low: float = _env("CRAG_LOW", 0.60)
    strip_threshold: float = _env("STRIP_THRESHOLD", 0.50)
    strips_per_subq: int = 12
    max_corrective_rounds: int = 2
    max_sufficiency_iters: int = 2

    # LLM concurrency (measured ~1.1 calls/s at 8-way on Colab's 2 vCPUs)
    llm_concurrency: int = _env("LLM_CONCURRENCY", 6)

    extra: dict = field(default_factory=dict)

    @property
    def area_dir(self) -> Path:
        return self.raw_dir / "harvey-labs" / "tasks" / self.practice_area


CFG = Config()
