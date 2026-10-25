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
    # Every antitrust matter is already in the corpus; these deal-regulatory matters (merger control / HSR / SPA
    # closing conditions from corporate-ma, CFIUS from international-trade-sanctions) add ~38% more documents and
    # give the eval matters it has never seen (docs/01-dataset.md). draft-* matters add documents but no questions.
    extra_tasks: tuple = (
        "corporate-ma/compare-merger-filing-requirements-across-multiple-jurisdictions",
        "corporate-ma/compare-merger-control-thresholds",
        "corporate-ma/analyze-sellers-counsel-markup-of-stock-purchase-agreement",
        "corporate-ma/extract-closing-conditions",
        "corporate-ma/analyze-counterparty-spa-markup",
        "corporate-ma/review-material-contract-review",
        "corporate-ma/identify-issues-in-counterpartys-draft-stock-purchase-agreement",
        "corporate-ma/draft-merger-notification-filing",
        "international-trade-sanctions/extract-national-security-factors-from-transaction-summary",
        "international-trade-sanctions/identify-compliance-and-substantive-deficiencies-in-draft-cfius-voluntary-notice",
        "international-trade-sanctions/identify-issues-in-cfius-mandatory-declaration",
        "international-trade-sanctions/analyze-counterparty-markup-of-mitigation-agreement",
        "international-trade-sanctions/compare-mitigation-agreement-against-national-security-terms",
    )
    raw_dir: Path = _env("RAW_DIR", REPO_ROOT / "data" / "raw")
    artifacts_dir: Path = _env("ARTIFACTS_DIR", REPO_ROOT / "artifacts")
    index_dir: Path = _env("INDEX_DIR", REPO_ROOT / "data" / "index")

    # chunking
    max_chunk_tokens: int = _env("MAX_CHUNK_TOKENS", 600)
    chunk_overlap_tokens: int = _env("CHUNK_OVERLAP_TOKENS", 60)

    # models
    embed_model: str = _env("EMBED_MODEL", "BAAI/bge-m3")
    rerank_model: str = _env("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    context_model: str = _env("CONTEXT_MODEL", "Qwen/Qwen3.5-4B")  # fallback: Qwen/Qwen3-4B-Instruct-2507
    fast_llm: str = _env("FAST_LLM", "haiku")      # router, grader, rewrite, sufficiency
    strong_llm: str = _env("STRONG_LLM", "sonnet")  # planning, generation, judge

    # retrieval
    dense_k: int = 50
    bm25_k: int = 50
    rrf_k: int = 60
    rerank_candidates: int = 30
    rerank_top: int = 8
    # "rrf": final order = RRF of the first-stage (hybrid) order and the reranker order; "replace": reranker order only.
    # Hit.score is the rerank score either way, so the CRAG thresholds keep their meaning.
    rerank_fusion: str = _env("RERANK_FUSION", "rrf")

    # CRAG thresholds on the top rerank score. Defaults come from calibration on the header index (60 answerable
    # questions vs. 60 cross-matter negatives): no answerable question scored below 0.65 and no unanswerable one above
    # 0.92; ~11% of retrievals fall in between and go to the LLM grader. Overridden at runtime by
    # artifacts/eval/crag_calibration.json (see agent.graph.load_calibration).
    crag_high: float = _env("CRAG_HIGH", 0.92)
    crag_low: float = _env("CRAG_LOW", 0.65)
    # calibration: each threshold may misclassify at most this share of its own class (tau_low: answerable questions
    # scoring below it; tau_high: unanswerable ones scoring above it), so one outlier can't move a threshold
    crag_tail: float = _env("CRAG_TAIL", 0.02)
    strip_threshold: float = _env("STRIP_THRESHOLD", 0.50)
    strips_per_subq: int = 12
    max_corrective_rounds: int = 2
    max_sufficiency_iters: int = 1
    gen_strips_per_subq: int = 8   # evidence passed to generation: best strips per sub-question...
    gen_max_evidence: int = 16     # ...and in total (round-robin across sub-questions)

    # LLM concurrency (measured ~1.1 calls/s at 8-way on Colab's 2 vCPUs)
    llm_concurrency: int = _env("LLM_CONCURRENCY", 6)

    extra: dict = field(default_factory=dict)

    @property
    def area_dir(self) -> Path:
        return self.raw_dir / "harvey-labs" / "tasks" / self.practice_area


CFG = Config()
