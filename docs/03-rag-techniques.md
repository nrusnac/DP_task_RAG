# 03 · Advanced RAG techniques: what, why, how

Each technique fixes a specific way naive RAG (fixed-size chunks → embed → top-k → generate) fails on legal data rooms.

| # | Technique | Failure it fixes | Where |
|---|---|---|---|
| 1 | Structure-aware parsing and chunking | chunks cut clauses and tables in half | `ingest/parsers.py`, `ingest/chunking.py` |
| 2 | Contextual Retrieval (per-document variant) | chunks lose meaning out of context | `ingest/contextualize.py`, `scripts/contextualize_vllm.py` |
| 3 | Hybrid search (BGE-M3 + BM25) with RRF | dense misses exact figures and section numbers | `retrieve/hybrid.py` |
| 4 | Cross-encoder reranking | noisy top-k | `retrieve/hybrid.py`, `index/models.py` |
| 5 | Metadata scoping (data-room filter) | cross-matter contamination | `HybridRetriever.mask` |
| 6 | Query transformation: decomposition, keyword rewrite, HyDE | the user's words are a poor query | agent `plan`, `rewrite` |
| 7 | Adaptive-RAG routing | one pipeline for all query complexities | agent `route` ([04](04-agentic-module.md)) |
| 8 | CRAG: grading, knowledge refinement, corrective retrieval, abstention | blind trust in retrieval | agent `grade`, `refine`, `rewrite` ([04](04-agentic-module.md)) |
| 9 | Grounded generation with citations | unverifiable answers | agent `generate` |

## 1. Structure-aware chunking
- **How:** a section (heading + paragraphs) becomes one chunk if it is ≤ ~600 tokens; otherwise it is split at paragraph boundaries with a 60-token overlap. Oversized paragraphs are split by sentences.
  - **Tables:** small ones are inlined; large ones are split by row blocks **with the header row repeated**, so a block of numbers still says which column is which.
  - **Slides and e-mails:** atomic.
  - **Tiny chunks** are merged forward.
  - Every chunk gets a deterministic header `[doc title] (type) › section path`.
- **Why:** the unit of meaning in legal text is the clause or section. Self-contained chunks are also what lets us drop parent-child retrieval (see the [decision log](07-decision-log.md)).

## 2. Contextual Retrieval
- **Idea** ([Anthropic, 2024](https://www.anthropic.com/news/contextual-retrieval)): an LLM writes a 1–2 sentence context per chunk, using the whole document, and it is prepended before embedding and BM25 indexing. Anthropic reports 35% fewer top-20 retrieval failures, 49% with contextual BM25, and 67% with reranking.
- **The T4 forced a redesign:**
  - The original recipe (one document + one chunk per prompt) measured **2.62 s/chunk ≈ 2.9 h**: the ~8k-token document is re-sent for every ~60-token output, and the prefix cache helped little.
  - **Our variant sends each document once** with inline `[[c1]]…[[cN]]` markers and asks for one line per chunk. Measured **0.98 s/chunk, 99.6% coverage**, and more specific contexts.
  - Documents over 10k tokens run in windows, each window carrying the document's opening for orientation.
- **Model:** Qwen2.5-3B-Instruct on vLLM (fp16; Triton attention, because the T4 has no FlashAttention). It runs in its own venv as a subprocess, so its torch can't break the notebook and its VRAM is released on exit.
- **Ablation:** `raw` vs. `header` vs. `ctx` index variants.

## 3. Hybrid search + Reciprocal Rank Fusion
- **Dense:** BGE-M3, 1024-dim, normalised, cosine via an exact numpy matmul (~1 ms at 5.8k vectors; an ANN index would only add approximation).
- **Sparse:** BM25 (`bm25s`) with a **legal-aware token pattern** that keeps `3,552`, `14.3`, `50.0%` and `$1.2` as single tokens.
- **Fusion:** RRF(d) = Σᵣ 1/(60 + rankᵣ(d)). It uses ranks rather than scores because BM25 (unbounded) and cosine (~0.3–0.9) live on incomparable scales, and it rewards agreement between the two retrievers.
- After a CRAG rewrite, the dense side is queried with the **HyDE passage and the original question** (both vote in RRF), while BM25 uses the rewritten keywords.

## 4. Cross-encoder reranking
- **Retrieve wide, rerank narrow:** dense top-50 ∥ BM25 top-50 → RRF top-30 → **bge-reranker-v2-m3** → top-8.
- The cross-encoder reads (query, chunk) *jointly*, so it is far more precise than the bi-encoder, but it costs one forward pass per pair (0.75 s for 30 on the T4).
- **Measured property:** its sigmoid scores are compressed high (median 0.76 even for weak candidates), so CRAG thresholds are calibrated on the eval set rather than fixed at 0.5 (`run_eval.py calibrate`).

## 5. Metadata scoping
Every chunk carries `task_id`. A question about a matter searches only that matter's data room (a boolean mask on the dense scores and BM25's `weight_mask`).
- **Without it:** the smoke test answered **1,882**, another matter's HHI, instead of **3,552**.
- In legal practice, mixing matters is both a correctness bug and a confidentiality bug.
- `task_id=None` searches the whole corpus.

## 6. Query transformation
- **Decomposition:** the `complex` route splits the question into 2–5 sub-questions.
- **Keyword rewrite:** BM25-oriented terms.
- **HyDE** ([Gao et al., 2022](https://arxiv.org/abs/2212.10496)): embed a hypothetical answer passage, which looks more like a document than the question does.
- Transformations are applied **only when the agent decides they are needed**: decomposition on complex queries, and rewrite + HyDE only after CRAG judges a retrieval incorrect or ambiguous.

## 9. Grounded generation
- The generator sees only refined, numbered evidence strips.
- It must cite each claim `[n]`, quote figures exactly, and write "The documents provided do not establish …" for sub-questions CRAG marked *insufficient*.
- Citations map back to `chunk_id`s, so they are checkable (the eval measures them).
