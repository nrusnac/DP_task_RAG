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
  - **Our variant sends each document once** with inline `[[c1]]…[[cN]]` markers and asks for one line per chunk. Measured **0.98 s/chunk, 99.6% coverage** on a 20-doc benchmark; the full run on `main` (Qwen2.5-3B) covered **98.3% of 5,756 chunks in ~90 min** (the rest keep the header only). Contexts are more specific too.
  - Documents over 10k tokens run in windows, each window carrying the document's opening for orientation.
- **Model:** **Qwen3.5-4B** on vLLM (the newest small Qwen; there is no small Qwen3.6, whose open models are 27B and 35B-A3B and don't fit a T4). It replaces Qwen2.5-3B-Instruct, which wrote the contexts on `main`.
  - fp16, because the T4 has no bf16; thinking disabled through the chat template (`enable_thinking=False`); no image inputs allowed, so vLLM reserves nothing for its vision encoder.
  - Qwen3.5's hybrid Gated-DeltaNet attention is new on a T4 in fp16, so notebook 01 runs a 2-document smoke test first; the fallback is `Qwen/Qwen3-4B-Instruct-2507` (a standard transformer).
  - It runs in its own venv as a subprocess, so its torch can't break the notebook and its VRAM is released on exit. Every output row records the model, so contexts from two models are never mixed.
- **Ablation:** `raw` vs. `header` vs. `ctx` index variants.

## 3. Hybrid search + Reciprocal Rank Fusion
- **Dense:** BGE-M3, 1024-dim, normalised, cosine via an exact numpy matmul (~1 ms at 8.6k vectors; an ANN index would only add approximation).
- **Sparse:** BM25 (`bm25s`) with a **legal-aware token pattern** that keeps `3,552`, `14.3`, `50.0%` and `$1.2` as single tokens.
- **Fusion:** RRF(d) = Σᵣ 1/(60 + rankᵣ(d)). It uses ranks rather than scores because BM25 (unbounded) and cosine (~0.3–0.9) live on incomparable scales, and it rewards agreement between the two retrievers.
- After a CRAG rewrite, the dense side is queried with the **HyDE passage and the original sub-question** (both vote in RRF), while BM25 uses the rewritten keywords. The reranker also scores against the **original sub-question**, not the keywords (`search(question=...)`). On `main` it scored against the keywords, so a rewritten round's scores were on a different footing from the thresholds CRAG had calibrated on natural-language questions.

## 4. Cross-encoder reranking
- **Retrieve wide, rerank narrow:** dense top-50 ∥ BM25 top-50 → RRF top-30 → **bge-reranker-v2-m3** → top-8.
- The cross-encoder reads (query, chunk) *jointly*, so it is far more precise than the bi-encoder, but it costs one forward pass per pair (0.75 s for 30 on the T4).
- **Measured property:** its sigmoid scores are compressed high (median 0.76 even for weak candidates), so CRAG thresholds are calibrated on the dev split rather than fixed at 0.5 (`run_eval.py calibrate`).
- **Rank fusion with the first stage (`CFG.rerank_fusion="rrf"`, default):** on `main` the reranker order *replaced* the hybrid order, and it lowered recall@5 (header index: hybrid 0.912 → reranked 0.860; it helped recall@1). Now the final order is RRF of the hybrid order and the reranker order, so a chunk both stages like wins and one stage's mistake is damped. `Hit.score` stays the rerank score, so CRAG's thresholds keep their meaning; because the top hit no longer always has the highest score, CRAG uses the **max** score of the hits. `"replace"` keeps the old behaviour, and the retrieval eval reports both.
  - **Measured locally on the grown corpus (`header` index):** on dev (82 labelled), `rrf` vs `replace` = R@5 0.927 vs 0.902, R@10 0.976 vs 0.939, but R@1 0.671 vs 0.732 and MRR 0.786 vs 0.803. The rule fixed before looking (keep `rrf` unless `replace` wins on both R@5 and MRR on dev) kept `rrf`. On test (98 labelled, informational only) `replace` is ahead on R@1/R@5/MRR (0.592/0.898/0.723 vs 0.571/0.867/0.703) and reranking helps a lot over plain hybrid (R@5 0.827 → 0.898), so `main`'s "reranking costs recall@5" was a property of the dev matters. Differences are 2–3 questions either way. Choosing `replace` because of these test numbers would turn test into a tuning set; `HRAG_RERANK_FUSION=replace` switches it.

## 5. Metadata scoping
Every chunk carries `task_id`. A question about a matter searches only that matter's data room (a boolean mask on the dense scores and BM25's `weight_mask`).
- **Without it:** in the smoke test, 2 of the top-3 evidence strips for the South Central HHI question came from *other matters'* HHI tables (2,666 and ~927).
- In legal practice, mixing matters is both a correctness bug and a confidentiality bug.
- `task_id=None` searches the whole corpus.

## 6. Query transformation
- **Decomposition:** the `complex` route splits the question into 2–5 sub-questions.
- **Keyword rewrite:** BM25-oriented terms.
- **HyDE** ([Gao et al., 2022](https://arxiv.org/abs/2212.10496)): embed a hypothetical answer passage, which looks more like a document than the question does.
- Transformations are applied **only when the agent decides they are needed**: decomposition on complex queries, and rewrite + HyDE only after CRAG judges a retrieval incorrect or ambiguous.

## 9. Grounded generation
- The generator sees only refined, numbered evidence strips.
- It must cite each claim `[n]`, quote figures exactly, and write "The documents provided do not establish …" for sub-questions CRAG marked *insufficient* (and treat *weak* ones as indirect evidence, [04](04-agentic-module.md)).
- Rules added on this branch, one per pattern seen in the unfaithful answers ([05](05-evaluation.md#unfaithful-answer-patterns)): no "only/all/none" claims the evidence doesn't make; inferences and characterizations under "Analysis:"; headings and bottom lines add no claims; proposals and redlines are not presented as agreed terms; signatures, dates and markings exactly as shown; facts attributed only to the document or party that states them; a final self-check that drops uncited sentences. The baselines use the same prompt, so the comparison stays fair.
- The agent passes at most 8 strips per sub-question and 16 in total (round-robin, best first), and drops strips already contained in a whole chunk it passes.
- Citations map back to `chunk_id`s, so they are checkable (the eval measures them).
