# 07 · Decision log

Each entry: the decision, the alternatives considered, and why. Numbers come from the prep smoke test on a Colab T4
(`prep/colab_smoke_test.ipynb`) or from this repo's eval.

| # | Decision | Alternatives | Why |
|---|---|---|---|
| 1 | **Subset: the whole `antitrust-competition` area** (31 matters, 230 docs, ~2.1M tokens) | a handful of tasks; a cross-area mix | Big enough that RAG beats long-context stuffing (largest matter is 198k tokens); small enough to index on a T4 in minutes; coherent domain with realistic distractors |
| 2 | **LLM via `claude_agent_sdk` + subscription OAuth, with an API-key fallback** | API key only | The task prefers the SDK. The token lives in Colab Secrets, is passed only to the SDK subprocess, and is never printed. `ANTHROPIC_API_KEY` is blanked in that subprocess because it would otherwise take precedence and bill the API. The fallback lets reviewers without a subscription run it |
| 3 | **SDK as a plain LLM** (`tools=[]`, `setting_sources=[]`, `max_turns=1`, or 2 with a JSON schema) | let the SDK be the agent | Orchestration belongs in LangGraph (explicit, testable, traced). Structured output measured at 2 turns, hence `max_turns=2`. Thinking disabled for router/grader calls (Haiku spent thinking tokens on trivial calls) |
| 4 | **Haiku for routing/grading/rewriting/sufficiency, Sonnet for planning/generation/judging** | one model everywhere | Short classification calls don't need the strong model; ~2 s/call either way on Colab, so the gain is quota, not latency |
| 5 | **Parse docx XML directly with lxml** | python-docx, unstructured, docling | python-docx drops tracked insertions (the redline task needs them) and crashes on malformed XML; docling/unstructured are heavy installs for Colab |
| 6 | **Structure-aware chunks ≤ ~600 tokens, tables split with repeated header** | fixed 512-token windows | Clauses and tables stay intact; self-contained chunks make parent-child unnecessary |
| 7 | **No parent-child retrieval** | small-to-big | Conflicts with CRAG knowledge refinement (widening vs. narrowing). Contextual headers + self-contained chunks cover its benefit |
| 8 | **Contextual Retrieval with Qwen2.5-3B on vLLM, one prompt per document** | per-chunk prompts (Anthropic's recipe); Claude Haiku; headers only | Per-chunk measured 2.62 s/chunk (≈ 2.9 h); per-document 0.98 s/chunk with 99.6% coverage and more specific contexts. Keeps the GPU doing the bulk work and the subscription quota for reasoning. Headers-only kept as an ablation variant |
| 9 | **vLLM in its own venv, as a subprocess** | install into the notebook env | vLLM pins its own torch (would break the notebook env); a process exit is the only guaranteed way to release its 85% VRAM reservation |
| 10 | **BGE-M3 + bge-reranker-v2-m3** | e5/gte embeddings, Cohere/Jina rerank APIs | Strong open models, same family, run fp16 on a T4 (1.8 GB / ~50 chunks/s; rerank 30 pairs in 0.75 s); no external API |
| 11 | **Exact numpy dense search, no FAISS** | FAISS flat/IVF, HNSW | 5.8k × 1024 vectors: exact search ≈ 1 ms. ANN only adds approximation and a dependency. Swap in above ~10⁶ chunks |
| 12 | **BM25 via bm25s with a legal token pattern** | rank_bm25, BGE-M3 sparse | Fast, supports a `weight_mask` (scoping) and save/load; the pattern keeps "3,552"/"14.3"/"50.0%" whole |
| 13 | **RRF (k=60) instead of score blending** | weighted sum of normalised scores | BM25 and cosine scales are incomparable; RRF is tuning-free and rewards agreement |
| 14 | **Scope = the matter's data room by default** | corpus-wide search | In the unscoped smoke test, 2 of the top-3 evidence strips were other matters' HHI tables (2,666 and ~927); confidentiality |
| 15 | **Agentic module at the retrieval step (Adaptive-RAG + CRAG)** | agentic generation (self-critique); a ReAct agent with a search tool | Retrieval is where failures originate; all-pass grading punishes misses; bounded explicit graph is auditable, a free-form ReAct loop is not ([04](04-agentic-module.md)) |
| 16 | **Router = Haiku with a JSON schema** | trained T5 classifier (paper) | No routing labels, no time to train; ~2 s per call |
| 17 | **CRAG evaluator = reranker thresholds + LLM only when ambiguous** | fine-tuned T5 (paper); LLM grades everything | Reuses a model we already run on the GPU; LLM cost only where the score is uninformative. Thresholds calibrated with cross-matter negatives: τ_low 0.65 / τ_high 0.92 were error-free on 120 calibration queries; ~11% of retrievals go to the LLM |
| 18 | **No web search in CRAG; abstain** | web fallback (paper) | Matter-specific, confidential facts. "Not found" beats a fabricated figure |
| 19 | **Eval set from Harvey criteria + silver labels by key-fact matching** | hand labelling; LLM-labelled relevance | Dataset has no `sources`; string-matching verbatim facts is cheap, objective, reproducible. Drafting tasks excluded (their criteria grade the lawyer's own deliverable, not the documents) |
| 20 | **Correctness judged with Harvey's own rubric prompt** | custom judge prompt | Same standard as the benchmark; reasoning-before-verdict schema as in `lab_core` |
| 21 | **Phoenix in-notebook** | Langfuse, LangSmith | No account, runs in Colab, OpenInference conventions |
| 22 | **`JAX_PLATFORMS=cpu` set in `hrag.config`** | – | Colab ships GPU JAX; `import bm25s` runs a JAX op at import, and JAX then pre-allocates 75% of VRAM outside torch (we hit an OOM: 11.3 GB held, torch saw 0.01 GB) |
| 23 | **Tuned the agent after a dry run** (router: "when in doubt, simple"; 1 sufficiency iteration; own reasoning labelled "Analysis:") | keep the first version | The first dry run sent targeted questions down the complex path (20–29 LLM calls, memo-length answers to narrow questions), and the faithfulness judge flagged unlabelled legal inferences |
| 24 | **QA questions must ask for everything their criterion requires** (incl. recommendations) | factual questions only | Otherwise the Harvey judge fails a correct factual answer because the criterion wanted advice; that penalised every system equally and hid real differences |
