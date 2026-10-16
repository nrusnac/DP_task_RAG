# DP_task_RAG: advanced + agentic RAG over Harvey LAB legal data rooms

RAG over the **`antitrust-competition`** practice area of [Harvey LAB](https://github.com/harveyai/harvey-labs/tree/main/tasks): 31 matters, 230 documents (docx/xlsx/eml/pptx), ~2.1M tokens.
- **Runs on a free Colab T4.** The GPU runs embeddings, reranking, CRAG strip scoring and the one-off contextualization LLM.
- **Uses Claude through `claude_agent_sdk`** for reasoning, with the subscription token kept in Colab Secrets. An Anthropic-API fallback is included.

| Notebook | What | Colab |
|---|---|---|
| `notebooks/02_demo_and_eval.ipynb` | build indexes, demo the agent, run the evaluation, Phoenix traces | [Open in Colab](https://colab.research.google.com/github/nrusnac/DP_task_RAG/blob/main/notebooks/02_demo_and_eval.ipynb) |
| `notebooks/01_contextualize_colab.ipynb` | one-off Contextual Retrieval job (Qwen2.5-3B on vLLM, ~80–90 min, output committed) | [Open in Colab](https://colab.research.google.com/github/nrusnac/DP_task_RAG/blob/main/notebooks/01_contextualize_colab.ipynb) |

## What's inside
**Advanced RAG techniques** ([docs/03](docs/03-rag-techniques.md)):
- structure-aware parsing (tracked changes, malformed XML) and chunking
- **Contextual Retrieval**, in a per-document variant that's 2.7× faster on a T4
- **hybrid BGE-M3 + BM25 with RRF**
- **cross-encoder reranking**
- data-room scoping
- query decomposition / rewriting / **HyDE**

**Agentic module** ([docs/04](docs/04-agentic-module.md)): a LangGraph controller for the **retrieval step**.
- **Adaptive-RAG** routes each query (direct / simple / complex).
- **CRAG** grades the evidence with reranker scores (Claude only when the score is ambiguous), refines it to relevant sentence strips, rewrites failing queries with keywords + HyDE, and **abstains** instead of guessing.
- A sufficiency loop fills gaps on complex queries.
- Every decision is logged and traced.

**Evaluation** ([docs/05](docs/05-evaluation.md)):
- a QA set derived from Harvey rubric criteria, with silver retrieval labels
- retrieval ablations
- CRAG threshold calibration
- naive vs. single-shot vs. agent, judged with **Harvey's own rubric prompt** plus faithfulness
- the Adaptive-RAG ablation

**Observability** ([docs/06](docs/06-observability.md)): Arize Phoenix in the notebook, with LangGraph auto-instrumented and manual retriever, reranker and LLM spans.

```mermaid
flowchart LR
    D[(230 docs)] --> P[parse + chunk] --> C[contextualize<br/>Qwen2.5-3B · vLLM · T4] --> I[BGE-M3 + BM25]
    Q([question]) --> R{router} --> RET[hybrid + rerank] --> G{CRAG} --> RF[refine] --> GEN[Claude · cited answer]
    G -.rewrite + HyDE.-> RET
    I --- RET
```

All design choices, with alternatives and measurements: [docs/07-decision-log.md](docs/07-decision-log.md). Dataset findings: [docs/01](docs/01-dataset.md). Architecture and the full node-by-node flow: [docs/02](docs/02-architecture.md).

## Results
**Key findings.** The full analysis is in [docs/05](docs/05-evaluation.md); the tables below are generated from `artifacts/eval/`.
- **Hybrid > BM25 or dense alone:** R@1 0.684 vs. 0.614, MRR 0.784 vs. 0.745 / 0.728. **Reranking** raises R@1 (0.667 → 0.702 on `header`) at some cost to R@5.
- **Scoping to the matter's data room matters:** corpus-wide dense search puts ~17% of top-5 hits in *other* matters; hybrid + rerank cuts that to ~7%, and scoping to 0.
- **Contextual Retrieval didn't measurably help on this eval.** The labels are verbatim-fact matches, a matter has only ~186 chunks, and the cheap deterministic header already gives most of the context. Reported as is.
- **CRAG thresholds are measured:** answerable vs. cross-matter-negative questions are almost perfectly separated by the top rerank score (negatives median 0.017, positives p10 0.89).
- **Generation on answerable questions** (30 questions): single-shot advanced RAG 0.967, naive 0.90, agent 0.867.
  - Every agent failure was **over-pruning by CRAG refinement** on list, enumeration and clause questions, with the gold chunk retrieved and graded correct. Fixed by passing clearly relevant hits whole.
  - The agent uses **85% less evidence** and has the highest faithfulness.
- **Adaptive-RAG:** 0.90 at 4.1 calls, vs. always-complex 0.85 at 7.6 calls and always-simple 0.85 at 1.45 calls.
- **Unanswerable questions** (the agent's real job) are in the last table.

<!-- RESULTS:START -->
**Retrieval** (57 questions with silver labels, scoped to the matter's data room)

| variant / mode | recall@1 | recall@5 | recall@10 | mrr | ndcg@10 | foreign@5 | ms_per_query |
|---|---|---|---|---|---|---|---|
| `raw/bm25` | 0.614 | 0.895 | 0.965 | 0.745 | 0.653 | 0.0 | 13 |
| `raw/dense` | 0.614 | 0.895 | 0.947 | 0.728 | 0.586 | 0.0 | 56 |
| `raw/hybrid` | 0.684 | 0.912 | 0.965 | 0.784 | 0.672 | 0.0 | 54 |
| `raw/hybrid+rerank` | 0.614 | 0.877 | 0.912 | 0.734 | 0.649 | 0.0 | 958 |
| `raw/dense (corpus-wide)` | 0.596 | 0.86 | 0.912 | 0.706 | 0.546 | 0.168 | 56 |
| `raw/hybrid+rerank (corpus-wide)` | 0.614 | 0.842 | 0.912 | 0.718 | 0.627 | 0.07 | 1001 |
| `header/bm25` | 0.649 | 0.895 | 0.965 | 0.754 | 0.671 | 0.0 | 11 |
| `header/dense` | 0.649 | 0.825 | 0.93 | 0.729 | 0.605 | 0.0 | 61 |
| `header/hybrid` | 0.667 | 0.912 | 0.982 | 0.777 | 0.667 | 0.0 | 57 |
| `header/hybrid+rerank` | 0.702 | 0.86 | 0.912 | 0.77 | 0.687 | 0.0 | 1047 |
| `header/dense (corpus-wide)` | 0.649 | 0.825 | 0.877 | 0.719 | 0.555 | 0.165 | 52 |
| `header/hybrid+rerank (corpus-wide)` | 0.684 | 0.842 | 0.912 | 0.758 | 0.653 | 0.074 | 1013 |
| `ctx/bm25` | 0.649 | 0.877 | 0.947 | 0.748 | 0.666 | 0.0 | 10 |
| `ctx/dense` | 0.667 | 0.842 | 0.93 | 0.737 | 0.605 | 0.0 | 56 |
| `ctx/hybrid` | 0.632 | 0.877 | 0.965 | 0.74 | 0.633 | 0.0 | 52 |
| `ctx/hybrid+rerank` | 0.684 | 0.842 | 0.912 | 0.752 | 0.669 | 0.0 | 1072 |
| `ctx/dense (corpus-wide)` | 0.614 | 0.789 | 0.912 | 0.695 | 0.53 | 0.172 | 50 |
| `ctx/hybrid+rerank (corpus-wide)` | 0.684 | 0.842 | 0.912 | 0.753 | 0.655 | 0.067 | 1043 |

**CRAG calibration** (variant `ctx`): top rerank score for answerable questions (positives) vs. the same questions asked against another matter's data room (negatives). Suggested thresholds `{'crag_high': 0.95, 'crag_low': 0.09}`; coverage `{'correct_pos': 0.902, 'incorrect_neg': 0.719, 'ambiguous_share': 0.202}`

| top rerank score | positives | gold_retrieved | negatives |
|---|---|---|---|
| [0.00,0.10) | 1 | 1.0 | 42 |
| [0.10,0.30) | 0 |  | 4 |
| [0.30,0.50) | 0 |  | 3 |
| [0.50,0.70) | 1 | 1.0 | 1 |
| [0.70,0.90) | 1 | 1.0 | 5 |
| [0.90,0.95) | 4 | 0.5 | 2 |
| [0.95,0.99) | 12 | 0.92 | 0 |
| [0.99,1.00) | 38 | 0.92 | 0 |

**Generation** (Adaptive-RAG ablation, Colab T4, `ctx` index, 20 questions; Harvey rubric judge = Claude Sonnet)

| system | n | pass_rate | pass_rate_fact | pass_rate_analysis | faithful_rate | abstain_rate | evidence_has_gold | avg_evidence_words | avg_llm_calls | avg_seconds | routes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `agent` | 20 | 0.9 | 0.857 | 1.0 |  | 0.1 | 0.857 | 443.15 | 4.1 | 39.68 | simple:16, complex:4 |
| `agent:complex` | 20 | 0.85 | 0.857 | 0.833 |  | 0.1 | 0.929 | 873.95 | 7.6 | 66.525 | complex:20 |
| `agent:simple` | 20 | 0.85 | 0.857 | 0.833 |  | 0.15 | 0.857 | 280.05 | 1.45 | 19.61 | simple:20 |

**Generation** (Colab T4, `ctx` index, 30 questions; Harvey rubric judge = Claude Sonnet)

| system | n | pass_rate | pass_rate_fact | pass_rate_analysis | faithful_rate | abstain_rate | evidence_has_gold | avg_evidence_words | avg_llm_calls | avg_seconds | routes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `agent` | 30 | 0.867 | 0.85 | 0.9 | 0.7 | 0.167 | 0.889 | 335.967 | 3.133 | 24.147 | simple:27, complex:3 |
| `naive` | 30 | 0.9 | 0.95 | 0.8 | 0.633 | 0.1 | 0.944 | 1978.267 | 1.0 | 11.153 | fixed:30 |
| `single_shot` | 30 | 0.967 | 0.95 | 1.0 | 0.667 | 0.1 | 0.889 | 2226.733 | 1.0 | 12.543 | fixed:30 |
<!-- RESULTS:END -->

## Quickstart
**Colab (recommended):**
1. Open `02_demo_and_eval.ipynb` and choose *Runtime → T4 GPU*.
2. Add the secret `CLAUDE_CODE_OAUTH_TOKEN` (run `claude setup-token` locally to get one) or `ANTHROPIC_API_KEY`.
3. *Run all.*

**Local:**
```bash
pip install -e ".[obs,dev]"
python -m hrag.ingest.build                      # A1-A3: download subset, parse, chunk  -> artifacts/
python -m hrag.index.build --variants raw header ctx   # A5 (needs artifacts/contexts.jsonl for ctx)
python -m hrag.eval.run_eval retrieval           # retrieval ablations (no LLM)
python -c "from hrag.agent.graph import answer_sync as a; print(a('What is the post-merger HHI for flame retardants in the South Central region?', task_id='analyze-iss-antitrust-transaction-structure')['answer'])"
pytest -q
```

**Credentials** are read from the environment or Colab Secrets and are never printed. With `CLAUDE_CODE_OAUTH_TOKEN` (or a local `claude` login) the Agent SDK uses your subscription. With only `ANTHROPIC_API_KEY`, the API backend is used. Bring your own: please don't share tokens.

## Repository
```
src/hrag/   config · llm · ingest/ · index/ · retrieve/ · agent/ · eval/ · obs/
scripts/    contextualize_vllm.py (standalone, resumable)
notebooks/  01_contextualize_colab · 02_demo_and_eval
artifacts/  chunks.parquet · tasks.jsonl · contexts.jsonl · eval/
docs/       01-dataset · 02-architecture · 03-rag-techniques · 04-agentic-module · 05-evaluation · 06-observability · 07-decision-log
prep/       colab_smoke_test.ipynb (pre-build de-risking: SDK, GPU models, vLLM, Phoenix on a T4)
```

Data: Harvey LAB, MIT License, © Harvey AI. `artifacts/` holds chunks derived from it.
