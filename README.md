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
- **CRAG** grades the evidence with reranker scores (Claude only when the score is ambiguous), passes clearly relevant hits whole and reduces marginal ones to relevant sentence strips, rewrites failing queries with keywords + HyDE, and **abstains** instead of guessing.
- A sufficiency loop fills gaps on complex queries.
- Every decision is logged and traced.

**Evaluation** ([docs/05](docs/05-evaluation.md)):
- a QA set derived from Harvey rubric criteria, with silver retrieval labels
- retrieval ablations
- CRAG threshold calibration
- naive vs. single-shot vs. agent, judged with **Harvey's own rubric prompt** plus faithfulness
- the Adaptive-RAG ablation
- unanswerable questions (each asked in another matter's data room)

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
- **CRAG thresholds are measured with cross-matter negatives:** the top rerank score almost perfectly separates answerable from unanswerable questions (`ctx`: negatives median 0.011, positives p10 0.94). The error-free-band rule is outlier-sensitive (`ctx` τ 0.09 / 0.95, so 20% of retrievals are LLM-graded).
- **Generation (final Colab run, 30 questions): agent 0.967, single-shot 0.933, naive 0.90.** That's a statistical tie: agent − single-shot +3.3 points, 95% CI [0.0, +10.0] over matters, p = 1.0 ([how sure are we](docs/05-evaluation.md#how-sure-are-we)), and single-shot itself moved by one question between identical runs.
  - The first run had the agent at 0.867. 3 of its 4 failures were **CRAG refinement over-pruning** list, enumeration and clause content, with the gold chunk retrieved and graded correct. The 4th is the chunking miss below.
  - The fix (pass clearly relevant hits whole) **flipped exactly the 3 predicted questions to pass, with no regressions**.
  - Cost: 3.7 LLM calls vs. 1, and 46% less context than single-shot (both significant). Faithfulness fell after the fix (0.70 → 0.50), but that is borderline (p = 0.11), and the binary faithfulness judge proved noisy (below).
- **The one question every system fails** ("list all 25 custodians") is a **chunking issue**: each appendix table was split from the caption that says "custodians".
- **Adaptive-RAG (final, 20 questions):** all policies score 0.90. Adaptive costs 3.45 calls vs. 6.6 for always-complex and 1.45 for always-simple. Its earlier one-question edge didn't reproduce.
- **Unanswerable questions** (asked in the wrong matter's data room): single-shot declined 11/12, the agent 10/12. **Not better.** The grounded prompt does most of the declining, and CRAG's corrective search can surface look-alike evidence. A `ctx` rerun hit the Claude subscription session limit and is excluded.
- **What a deeper evaluation showed** ([docs/05](docs/05-evaluation.md#what-a-deeper-evaluation-showed-follow-up-work)): a follow-up on 126 questions from 16 unseen matters found this eval can't separate the systems. The pass rate saturates at ~95%, the binary faithfulness judge disagreed with itself on 44% of identical inputs, and 5 of 126 rubrics contradict the documents. With audited rubrics, claim-level and pairwise judges and matter-level statistics, the pass rate still tied, but where the agent took its own path a pairwise judge preferred its answer 36 times vs. 8 (p < 0.001).
- **Bottom line:** the retrieval stack (hybrid, rerank, scoping) is the accuracy foundation. The agent is *at least on par*, at lower context but higher LLM cost. Its main value is control and auditability: every failure traced to one node, and the main fix was predicted from traces and confirmed on Colab.

<!-- RESULTS:START -->
**Retrieval** (57 questions with silver labels; searched within the matter's data room unless marked corpus-wide; `foreign@5` = share of top-5 hits from other matters)

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

**Generation** (**final**, Colab T4, `ctx` index, 30 questions, after the refinement fix; Harvey rubric judge = Claude Sonnet)

| system | n | pass_rate | pass_rate_fact | pass_rate_analysis | faithful_rate | abstain_rate | evidence_has_gold | avg_evidence_words | avg_llm_calls | avg_seconds | routes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `agent` | 30 | 0.967 | 0.95 | 1.0 | 0.5 | 0.067 | 0.944 | 1201.0 | 3.733 | 28.37 | simple:24, complex:6 |
| `naive` | 30 | 0.9 | 0.95 | 0.8 | 0.7 | 0.1 | 0.944 | 1978.267 | 1.0 | 11.407 | fixed:30 |
| `single_shot` | 30 | 0.933 | 0.95 | 0.9 | 0.633 | 0.067 | 0.889 | 2226.733 | 1.0 | 12.747 | fixed:30 |

**Generation** (**final** Adaptive-RAG ablation, Colab T4, `ctx` index, 20 questions, after the fix; Harvey rubric judge = Claude Sonnet)

| system | n | pass_rate | pass_rate_fact | pass_rate_analysis | faithful_rate | abstain_rate | evidence_has_gold | avg_evidence_words | avg_llm_calls | avg_seconds | routes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `agent` | 20 | 0.9 | 0.929 | 0.833 |  | 0.0 | 0.857 | 1165.25 | 3.45 | 34.59 | simple:16, complex:4 |
| `agent:complex` | 20 | 0.9 | 0.929 | 0.833 |  | 0.0 | 1.0 | 2235.15 | 6.6 | 52.725 | complex:20 |
| `agent:simple` | 20 | 0.9 | 0.929 | 0.833 |  | 0.0 | 0.929 | 809.15 | 1.45 | 18.62 | simple:20 |

**Generation** (first Colab run, BEFORE the refinement fix, same 30 questions; Harvey rubric judge = Claude Sonnet)

| system | n | pass_rate | pass_rate_fact | pass_rate_analysis | faithful_rate | abstain_rate | evidence_has_gold | avg_evidence_words | avg_llm_calls | avg_seconds | routes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `agent` | 30 | 0.867 | 0.85 | 0.9 | 0.7 | 0.167 | 0.889 | 335.967 | 3.133 | 24.147 | simple:27, complex:3 |
| `naive` | 30 | 0.9 | 0.95 | 0.8 | 0.633 | 0.1 | 0.944 | 1978.267 | 1.0 | 11.153 | fixed:30 |
| `single_shot` | 30 | 0.967 | 0.95 | 1.0 | 0.667 | 0.1 | 0.889 | 2226.733 | 1.0 | 12.543 | fixed:30 |

**Generation** (first Colab Adaptive-RAG ablation, BEFORE the fix, same 20 questions; Harvey rubric judge = Claude Sonnet)

| system | n | pass_rate | pass_rate_fact | pass_rate_analysis | faithful_rate | abstain_rate | evidence_has_gold | avg_evidence_words | avg_llm_calls | avg_seconds | routes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `agent` | 20 | 0.9 | 0.857 | 1.0 |  | 0.1 | 0.857 | 443.15 | 4.1 | 39.68 | simple:16, complex:4 |
| `agent:complex` | 20 | 0.85 | 0.857 | 0.833 |  | 0.1 | 0.929 | 873.95 | 7.6 | 66.525 | complex:20 |
| `agent:simple` | 20 | 0.85 | 0.857 | 0.833 |  | 0.15 | 0.857 | 280.05 | 1.45 | 19.61 | simple:20 |

**Generation** (targeted local check of the fix, `header` index, 4 failed + 6 passed questions; Harvey rubric judge = Claude Sonnet)

| system | n | pass_rate | pass_rate_fact | pass_rate_analysis | faithful_rate | abstain_rate | evidence_has_gold | avg_evidence_words | avg_llm_calls | avg_seconds | routes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `agent` | 10 | 0.9 | 0.8 | 1.0 | 0.5 | 0.0 | 0.714 | 871.1 | 4.3 | 138.96 | complex:2, simple:8 |

**Unanswerable questions** (`header` index; each question asked in ANOTHER matter's data room; judged by Claude Haiku: did the answer decline?)

| system | n | declined_rate | hallucination_rate | avg_llm_calls |
|---|---|---|---|---|
| `single_shot` | 12 | 0.917 | 0.083 | 1.0 |
| `agent` | 12 | 0.833 | 0.167 | 7.42 |
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
python -m hrag.eval.run_eval generation --n 30 --systems naive single_shot agent --resume  # resumes after a session limit
python -m hrag.eval.stats artifacts/eval/gen_naive_single_shot_agent.jsonl   # paired CIs over matters (no LLM)
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
