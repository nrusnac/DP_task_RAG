# 04 · The agentic module: Adaptive-RAG router + CRAG corrective loop (LangGraph)

## Which RAG step, and why
The agent controls **retrieval**: *whether* to retrieve, *how much*, *with which query*, and *whether the evidence is good enough*. Generation stays a single, well-constrained call.

1. **Retrieval is the dominant failure point.** If the evidence isn't retrieved, no generator can recover it. If wrong evidence is retrieved, a fluent generator turns it into a confident wrong answer. The smoke test did exactly that: given truncated evidence, it answered "1,882" (the document's *national* HHI) instead of the regional 3,552, and two of its top-3 evidence strips came from other matters.
2. **The questions are heterogeneous.** Harvey criteria range from one-fact lookups ("cites post-merger HHI 3,552") to multi-hop synthesis ("explains the remedy strategy across overlapping markets"). Any fixed pipeline is either wasteful on the first kind or insufficient on the second.
3. **The scoring is all-pass, and legal work is zero-tolerance for invented facts.** Missing one fact fails a Harvey task, and fabricating one is worse than saying "not found". So the system must **notice** bad retrieval, **correct** it, and **abstain** when correction fails.
4. **Bounded and auditable.** At most 2 corrective rounds per sub-question and one sufficiency iteration (2 cost more than they added in the dry run). Every decision is logged in the state (`log`) and traced as a Phoenix span.

Generation was *not* made agentic on purpose: once the evidence is right, one constrained Sonnet call with citations is reliable, and agentic generation (self-critique loops) would multiply cost without addressing the main failure mode.

## The graph (`src/hrag/agent/graph.py`)
```
route ─direct─▶ direct_answer ──────────────────────────────────────────▶ END
  │simple ─▶ init_simple ─┐
  └complex ─▶ plan ───────┴▶ retrieve ─▶ grade ─▶ refine ─┬▶ rewrite ─▶ retrieve     (CRAG loop, ≤ 2 rounds)
                                                          ├▶ sufficiency ─┬▶ retrieve (new sub-questions, 1 iteration)
                                                          │               └▶ generate ─▶ END
                                                          └▶ generate ─▶ END
```
State: `question, task_id, route, subqs[{q, query, hyde, rounds, status, best, hits, strips}], iteration, answer, evidence, citations, log`.

### Adaptive-RAG router ([Jeong et al., 2024](https://arxiv.org/abs/2403.14403))
- The paper trains a T5 classifier on automatically derived labels. We have no routing labels and a 4-hour budget, so we use **Claude Haiku with a JSON schema** (`direct | simple | complex` + reason), thinking disabled, ~2 s on Colab.
- **direct:** answers without retrieval and says so.
- **simple:** one retrieval pass over the question itself.
- **complex:** Sonnet decomposes the question, and a sufficiency check may add sub-questions for gaps.
- The `policy` argument forces a route (`agent:simple`, `agent:complex`) for the ablation.

### CRAG ([Yan et al., 2024](https://arxiv.org/abs/2401.15884))
| Verdict | Our condition | Action |
|---|---|---|
| correct | best rerank score ≥ τ_high | knowledge refinement: sentence/row strips scored by the reranker, keep ≥ τ_strip, recompose in document order |
| ambiguous | τ_low ≤ best < τ_high, and Haiku judges ≥ 1 of the top 5 relevant | refine the relevant hits **and** one corrective retrieval; merge |
| incorrect | best < τ_low, or Haiku judges none relevant | rewrite (BM25 keywords + HyDE) and retrieve again; after 2 rounds, mark **insufficient** and abstain on that sub-question |

**Thresholds are measured, not guessed** (`run_eval.py calibrate`). The QA set has only answerable questions, whose top
rerank scores saturate (p10 = 0.89), so on its own it can't calibrate anything. We add **cross-matter negatives**: each
question is also asked against another matter's data room, where its answer cannot exist (median top score 0.017). On
the 60 + 60 calibration queries:
- **τ_low = 0.65:** no answerable question scored below it. It catches 90% of negatives without an LLM call.
- **τ_high = 0.92:** no unanswerable question scored above it. It passes 91% of positives without an LLM call.
- **In between (10.8% of retrievals):** the Haiku grader decides.

That is CRAG's three-way evaluator with measured, error-free boundaries on the calibration set.

**Deviations from the paper, and why:**
- **Evaluator:** our existing cross-encoder replaces the paper's fine-tuned T5, with thresholds calibrated on our eval set. The LLM is called only in the ambiguous band, which keeps cost low.
- **No web-search fallback:** the facts are matter-specific and the data room is confidential, so we abstain instead.
- **Refinement replaces parent-child expansion:** refinement narrows evidence to relevant strips, while parent-child widens it to whole sections. They pull in opposite directions.

## How we show it's worth it ([05-evaluation.md](05-evaluation.md))
- **agent vs. single-shot advanced RAG vs. naive RAG:** Harvey-judge pass rate, faithfulness, abstention, evidence size.
- **adaptive vs. always-simple vs. always-complex:** pass rate against LLM calls and latency. The claim under test is that adaptive approaches always-complex quality at lower cost.
- **CRAG refinement:** evidence words kept vs. the raw top-8 chunks, while the gold chunk is still cited.
