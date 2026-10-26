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
State: `question, task_id, route, subqs[{q, query, hyde, rounds, status, best, hits, chunks}], iteration, answer, evidence, citations, log`.

### Adaptive-RAG router ([Jeong et al., 2024](https://arxiv.org/abs/2403.14403))
- The paper trains a T5 classifier on automatically derived labels. We have no routing labels and a 4-hour budget, so we use **Claude Haiku with a JSON schema** (`direct | simple | complex` + reason), thinking disabled, ~2 s on Colab.
- **direct:** answers without retrieval and says so.
- **simple:** one retrieval pass over the question itself.
- **complex:** Sonnet decomposes the question, and a sufficiency check may add sub-questions for gaps.
- The `policy` argument forces a route (`agent:simple`, `agent:complex`) for the ablation.

### CRAG ([Yan et al., 2024](https://arxiv.org/abs/2401.15884))
| Verdict | Our condition | Action |
|---|---|---|
| correct | best rerank score ≥ τ_high | keep all top-8 hits as **whole chunks**: the same evidence single-shot gets |
| ambiguous | τ_low ≤ best < τ_high, and Haiku judges ≥ 1 of the top 5 relevant | keep the hits Haiku picked (whole) **and** run one corrective retrieval; merge |
| incorrect | best < τ_low, or Haiku judges none relevant | rewrite (BM25 keywords + HyDE) and retrieve again. In the **last** round a score below τ_low still goes to Haiku, so nothing is given up on a score alone; if Haiku keeps nothing, the sub-question is **insufficient** |

**Knowledge refinement works on whole chunks.** Earlier versions cut kept hits into sentence or table-row strips. That
backfired twice. On `main`, the strips cut lists and clauses apart, and the fix was to pass the clearly relevant hits
whole. On this branch's first test run, the strips (plus a cap and a keep filter tied to τ_low) left the agent without
the clause, letter or table it needed in 10 of its 18 failures. Single-shot got those questions right from the same
top hits, passed whole. So the refinement step now decides *which* chunks stay, never which sentences.

**Terminal statuses** (this branch). Only a sub-question graded *correct* in some round ends **done**. One that runs
out of corrective rounds with only grader-kept chunks ends **weak**. It is logged and reported, but generation gets no
warning about it: an early version told the generator that such evidence was "indirect", and it then declined answers
the evidence supported (3 failures on the first test run). One with nothing kept ends **insufficient**, and that
always takes a Haiku grader's "none relevant", never a score alone. When *every* sub-question is insufficient,
`generate` makes no LLM call and returns a fixed "The documents provided do not establish …" answer, so the agent
can't talk itself into an answer from nothing. On `main`, weak sub-questions counted
as done; in both of the agent's unanswerable-question failures, the evidence came from rounds graded ambiguous at top
scores of 0.13–0.84 and was then used as if it answered the question.

**Evidence passed to generation:** up to 8 whole chunks per sub-question, taken round-robin across sub-questions,
up to 16 in all, each chunk once. A simple question graded correct therefore gets exactly single-shot's evidence and
prompt, so on the confident path the agent can't do worse than single-shot; its extra machinery only acts where
retrieval is uncertain (corrective rounds), the question has several parts (decomposition), or the answer isn't
there (abstention).

**Thresholds are measured, not guessed** (`run_eval.py calibrate`). The QA set has only answerable questions, whose top
rerank scores saturate (p10 = 0.89), so on its own it can't calibrate anything. We add **cross-matter negatives**: each
question is also asked against another matter's data room, where its answer cannot exist (median top score 0.017).
On this branch the negatives come from a matter of the **same practice area** (harder look-alikes), and calibration uses
the **dev split only**.

The rule on `main` was error-free on the calibration set: τ_low = the lowest score of any positive, τ_high = just
above the highest negative. In between, the Haiku grader decides. The table shows what it produced:

| Index | Positives p10 | Negatives median | τ_low | τ_high | Ambiguous share (LLM graded) |
|---|---|---|---|---|---|
| `header` (local) | 0.89 | 0.017 | 0.65 | 0.92 | 11% |
| `ctx` (Colab; **what the committed agent loads**) | 0.94 | 0.011 | 0.09 | 0.95 | 20% |

The two classes are almost perfectly separated, but the error-free rule is **sensitive to single outliers**. On `ctx`, one
answerable question topped out below 0.10 and one negative reached 0.95, which widened the band. That cost more than
calls: with τ_low = 0.09, retrievals topping out at 0.13–0.44 were "ambiguous", the grader kept loosely related
passages, and the agent answered two unanswerable questions from them.

**This branch uses a quantile rule** (`run_eval.thresholds`, `CFG.crag_tail = 0.02`): τ_low = the 2nd percentile of the
positives and τ_high = the 98th percentile of the negatives, i.e. each threshold may misclassify at most 2% of its own
class, so one outlier can't move it. On the `main` calibration scores this gives τ_low = 0.59 and τ_high = 0.91 (6%
ambiguous); both unanswerable-question failures above would have been graded incorrect or weak. Answerable questions
below τ_low still get the two corrective rewrites, and the Haiku grader's look in the last round, before the agent
abstains. That last look was added after the first test run: three answerable questions there scored 0.47–0.71 in
every round (the reranker scores the same top chunk against the same sub-question, so rewrites that return it can't
change the score) and were abstained on without any LLM reading the passages.

**Deviations from the paper, and why:**
- **Evaluator:** our existing cross-encoder replaces the paper's fine-tuned T5, with thresholds calibrated on our eval set. The LLM is called only in the ambiguous band and before giving up in the last round, which keeps cost low.
- **No web-search fallback:** the facts are matter-specific and the data room is confidential, so we abstain instead.
- **Refinement at chunk level, not the paper's decompose-then-recompose strips:** see "Knowledge refinement works on whole chunks" above. Parent-child expansion (widening to whole sections) remains a possible next step.

## How we show it's worth it ([05-evaluation.md](05-evaluation.md))
- **agent vs. single-shot advanced RAG** (and naive RAG on `main` and the first test run): Harvey-judge pass rate, faithfulness, abstention, evidence size.
- **adaptive vs. always-simple vs. always-complex:** pass rate against LLM calls and latency. The claim under test is that adaptive approaches always-complex quality at lower cost.
- **CRAG refinement:** whether the evidence passed to generation still contains the gold chunk (`evidence_has_gold`), next to single-shot's.
