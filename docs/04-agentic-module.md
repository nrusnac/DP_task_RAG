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
  └complex ─▶ plan ───────┴▶ retrieve ─▶ grade ─▶ refine ─┬▶ rewrite ─▶ retrieve     (CRAG loop, 1 round)
                                                          ├▶ sufficiency ─┬▶ retrieve (new sub-questions, 1 iteration)
                                                          │               └▶ generate ─▶ END
                                                          └▶ generate ─▶ END
```
State: `question, task_id, route, subqs[{q, query, hyde, rounds, status, best, hits, chunks, seen}], iteration, answer, evidence, citations, log` (+ `reuse`/`reused`, an eval-only hook, see [05](05-evaluation.md#eval-v3-an-eval-that-can-tell-the-systems-apart)).
On both routes sub-question 0 is the question itself.

### Adaptive-RAG router ([Jeong et al., 2024](https://arxiv.org/abs/2403.14403))
- The paper trains a T5 classifier on automatically derived labels. We have no routing labels and a 4-hour budget, so we use **Claude Haiku with a JSON schema** (`direct | simple | complex` + reason), thinking disabled, ~2 s on Colab.
- **direct:** answers without retrieval and says so.
- **simple:** one retrieval pass over the question itself.
- **complex:** the question itself is searched first; Sonnet then decomposes it, seeing what that search found, and a Sonnet sufficiency check may add sub-questions for gaps and for unstated links between entities.
- The router also sends questions that ask how two parties or facts relate, or that need something identified first ("X's former employer"), down the complex route.
- The `policy` argument forces a route (`agent:simple`, `agent:complex`) for the ablation.

### CRAG ([Yan et al., 2024](https://arxiv.org/abs/2401.15884))
| Verdict | Our condition | Action |
|---|---|---|
| correct | best rerank score ≥ τ_high | keep all top-8 hits as **whole chunks**: the same evidence single-shot gets |
| ambiguous | τ_low ≤ best < τ_high, and Haiku judges ≥ 1 of the top 5 relevant | keep **all** top-8 hits (whole: the grader is a gate, not a filter) **and** run one corrective retrieval that searches past the chunks already judged or kept; add only the new chunks Haiku accepts |
| incorrect | best < τ_low, or Haiku judges none relevant | rewrite (BM25 keywords + HyDE) and retrieve again, once. In that **last** round a score below τ_low still goes to Haiku, so nothing is given up on a score alone; if Haiku keeps nothing, the sub-question is **insufficient** |

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

**Evidence passed to generation:** sub-question 0 (the question itself) first, up to its top 8 whole chunks; then up
to 8 more chunks per sub-question, round-robin, up to 16 in all, each chunk once. Whenever the question's own search is
usable the agent's evidence therefore contains single-shot's, and a simple question graded correct gets exactly
single-shot's evidence and prompt. Its extra machinery only adds: where retrieval is uncertain (corrective round), the
question has several parts or links (decomposition, sufficiency), or the answer isn't there (abstention).

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
| `ctx` (Colab; retrieval ablation only on this branch) | 0.94 | 0.011 | 0.09 | 0.95 | 20% |

The two classes are almost perfectly separated, but the error-free rule is **sensitive to single outliers**. On `ctx`, one
answerable question topped out below 0.10 and one negative reached 0.95, which widened the band. That cost more than
calls: with τ_low = 0.09, retrievals topping out at 0.13–0.44 were "ambiguous", the grader kept loosely related
passages, and the agent answered two unanswerable questions from them.

**This branch uses a quantile rule** (`run_eval.thresholds`, `CFG.crag_tail = 0.02`): τ_low = the 2nd percentile of the
positives and τ_high = the 98th percentile of the negatives, i.e. each threshold may misclassify at most 2% of its own
class, so one outlier can't move it. On the `main` calibration scores this gives τ_low = 0.59 and τ_high = 0.91 (6%
ambiguous); both unanswerable-question failures above would have been graded incorrect or weak. Answerable questions
below τ_low still get a corrective rewrite, and the Haiku grader's look in the last round, before the agent
abstains. That last look was added after the first test run: three answerable questions there scored 0.47–0.71 in
every round (the reranker scores the same top chunk against the same sub-question, so rewrites that return it can't
change the score) and were abstained on without any LLM reading the passages.

The public agent and evaluation default to `header`. A saved calibration is loaded only when its recorded variant
matches the active index; `HRAG_CRAG_HIGH` and `HRAG_CRAG_LOW` override the two thresholds independently.

**Deviations from the paper, and why:**
- **Evaluator:** our existing cross-encoder replaces the paper's fine-tuned T5, with thresholds calibrated on our eval set. The LLM is called only in the ambiguous band and before giving up in the last round, which keeps cost low.
- **No web-search fallback:** the facts are matter-specific and the data room is confidential, so we abstain instead.
- **Refinement at chunk level, not the paper's decompose-then-recompose strips:** see "Knowledge refinement works on whole chunks" above. Parent-child expansion (widening to whole sections) remains a possible next step.

## Changes after the second test run
The second test run tied the agent with single-shot. Reading its records ([05](05-evaluation.md#second-test-run-why-the-eval-could-not-separate-the-systems))
showed four agent weaknesses, fixed in commit `7ae4f69` (the agent was frozen there before the hard sets were built):

| Problem on `test_v2` | Fix |
|---|---|
| The planner wrote generic sub-questions (five textbook Daubert questions for "which issues are highest priority") and the question itself was never searched; single-shot found the answer by searching it | **The question is always search 0** and keeps single-shot's top 8 at the head of the evidence; **the planner sees that search's results** (doc labels + first 200 characters) and writes 1–4 sub-questions for the parts it misses, in the matter's own names and terms. No extra call |
| A link the answer turned on (a nominee's former employer is a subsidiary of a fund investor) was stated in the data room but never searched: the plan was written before any evidence, and Haiku called the evidence sufficient | **Sufficiency on Sonnet** with ~1,200 characters per chunk, asked also for **unstated links**: an entity named in one passage whose relation to another is needed, searched with both names verbatim. The router sends relation / identify-first questions down the complex route. On that question the agent now asks the link question, retrieves the subsidiary passage and answers correctly in 4 calls (it motivated the fix, so it is no evidence of improvement) |
| 33 retried sub-questions, none improved; 27 had the identical best score in all three rounds | **One corrective round that searches past the chunks already judged or kept**, so the grader sees new candidates |
| `ambiguous` kept only the grader's picks: less evidence than single-shot | In round 0 the grader is a **gate**: when anything is relevant, all top hits stay; a retry adds only what the grader accepts |

## How we show it's worth it ([05-evaluation.md](05-evaluation.md))
- **agent vs. single-shot advanced RAG** (and naive RAG on `main` and the first test run): Harvey-judge pass rate, claim-level faithfulness, pairwise preference, abstention, evidence size, with matter-clustered inference; on the rubric questions (a regression check) and on strict post-freeze **hard sets** (two-step links and three-document questions), where multi-step retrieval can matter.
- **adaptive vs. always-simple vs. always-complex:** pass rate against LLM calls and latency. The claim under test is that adaptive approaches always-complex quality at lower cost.
- **CRAG refinement:** whether the evidence passed to generation still contains the gold chunk (`evidence_has_gold`), next to single-shot's.
