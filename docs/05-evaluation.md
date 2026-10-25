# 05 · Evaluation

## The problem
Harvey LAB grades *agent deliverables* (memos) with rubric criteria, and has **no retrieval gold labels**: `criteria[].sources` is empty in every task. So we build a retrieval + QA eval from the rubric itself.

## Pipeline (`src/hrag/eval/`)
```mermaid
flowchart LR
    T["task.json criteria"] --> QA["C1 QA generation<br/>Claude Sonnet: question + key facts"]
    QA --> SL["C2 silver labels<br/>key facts ∈ chunk text"]
    SL --> RE["C3 retrieval eval<br/>Recall@k, MRR, nDCG"]
    QA --> GE["C4 generation eval<br/>Harvey judge + faithfulness"]
    GE --> AG["C5 agent metrics<br/>routes, calls, latency"]
```

**Splits.** `qa.jsonl` has a `split` field:
- **`dev`** (110 questions, 14 antitrust matters): every question used on `main`. They shaped the design: 30 tuned prompts and the refinement fix, 57 labelled ones set the CRAG thresholds and the retrieval choices.
- **`test`** (126 questions, 16 matters): the 4 antitrust matters with no questions on `main` plus the 12 non-drafting deal-regulatory matters added to the corpus ([01](01-dataset.md)). Generated after the branch's code changes, and used by nothing that tunes the system. A **matter-level** held-out set: no test matter appears in dev. (One caveat: the demo cells use one test matter, `analyze-iss-antitrust-transaction-structure`, so its documents were seen during development, but none of its rubric criteria were.)

**C1 · QA generation** (`build_qa.py`): `build` sampled 14 non-drafting tasks × 8 criteria (dev); `extend` adds 8 criteria for every non-drafting matter without questions (test), keeping existing questions untouched. One Sonnet call per task rewrites each criterion as:
- the *question a lawyer would ask the data room*, without revealing the answer
- an expected answer
- 0–4 **verbatim key facts** ("3,552", "VelaroConnect", "Section 7A")
- a kind: `fact` / `analysis` / `skip`

Drafting tasks are excluded, and `skip` criteria are dropped, because those criteria grade the lawyer's own deliverable (format, a proposal it should make) rather than what the documents say.

**C2 · Silver labels:** a chunk of the *same matter* is gold if it contains the key facts (all of 1–2 facts, otherwise ≥ 2/3 of them; normalised, with "1,242" ≡ "1242"). *Silver, not gold:* cheap, objective and reproducible; good for **comparing configurations**, not for absolute claims.
- When the facts match **more than 8 chunks** (too generic on their own; 25 dev questions on `main` were left unlabelled for this), one Sonnet call sees the question, expected answer and up to 30 of those matching chunks and picks the ones that state the answer (`gold_source: "llm"`, at most 8). The candidates come from string matching, never from our retriever, so the labels don't favour it.
- Result: dev 82/110 labelled (57 string + 25 LLM; was 57/110), test 98/126 (77 + 21). The rest have no verbatim facts (analysis questions) or facts that match nothing. `build_qa relabel --split dev` recomputes labels without regenerating questions.

**C3 · Retrieval** (`run_eval.py retrieval`, no LLM cost), **per split**: every combination of index variant `raw` / `header` / `ctx` × `bm25` / `dense` / `hybrid` / `hybrid+rerank` (reranker order replaces the hybrid order) / `hybrid+rerank(rrf)` (rank fusion of both, the default on this branch), scored by:
- **Recall@k** (hit rate)
- **MRR**
- **nDCG@10**

**CRAG calibration** (`run_eval.py calibrate`, **dev split only**): the top rerank score for answerable questions (positives) vs. the same questions asked in another matter's data room of the **same practice area** (cross-matter negatives). τ_low = 2nd percentile of the positives, τ_high = 98th percentile of the negatives (`CFG.crag_tail`); the band in between is LLM-graded. On `main` the rule was min/max (error-free), which one outlier per side could move ([04](04-agentic-module.md#crag-yan-et-al-2024)).

**Unanswerable questions** (`run_eval.py negatives`, test split, 40 questions): cross-matter negatives (same practice area) run end to end through single-shot and the agent. A Haiku judge checks whether each answer declined instead of answering from look-alike evidence.

**C4 · Generation** (`run_eval.py generation`, default: the whole test split): systems compared on the same questions:
- `naive`: dense-only, raw-text index, top-8 chunks, one generation call
- `single_shot`: hybrid + rerank on the contextual index, top-8 chunks, one call (advanced but fixed)
- `agent`: the LangGraph Adaptive-RAG + CRAG module
- `agent:simple` / `agent:complex`: forced routes, for the Adaptive-RAG ablation

Metrics:
- **Harvey-judge pass rate:** Harvey's own `rubric_criterion.txt` prompt against the *original* `match_criteria`, split by fact and analysis
- **faithfulness:** Haiku checks every claim against the evidence
- **abstention rate**
- **evidence-has-gold:** retrieval recall inside the full system
- **evidence words:** what CRAG refinement removes
- **LLM calls and latency per question**
- **route distribution**

Every eval item is a traced span (`eval.item`), so failures can be opened in Phoenix.

**Long runs and the subscription limit.** The test eval is ~1,400 Claude calls (5c) plus ~800 (5d, 5e), more than one
subscription session. A session-limit error (`llm.QuotaError`) is raised at once instead of being retried, the run
stops scheduling new questions, and the unfinished ones are not recorded. `resume=True` (`--resume`) keeps every
finished answer and runs only the missing (question, system) pairs, so the notebook cell is simply re-run after the
reset. On `main`, such errors were retried, recorded as failures, and the run continued: 18 of 24 runs of one test
were lost that way.

## Unfaithful-answer patterns
The agent's 15 unfaithful answers in the final `main` run (faithfulness 0.50 vs. single-shot 0.633 and naive 0.70) fall
into six patterns (some answers show several). Each became a rule in the shared generation prompt on this branch
([03](03-rag-techniques.md#9-grounded-generation)):

| Pattern | Questions |
|---|---|
| exclusivity or completeness the evidence doesn't state ("the only …") | C-019, C-045, C-054, C-006, C-060 |
| inferences or characterizations not labelled "Analysis:" | C-050, C-054 |
| headings or bottom lines that add a claim | C-022, C-074 |
| a proposed redline presented as adopted terms | C-005 |
| "signed" when the signature line was blank | C-020 |
| a fact attributed to a source or party that doesn't state it | C-037, C-046, C-004 |

Within the agent, unfaithful answers came with ~1.7× the evidence of faithful ones (13.5 vs 9.1 items), which is why
the evidence passed to generation is now capped ([04](04-agentic-module.md)).

## Results
Run on a Colab T4 with `notebooks/02_demo_and_eval.ipynb`. The JSON outputs are in `artifacts/eval/`.

> **Branch `agent-improvements`:** the tables below are the `main` results (dev questions, old corpus, Qwen2.5-3B
> contexts). They will be replaced by the rerun on this branch (test split, 44-matter corpus, Qwen3.5-4B contexts).

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

## Analysis (honest reading of the numbers)

**Retrieval (57 labelled questions; one question = 1.75 points, so small gaps are noise)**
- **Hybrid beats either single retriever.** On the `raw` index: R@1 0.684 vs. 0.614 / 0.614 (BM25 / dense), MRR 0.784 vs. 0.745 / 0.728. The best R@10 is `header/hybrid` (0.982).
- **Reranking trades recall@5–10 for precision@1.** On `header`, R@1 rises 0.667 → 0.702 while R@5 falls 0.912 → 0.860. With silver labels, the reranker sometimes prefers a chunk that answers the question semantically but lacks the verbatim facts.
- **Contextual Retrieval did not measurably help here** (`ctx/hybrid` R@5 0.877 vs. `header/hybrid` 0.912). Likely reasons:
  1. The labels are verbatim-fact matches, and generic context words dilute BM25 (`ctx/bm25` 0.877 vs. `header/bm25` 0.895).
  2. Within one matter (~186 chunks) there is little ambiguity left for context to resolve, and the cheap deterministic header already carries the document and section.
  3. A 3B model's contexts are often generic ("Lists the parties' signatures and the date of the agreement").

  The technique is built to disambiguate across a large corpus, and corpus-wide it only ties (R@1 0.684 = `header`).
- **Scoping matters.** Corpus-wide dense retrieval puts **~17% of top-5 hits from other matters** (`foreign@5` 0.165–0.172). Hybrid + rerank cuts that to ~7%, and scoping to the data room removes it entirely.

**CRAG calibration on the `ctx` index** gave τ_high 0.95 / τ_low 0.09. The error-free-band rule is sensitive to single outliers (one negative scored 0.95, one positive 0.09), so the ambiguous band was wide and Haiku graded more often (90 `crag_grade` calls). A percentile rule (e.g. 5th/95th) would be more robust; that's the next step.

**Generation, first Colab run (before the fix; 30 questions, `ctx` index)**
- Pass rates: naive **0.90**, single-shot **0.967**, agent **0.867**.
- Paired (agent vs. single-shot): 26 both pass, **3 fail only for the agent**, 1 both fail (McNemar p = 0.25).
- **Root cause of 3 of the 4 agent failures:** the gold chunk *was* retrieved and CRAG graded it `correct` (0.98–1.00). Then **knowledge refinement pruned it to 73–475 words**. (The 4th, C-100, was pruned too, but its deeper cause is a chunking miss shared by all systems; see below.) These were list, enumeration and clause questions ("each change the DOJ made", the carve-out text), where individual sentences don't look relevant on their own. This is the cost of choosing refinement over parent-child ([07](07-decision-log.md) #7).
- **Fix:** hits scored ≥ τ_high (clearly relevant) are passed **whole**; only marginal hits are strip-filtered. A targeted local check (`header` index, the 4 failed + 6 passed questions) predicted that 3 of the 4 would flip with no regressions.

**Generation, final Colab run (after the fix; same 30 questions, `ctx` index)**
- Pass rates: **agent 0.967 (29/30)**, single-shot 0.933 (28/30), naive 0.90 (27/30).
- **The fix did exactly what the diagnosis predicted.** The 3 over-pruned questions (C-016, C-005, C-046) flipped from fail to pass, and none of the agent's other passes regressed.
- Paired (agent vs. single-shot): 28 both pass, 1 passes only for the agent, 1 both fail (McNemar p = 1.0). **Statistically a tie.**
- **Run-to-run noise, measured:** with unchanged code, single-shot flipped one question (C-046, pass → fail), and the faithfulness of naive / single-shot moved by about ±7 points (0.633 → 0.70 and 0.667 → 0.633). So one question of pass rate, or a few points of faithfulness, is within noise at n = 30.
- **The remaining failure (C-100, "list all 25 custodians") fails for all three systems, and it is a chunking issue.**
  - The 25 names *are* in the data room, in three appendix tables (Appendix A/B/C).
  - Each table became its own chunk, starting with `| No. | Full Name | Title |…` under the section path "APPENDIX A". The word "custodians" lives only in the caption paragraph just before each table, which went into a separate chunk.
  - The tables rank 20th–30th or lower, outside the top-8. The LLM context lines ("Appendix A – Wave 1 Custodians") didn't lift them enough.
  - **Next step:** attach a table's caption to the table chunk, and for "list all" questions retrieve the whole section.
- **What the fix cost:**
  - Evidence per answer rose from 336 to **1,201 words** (still **−46%** vs. single-shot's 2,227).
  - The agent's faithfulness fell from 0.70 to **0.50**, more than the measured noise. With whole chunks the generator writes longer answers with more unlabelled characterisations (e.g. "It is a legend, not an access control"). Correctness was bought with some faithfulness. A claim-verification node, or stricter "Analysis:" labelling, is the next step.
  - **3.7 LLM calls and 28 s** per question, vs. 1 call and 13 s.

**Adaptive-RAG**
- **Final run (20 questions): all three policies score 0.90.** Adaptive costs **3.45 calls / 35 s**, vs. always-complex 6.6 / 53 s and always-simple 1.45 / 19 s.
- In the first run adaptive led 0.90 vs. 0.85 / 0.85. That one-question edge did **not** reproduce, so it was noise.
- On this set the router sends ~80% of questions down the simple path. **Always-simple is the cheapest at equal accuracy.** Adaptive halves the cost of always-complex, but its accuracy value would need genuinely multi-hop questions to show.

**Unanswerable questions** (12 questions, each asked in another matter's data room; `header` index, local):
- Single-shot declined **11/12**; the agent declined **10/12** at 7.4 vs. 1 LLM calls.
- The difference is one question (noise), but the hypothesis that the agent abstains better is **not supported**. Two reasons from the traces:
  1. The **shared grounded generator prompt** ("say 'the documents do not establish …'") already makes Sonnet decline when evidence is unrelated, in both systems.
  2. **CRAG's corrective loop cuts both ways.** Rewrite + HyDE *searches harder* and can surface look-alike evidence from the wrong data room (e.g. another matter's IT memo on employees' personal devices), which the generator then answers from.

  A stricter grader on corrective rounds, or abstaining when only rewritten queries find evidence, would be the next step.
- The `ctx` rerun of this test **hit the Claude subscription's session limit** ("You've hit your session limit"): 18 of 24 runs errored. It is excluded from the tables, and the raw records are kept as `unanswerable_ctx_INCOMPLETE_session_limit.jsonl`. Operational lesson: at ~700 calls per full eval session, subscription quota is a real constraint, which is what the API-key fallback is for.

**Conclusion**
- After a fix that the per-node traces pointed to, **the agent is the best-scoring system (29/30 vs. 28/30 vs. 27/30)**, but the gaps are within measured run-to-run noise. The honest claim is **"at least on par"**, not "better".
- **What the agentic layer adds:**
  - **Efficiency:** 46% less context than single-shot (85% before the fix); adaptive routing at half the cost of always-complex.
  - **Auditability:** every route, CRAG verdict and rewrite is logged and traced.
  - **A diagnosable structure:** each failure traced to one node (refinement over-pruning; corrective search finding look-alikes; table/caption chunking). The over-pruning fix was predicted from the traces and confirmed question by question on Colab.
- **What it costs:** ~3.7× the LLM calls, ~2× the latency, and in its current form lower faithfulness.
- **The accuracy foundation is the retrieval stack** (structure-aware chunks, hybrid + RRF, rerank, data-room scoping): even naive dense RAG scores 0.90 with this generator.

## Findings that changed the design
- **The silver labels favour BM25 by construction.** A chunk is gold *because* it contains the verbatim key facts, which is exactly what BM25 matches. So BM25's strong showing within a matter is partly a labelling artefact. Hybrid search is the safer default: it never loses much to the best single retriever and wins on Recall@5.
- **Searching only within the matter makes the task easy** (~186 chunks per matter). Corpus-wide rows show the harder setting and the **contamination** (`foreign@5`: share of top-5 hits from *other* matters) that scoping removes.
- **Reranker scores saturate on answerable questions** (52/60 top scores ≥ 0.95 in the first calibration), so a reranker-only CRAG evaluator would call nearly everything "correct". Calibration therefore adds **cross-matter negatives**: every question is also asked against another matter's data room, where its answer can't exist. τ_high and τ_low are chosen from positives vs. negatives, and the agent loads them from `artifacts/eval/crag_calibration.json`. The band in between goes to the Haiku grader.

## Limitations
- Silver labels can miss gold chunks that paraphrase a fact ("fifty percent") and can credit chunks that mention a figure in passing. Fine for relative comparisons; absolute recall is approximate.
- One judge model (Claude), not Harvey's dual-judge profile (Claude + GPT), and our answers are short QA answers rather than full memos. Criteria written for memos are judged "in substance" for the question asked.
- The sample is small (tens of questions), so differences of a few points are noise.
- On `main`, the CRAG thresholds were calibrated on the same questions used in the generation eval, and the refinement fix was confirmed on the same 30 questions that exposed it. **This branch separates them:** calibration and every tuning decision use dev; the test split (16 matters never used) is only evaluated.
- The prompt rules on this branch were written from the dev answers' failures; the test split is where they can be judged fairly.
- **Run-to-run noise** (measured on unchanged pipelines): about ±1 question of pass rate and about ±7 points of faithfulness at n = 30.
