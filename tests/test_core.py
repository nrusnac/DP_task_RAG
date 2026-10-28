"""Fast unit tests (no GPU, no LLM). Run: pytest -q"""
from pathlib import Path

import pandas as pd
import pytest

from hrag.agent import graph
from hrag.config import CFG
from hrag.eval.build_qa import silver_labels
from hrag.eval.metrics import mrr, ndcg_at, recall_at
from hrag.ingest.chunking import chunk_document
from hrag.ingest.contextualize import clean_context
from hrag.retrieve.hybrid import HybridRetriever

AREA = CFG.area_dir


def test_rrf_rewards_agreement():
    fused = dict(HybridRetriever.rrf([[1, 2, 3], [3, 1, 4]], k=60))
    assert max(fused, key=fused.get) in (1, 3)          # ranked high by both lists
    assert fused[1] > fused[2] and fused[3] > fused[4]  # agreement beats a single high rank
    assert abs(fused[1] - (1 / 61 + 1 / 62)) < 1e-12


def test_metrics():
    ranked, gold = ["a", "b", "c"], {"b"}
    assert recall_at(ranked, gold, 1) == 0 and recall_at(ranked, gold, 2) == 1
    assert mrr(ranked, gold) == 0.5
    assert 0 < ndcg_at(ranked, gold, 10) < 1


def test_table_chunks_repeat_header():
    rows = [["Market", "HHI"]] + [[f"m{i}", str(1000 + i)] for i in range(400)]
    chunks = chunk_document("t", "d.xlsx", "xlsx", "Doc", [{"kind": "table", "rows": rows}], max_tok=120)
    assert len(chunks) > 1
    assert all(c.text.startswith("| Market | HHI |") for c in chunks)


def test_sections_and_tiny_merge():
    els = [{"kind": "heading", "level": 1, "text": "Article I"},
           {"kind": "heading", "level": 2, "text": "Section 1.1 Definitions"},
           {"kind": "para", "text": "Short."},
           {"kind": "heading", "level": 2, "text": "Section 1.2 Price"},
           {"kind": "para", "text": "The purchase price is $10 million. " * 5}]
    chunks = chunk_document("t", "d.docx", "docx", "SPA", els)
    assert len(chunks) == 1  # tiny section merged forward
    assert chunks[0].section.startswith("Article I › Section 1.1")
    assert "Section 1.2 Price" in chunks[0].text


def test_clean_context():
    assert clean_context("Context: <SEBPA invites Ridgeline>") == "SEBPA invites Ridgeline"


def test_silver_labels_number_variants():
    df = pd.DataFrame(dict(chunk_id=["a", "b"], header=["h", "h"],
                           text=["Post-Merger HHI | 3,552 | delta 1,242", "unrelated"]))
    assert silver_labels({"key_facts": ["3,552", "1242"]}, df) == ["a"]
    assert silver_labels({"key_facts": []}, df) == []


def test_crag_edges():
    s = {"route": "complex", "iteration": 0, "subqs": [{"status": "retry"}]}
    assert graph.after_refine(s) == "rewrite"
    s["subqs"] = [{"status": "done"}]
    assert graph.after_refine(s) == "sufficiency"
    s["iteration"] = CFG.max_sufficiency_iters
    assert graph.after_refine(s) == "generate"
    assert graph.after_refine({"route": "simple", "subqs": [{"status": "insufficient"}]}) == "generate"
    assert graph.after_route({"route": "direct"}) == "direct_answer"


@pytest.mark.skipif(not AREA.exists(), reason="dataset not downloaded")
def test_docx_tracked_changes_and_malformed_xml():
    from hrag.ingest.parsers import parse
    redline = next(AREA.glob("*/documents/doj-redline-markup.docx"))
    _, els = parse(redline)
    text = " ".join(e.get("text", "") for e in els)
    assert "[INSERTED:" in text and "[DELETED:" in text
    broken = next(AREA.glob("*/documents/jv-business-plan-final.docx"))
    _, els = parse(broken)  # unescaped '&' in the XML must not crash
    assert len(els) > 10


def test_llm_counter_isolated_per_task():
    import asyncio
    from hrag import llm

    async def fake_run(n):
        ctr = llm.start_counter()
        for _ in range(n):
            c = llm._COUNTER.get(); c["calls"] += 1
            await asyncio.sleep(0)
        return ctr["calls"]

    async def main():
        return await asyncio.gather(fake_run(1), fake_run(3), fake_run(2))

    assert asyncio.run(main()) == [1, 3, 2]


# ------------------------------------------------------------------ agent-improvements branch
def test_fuse_rerank_orders_by_both_ranks_and_keeps_rerank_scores():
    from hrag.retrieve.hybrid import Hit
    cands = [Hit(i, f"c{i}", 0.0, "") for i in range(4)]          # first-stage order c0..c3
    out = HybridRetriever.fuse_rerank(cands, [0.1, 0.9, 0.8, 0.2])  # reranker order c1, c2, c3, c0
    assert [h.chunk_id for h in out] == ["c1", "c0", "c2", "c3"]
    assert {h.chunk_id: h.score for h in out} == {"c0": 0.1, "c1": 0.9, "c2": 0.8, "c3": 0.2}


def test_crag_thresholds_ignore_single_outliers():
    import numpy as np
    from hrag.eval.run_eval import thresholds
    pos = np.r_[0.05, np.linspace(0.6, 1.0, 99)]   # one answerable question retrieved terribly
    neg = np.r_[np.linspace(0.0, 0.7, 99), 0.99]   # one unanswerable question that looks answerable
    assert thresholds(pos, neg, tail=0.02) == (0.60, 0.70)
    lo, hi = thresholds(np.array([0.9, 0.95]), np.array([0.1, 0.2]), tail=0.02)
    assert lo == hi  # no overlap: the ambiguous band collapses


def _chunk(i):
    return dict(chunk_id=i, doc="d", text=f"text {i}", score=0.9)


def test_select_evidence_question_first_then_round_robin():
    subqs = [dict(chunks=[_chunk("A"), _chunk("B"), _chunk("C")]), dict(chunks=[_chunk("A"), _chunk("D")])]
    ev = graph._select_evidence(subqs, per_subq=8, max_total=16)
    assert [e["chunk_id"] for e in ev] == ["A", "B", "C", "D"]  # A once; grouped by sub-question
    ev = graph._select_evidence(subqs, per_subq=8, max_total=3)
    assert [e["chunk_id"] for e in ev] == ["A", "B", "C"]       # the question's own top chunks come first
    many = [dict(chunks=[_chunk(f"q{i}") for i in range(10)]), dict(chunks=[_chunk("K"), _chunk("L")]),
            dict(chunks=[_chunk("M")])]
    ev = [e["chunk_id"] for e in graph._select_evidence(many, per_subq=8, max_total=12, lead=8)]
    # 8 lead chunks, then round-robin over the question's later chunks (a retry), K/L and M
    assert ev == [f"q{i}" for i in range(10)] + ["K", "M"]


def _hit(i, score=0.9):
    from hrag.retrieve.hybrid import Hit
    return Hit(i, f"c{i}", score, "", dict(doc_id="d", section="s", chunk_text=f"text {i}"))


def test_grader_is_a_gate_in_round_0_and_a_filter_on_retry(monkeypatch):
    import asyncio
    from hrag import llm

    async def fake_complete(prompt, **kw):
        return {"relevant": [2]}

    monkeypatch.setattr(llm, "complete", fake_complete)
    hits = [_hit(i) for i in range(8)]
    mid = (CFG.crag_low + CFG.crag_high) / 2
    sq = lambda rounds: dict(id=0, q="q", status="retrieved", rounds=rounds, best=mid, hits=hits, chunks=[],  # noqa: E731
                             seen=[])
    out = asyncio.run(graph.grade({"subqs": [sq(0)], "log": []}))["subqs"][0]
    assert out["status"] == "ambiguous" and len(out["keep"]) == 8         # something relevant: keep all top hits
    assert out["seen"] == [f"c{i}" for i in range(5)]                      # the 5 passages the grader judged
    out = asyncio.run(graph.grade({"subqs": [sq(1)], "log": []}))["subqs"][0]
    assert [h.chunk_id for h in out["keep"]] == ["c1"]                     # a retry adds only what the grader accepts


def test_rewrite_round_searches_past_judged_chunks_only(monkeypatch):
    import asyncio
    calls = []

    class FakeRetriever:
        def search(self, query, **kw):
            calls.append(kw.get("exclude"))
            return [_hit(9)]

    monkeypatch.setattr(graph, "retriever", lambda variant="ctx": FakeRetriever())
    low = CFG.crag_low / 2
    sq = dict(id=0, q="q", query="q", hyde=None, status="retrieved", rounds=0, best=low, hits=[_hit(0, low)],
              chunks=[], seen=[])
    out = asyncio.run(graph.grade({"subqs": [sq], "log": []}))["subqs"][0]
    assert out["status"] == "incorrect" and out["seen"] == []              # score-only reject: not "seen"
    st = {"subqs": [dict(sq, status="pending", seen=["c0", "c3"])]}
    asyncio.run(graph.retrieve(st))
    assert calls == [{"c0", "c3"}]


def test_plan_searches_the_question_first(monkeypatch):
    import asyncio
    from hrag import llm
    prompts = []

    class FakeRetriever:
        def search(self, query, **kw):
            return [_hit(0, 0.95), _hit(1, 0.5)]

    async def fake_complete(prompt, **kw):
        prompts.append(prompt)
        return {"subquestions": ["Who signed the side letter?", "the question"]}

    monkeypatch.setattr(graph, "retriever", lambda variant="ctx": FakeRetriever())
    monkeypatch.setattr(llm, "complete", fake_complete)
    out = asyncio.run(graph.plan({"question": "The question", "matter": "m", "task_id": "t", "log": []}))
    s0, s1 = out["subqs"]                                                  # a copy of the question is dropped
    assert s0["q"] == "The question" and s0["status"] == "retrieved" and s0["best"] == 0.95
    assert s1["id"] == 1 and s1["status"] == "pending"
    assert "d § s: text 0" in prompts[0]                                   # the planner sees the first search


def test_generate_reuses_single_shot_answer_only_for_identical_evidence(monkeypatch):
    import asyncio
    from hrag import llm
    calls = []

    async def fake_complete(prompt, **kw):
        calls.append(prompt)
        return "fresh answer [1]"

    monkeypatch.setattr(llm, "complete", fake_complete)
    st = {"question": "q", "matter": "m", "log": [],
          "subqs": [dict(q="q", status="done", chunks=[_chunk("A"), _chunk("B")])]}
    out = asyncio.run(graph.generate(dict(st, reuse={"evidence_ids": ["A", "B"], "answer": "ss answer [2]"})))
    assert out["reused"] and out["answer"] == "ss answer [2]" and not calls
    out = asyncio.run(graph.generate(dict(st, reuse={"evidence_ids": ["B", "A"], "answer": "ss answer [2]"})))
    assert not out["reused"] and out["answer"] == "fresh answer [1]" and len(calls) == 1


def test_terminal_statuses_and_forced_abstain():
    import asyncio
    from hrag.eval.run_eval import abstained
    r = CFG.max_corrective_rounds
    kept = [{"chunk_id": "x", "text": "t"}]
    st = {"subqs": [dict(q="a", status="ambiguous", rounds=r, chunks=kept),
                    dict(q="b", status="incorrect", rounds=r, chunks=kept),
                    dict(q="c", status="incorrect", rounds=r, chunks=[])]}
    out = asyncio.run(graph.refine(st))  # no "keep": nothing new to add
    assert [s["status"] for s in out["subqs"]] == ["weak", "weak", "insufficient"]
    st = {"question": "q", "matter": "m", "log": [], "subqs": [dict(q="c", status="insufficient", chunks=[])]}
    out = asyncio.run(graph.generate(st))  # all insufficient: answered without an LLM call
    assert out["evidence"] == [] and abstained(out["answer"])


def test_low_score_goes_to_the_llm_grader_in_the_last_round(monkeypatch):
    import asyncio
    from hrag import llm
    from hrag.retrieve.hybrid import Hit
    graded = []

    async def fake_complete(prompt, **kw):
        graded.append(prompt)
        return {"relevant": [1]}

    monkeypatch.setattr(llm, "complete", fake_complete)
    hits = [Hit(0, "c0", 0.3, "", dict(doc_id="d", section="s", chunk_text="the answer"))]
    sq = lambda rounds: dict(id=0, q="q", status="retrieved", rounds=rounds, best=0.3, hits=hits, chunks=[])  # noqa: E731
    out = asyncio.run(graph.grade({"subqs": [sq(0)], "log": []}))
    assert out["subqs"][0]["status"] == "incorrect" and not graded       # early round: rewrite, no LLM call
    out = asyncio.run(graph.grade({"subqs": [sq(CFG.max_corrective_rounds)], "log": []}))
    assert out["subqs"][0]["status"] == "ambiguous" and len(graded) == 1  # last round: the grader decides


def test_abstained_needs_a_decline_without_citations():
    from hrag.eval.run_eval import abstained, hedged
    assert abstained("The documents provided do not establish the fee.")
    assert not abstained("Short answer: the documents provided do not establish X. They show Y [2].")
    assert hedged("They show Y [2]. The documents provided do not establish X.")


def test_quota_errors_fail_fast(monkeypatch):
    import asyncio
    from hrag import llm
    msg = "Claude Code returned an error result: You've hit your session limit · resets 8:50am (UTC) (exit code: 1)"
    assert llm._is_quota(Exception(msg))
    assert not llm._is_quota(Exception("429 rate_limit_error: exceeded your per-minute rate limit"))
    calls = []

    async def boom(*a, **k):
        calls.append(1)
        raise RuntimeError(msg)

    monkeypatch.setattr(llm, "backend", lambda: "sdk")
    monkeypatch.setattr(llm, "_sdk_call", boom)
    with pytest.raises(llm.QuotaError):
        asyncio.run(llm.complete("hi", purpose="t"))
    assert len(calls) == 1  # no retries


def test_resume_runs_only_missing_pairs_and_feeds_reuse(tmp_path):
    import asyncio
    import json
    from hrag import llm
    from hrag.eval.run_eval import _load_done, _run_items
    out = tmp_path / "gen.jsonl"
    items = [dict(qid=f"q{i}") for i in range(4)]
    systems = ["single_shot", "agent"]
    seen = []

    def work_factory(quota_at):
        async def work(it, system, prev):
            seen.append((it["qid"], system, sorted(prev)))
            if (it["qid"], system) == quota_at:
                raise llm.QuotaError("session limit")
            return dict(qid=it["qid"], system=system, verdict="pass")
        return work

    left = asyncio.run(_run_items(items, systems, work_factory(("q2", "agent")), out, {}, concurrency=1))
    assert left == 3 and len(open(out).readlines()) == 5                     # q2/agent, q3/both still missing
    done = _load_done(out, resume=True)
    seen.clear()
    assert asyncio.run(_run_items(items, systems, work_factory(None), out, done, 1)) == 0
    # the resumed agent run on q2 still sees q2's single-shot record (for the answer reuse)
    assert seen == [("q2", "agent", ["single_shot"]), ("q3", "single_shot", []), ("q3", "agent", ["single_shot"])]
    assert len(open(out).readlines()) == 8 and all("error" not in json.loads(l) for l in open(out))


def test_paired_stats():
    from hrag.eval.stats import compare, mcnemar_exact
    assert mcnemar_exact(2, 1) == 1.0 and abs(mcnemar_exact(10, 0) - 2 / 1024) < 1e-12
    recs = []
    for m in range(4):                       # 4 matters x 5 questions; the agent passes 2 more questions
        for q in range(5):
            qid = f"m{m}::q{q}"
            recs.append(dict(qid=qid, system="single_shot", verdict="pass" if q < 3 else "fail", llm_calls=1))
            recs.append(dict(qid=qid, system="agent", verdict="pass" if q < 3 or (q == 3 and m < 2) else "fail",
                             llm_calls=3, reused=q == 0))
    comp = compare(recs, n_boot=500)
    s = comp["all"]["metrics"]["pass (rubric)"]
    assert s["n"] == 20 and s["diff"] == 0.1 and (s["only_a"], s["only_b"]) == (2, 0)
    assert s["ci95"][0] <= 0.1 <= s["ci95"][1]
    assert comp["agent took its own path"]["n_questions"] == 16                # 4 reused answers left out


def test_key_fact_recall():
    from hrag.eval.run_eval import key_fact_recall
    assert key_fact_recall("The HHI rises to 3552, a delta of 1,242.", ["3,552", "1242", "Section 7A"]) == 0.667
    assert key_fact_recall("anything", []) is None


def test_rejudge_majority_and_pairwise_order(tmp_path, monkeypatch):
    import asyncio
    import json
    from hrag.eval import judge, run_eval
    monkeypatch.setattr(run_eval, "EVAL_DIR", tmp_path)
    rec = lambda q, s, v, ans, reused=False: dict(qid=q, task_id="t", system=s, kind="fact", verdict=v,  # noqa: E731
                                                  answer=ans, reused=reused, evidence_ids=[])
    lines = [rec("q1", "single_shot", "pass", "A1"), rec("q1", "agent", "fail", "B1"),
             rec("q2", "single_shot", "pass", "A2"), rec("q2", "agent", "pass", "A2", reused=True)]
    (tmp_path / "gen_t.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
    items = {q: dict(qid=q, task_id="t", question="?", expected_answer="x") for q in ("q1", "q2")}

    async def fake_judge(it, r):
        return {"verdict": "pass"}

    assert asyncio.run(run_eval.rejudge("t", items, "verdict", fake_judge, "pass")) == 0
    rj = [json.loads(l) for l in open(tmp_path / "rejudge_t.jsonl")]
    assert sorted((r["system"], r["majority"]) for r in rj) == [("agent", "pass"), ("single_shot", "pass")]
    records, _ = run_eval.load_records("t")
    assert {r["system"]: r.get("verdict_majority") for r in records if r["qid"] == "q1"} == \
        {"single_shot": "pass", "agent": "pass"}

    async def fake_pairwise(question, reference, first, second, evidence):
        return {"winner": "1" if first == "B1" else "2", "reasoning": ""}     # always prefers the agent's answer

    monkeypatch.setattr(judge, "judge_pairwise", fake_pairwise)
    asyncio.run(run_eval.pairwise("t", items, lambda it: ""))
    prefs = [json.loads(l) for l in open(tmp_path / "pairwise_t.jsonl")]
    assert [(p["qid"], p["winner"]) for p in prefs] == [("q1", "agent")]    # q2 reused: identical answers


def test_overmatched_facts_go_to_llm_labelling():
    from hrag.eval.build_qa import match_chunks
    df = pd.DataFrame(dict(chunk_id=[f"c{i}" for i in range(10)], header=["h"] * 10,
                           text=[f"Acme Corp clause {i}" for i in range(10)]))
    item = {"key_facts": ["Acme Corp"]}
    assert len(match_chunks(item, df)) == 10 and silver_labels(item, df) == []  # too generic for a string label


def test_bridge_candidates_link_two_documents():
    import random
    from hrag.eval.build_hard import bridge_candidates, entities
    assert "Zhonghe Provincial Semiconductor Manufacturing Co" in entities(
        "Senior Process Engineer, Zhonghe Provincial Semiconductor Manufacturing Co., Hefei")
    assert "Zhonghe PSIC" in entities('Zhonghe Provincial State Investment Corp. ("Zhonghe PSIC") holds 19.5%')
    df = pd.DataFrame(dict(
        chunk_id=["a", "b", "c", "d"], doc_id=["bio", "lpa", "lpa", "memo"], section=["s"] * 4, kind=["para"] * 4,
        text=["Dr. Lin Fang worked at Acme Widgets Ltd. in Hefei from 2011 to 2019 as process engineer.",
              "Principal subsidiaries of the investor include Acme Widgets Ltd., a fabrication company.",
              "Acme Widgets Ltd. supplies wafers to the fund's other holdings under long-term contracts.",
              "Unrelated memo about filing fees and HSR thresholds."]))
    cands = bridge_candidates(df, random.Random(0))
    assert [c["entity"] for c in cands] == ["Acme Widgets Ltd"]
    assert {r.doc_id for r in cands[0]["chunks"]} == {"bio", "lpa"}   # two different documents
