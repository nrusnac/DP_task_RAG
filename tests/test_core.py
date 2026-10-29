"""Fast unit tests (no GPU, no LLM). Run: pytest -q"""
import json
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


def test_paired_stats_cluster_by_matter():
    from hrag.eval import stats
    recs = []
    for m in range(4):  # the agent passes one extra question in each of 4 matters
        for k in range(3):
            q = f"matter{m}::C-{k}"
            recs += [dict(qid=q, system="agent", verdict="pass", faithful=True),
                     dict(qid=q, system="single_shot", verdict="pass" if k else "fail", faithful=True)]
    out = stats.compare(recs)
    s = out["metrics"]["pass (Harvey rubric judge)"]
    assert (out["n_questions"], out["n_matters"], s["only_a"], s["only_b"]) == (12, 4, 4, 0)
    assert s["diff"] == 0.333 and s["ci95"] == [0.333, 0.333]
    assert s["p"] == 0.125  # exact: 2 of the 16 sign patterns over 4 matters are as extreme
    assert out["metrics"]["faithful (Haiku judge)"]["p"] == 1.0


def test_quota_error_stops_generation_and_resume_runs_the_rest(tmp_path, monkeypatch):
    import asyncio
    from hrag import llm
    from hrag.eval import judge, run_eval

    assert llm._is_quota(RuntimeError("You've hit your session limit · resets 8:50am"))
    assert not llm._is_quota(RuntimeError("429 rate_limit_error"))
    items = [dict(qid=f"t::C-{i}", task_id="t", kind="fact", question="q", gold=[]) for i in range(4)]
    monkeypatch.setattr(run_eval, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(run_eval, "load_qa", lambda: [dict(i) for i in items])
    monkeypatch.setattr(run_eval, "available_variants", lambda: ["header"])
    calls = []

    async def fake_system(system, it, variant):
        calls.append(it["qid"])
        if quota and it["qid"] == "t::C-2":
            raise llm.QuotaError("session limit")
        return dict(answer="a", evidence=[], citations=[], llm_calls=1, seconds=0.1, route="simple")

    async def fake_judge(it, answer):
        return dict(verdict="pass", reasoning="")

    monkeypatch.setattr(run_eval, "_run_system", fake_system)
    monkeypatch.setattr(judge, "judge_correctness", fake_judge)
    quota = True
    asyncio.run(run_eval.generation(n=4, systems=["single_shot"], concurrency=1, judge_faith=False, tag="x"))
    first = [json.loads(l) for l in open(tmp_path / "gen_x.jsonl")]
    assert all("error" not in r for r in first) and "t::C-2" not in {r["qid"] for r in first}
    calls.clear()
    quota = False
    s = asyncio.run(run_eval.generation(n=4, systems=["single_shot"], concurrency=1, judge_faith=False, tag="x",
                                        resume=True))
    assert sorted(calls) == sorted({i["qid"] for i in items} - {r["qid"] for r in first})
    assert s["single_shot"]["n"] == 4
