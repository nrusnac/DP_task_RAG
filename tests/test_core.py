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
