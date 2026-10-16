"""Render artifacts/eval/*.json as Markdown tables and splice them into README.md and docs/05-evaluation.md
(between the RESULTS:START / RESULTS:END markers).

    python scripts/report.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "artifacts" / "eval"


def _table(rows: list[dict], cols: list[str], first: str) -> str:
    head = f"| {first} | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    return head + "\n".join(f"| {r['_name']} | " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols)
                            + " |" for r in rows)


def render() -> str:
    parts = []
    p = EV / "retrieval_results.json"
    if p.exists():
        res = json.loads(p.read_text())
        rows = [dict(v, _name=f"`{k}`") for k, v in res.items()]
        n = next(iter(res.values())).get("n")
        parts.append(f"**Retrieval** ({n} questions with silver labels; searched within the matter's data room unless marked corpus-wide; `foreign@5` = share of top-5 hits from other matters)\n\n"
                     + _table(rows, ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10", "foreign@5", "ms_per_query"], "variant / mode"))
    p = EV / "crag_calibration.json"
    if p.exists():
        cal = json.loads(p.read_text())
        rows = [dict(b, _name=b["range"]) for b in cal["buckets"] if b["positives"] or b["negatives"]]
        parts.append(f"**CRAG calibration** (variant `{cal['variant']}`): top rerank score for answerable questions "
                     f"(positives) vs. the same questions asked against another matter's data room (negatives). "
                     f"Suggested thresholds `{cal['suggested']}`; coverage `{cal['coverage']}`\n\n"
                     + _table(rows, ["positives", "gold_retrieved", "negatives"], "top rerank score"))
    labels = {  # display order: final results first, then the history that led to them
        "naive_single_shot_agent": "**final**, Colab T4, `ctx` index, 30 questions, after the refinement fix",
        "agent-simple_agent-complex_agent": "**final** Adaptive-RAG ablation, Colab T4, `ctx` index, 20 questions, after the fix",
        "before_fix_ctx": "first Colab run, BEFORE the refinement fix, same 30 questions",
        "ablation_before_fix_ctx": "first Colab Adaptive-RAG ablation, BEFORE the fix, same 20 questions",
        "after_fix_header_local": "targeted local check of the fix, `header` index, 4 failed + 6 passed questions",
    }
    found = {p.stem.replace("summary_gen_", ""): p for p in EV.glob("summary_gen_*.json")}
    for key in [k for k in labels if k in found] + sorted(k for k in found if k not in labels):
        s = json.loads(found[key].read_text())
        rows = [dict(v, _name=f"`{k}`", routes=", ".join(f"{a}:{b}" for a, b in (v.get("routes") or {}).items()))
                for k, v in s.items()]
        parts.append(f"**Generation** ({labels.get(key, key)}; Harvey rubric judge = Claude Sonnet)\n\n"
                     + _table(rows, ["n", "pass_rate", "pass_rate_fact", "pass_rate_analysis", "faithful_rate",
                                     "abstain_rate", "evidence_has_gold", "avg_evidence_words", "avg_llm_calls",
                                     "avg_seconds", "routes"], "system"))
    for p in sorted(EV.glob("summary_unanswerable_*.json")):
        s = json.loads(p.read_text())
        rows = [dict(v, _name=f"`{k}`") for k, v in s.items()]
        parts.append(f"**Unanswerable questions** (`{p.stem.replace('summary_unanswerable_', '')}` index; each question asked "
                     "in ANOTHER matter's data room; judged by Claude Haiku: did the answer decline?)\n\n"
                     + _table(rows, ["n", "declined_rate", "hallucination_rate", "avg_llm_calls"], "system"))
    return "\n\n".join(parts) or "*No results yet.*"


def splice(path: Path, body: str):
    txt = path.read_text()
    new = re.sub(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->",
                 f"<!-- RESULTS:START -->\n{body}\n<!-- RESULTS:END -->", txt, flags=re.S)
    path.write_text(new)


if __name__ == "__main__":
    body = render()
    for f in (ROOT / "README.md", ROOT / "docs" / "05-evaluation.md"):
        splice(f, body)
    print(body)
