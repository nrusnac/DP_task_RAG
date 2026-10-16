"""C1-C2: turn Harvey rubric criteria into a QA eval set with silver retrieval labels.

Harvey LAB has no retrieval gold labels (criteria[].sources is empty), so:
  C1  Claude rewrites each sampled criterion as the question a lawyer would ask the data room, plus the expected
      answer and 0-4 verbatim key facts (figures, names, section refs) that the source document must contain.
  C2  A chunk of that task's data room is a silver gold chunk if it contains the key facts (normalised match).

    python -m hrag.eval.build_qa --tasks 10 --per-task 8
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re

import pandas as pd

from .. import llm
from ..config import CFG

QA_PROMPT = """You are building an evaluation set for a retrieval system over one legal matter's data room.
The questions must test what the SOURCE DOCUMENTS say (facts, figures, terms, positions, issues they reveal), never
the deliverable the lawyer will write. If a criterion only grades the deliverable itself (its format, structure,
tone, or whether it "proposes"/"drafts"/"recommends" something not stated in the sources), set kind to "skip".
For EACH rubric criterion below, write:
- question: the question a lawyer would ask the data room to check the criterion. Self-contained (name the
  market, party, document or issue), answerable from the documents, and it must NOT reveal the answer.
- expected_answer: the answer implied by the criterion (1-2 sentences).
- key_facts: 0-4 short strings that must appear VERBATIM in the source document to answer it: figures exactly as
  a document would print them (e.g. "3,552", "50.0%", "$164 million"), party names, section numbers. Use [] when the
  criterion is analytical or judgment-based.
- kind: "fact" if it checks specific facts stated in the documents, "analysis" if it checks issues or reasoning
  that must be grounded in the documents, "skip" (see above).

Matter: {title}
Assignment given to the lawyer: {instructions}

Criteria:
{criteria}"""

QA_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["items"], "properties": {"items": {
    "type": "array", "items": {"type": "object", "additionalProperties": False,
                               "required": ["criterion_id", "question", "expected_answer", "key_facts", "kind"],
                               "properties": {"criterion_id": {"type": "string"}, "question": {"type": "string"},
                                              "expected_answer": {"type": "string"},
                                              "key_facts": {"type": "array", "items": {"type": "string"}},
                                              "kind": {"type": "string", "enum": ["fact", "analysis", "skip"]}}}}}}


def _norm(s: str) -> str:
    s = s.lower().replace("’", "'").replace("—", "-").replace("–", "-")
    return re.sub(r"\s+", " ", s).strip()


def _variants(fact: str) -> list[str]:
    f = _norm(fact)
    out = {f}
    if re.fullmatch(r"[\d,.$%]+(?: million| billion)?", f):
        out.add(f.replace(",", ""))
        out.add(f.replace("$", ""))
    return [v for v in out if len(v) >= 2]


def silver_labels(item: dict, task_chunks: pd.DataFrame) -> list[str]:
    facts = [f for f in item.get("key_facts", []) if f and len(f.strip()) >= 2]
    if not facts:
        return []
    texts = [_norm(t) for t in (task_chunks.header + " " + task_chunks.text)]
    texts = [t + " " + re.sub(r"(?<=\d),(?=\d{3})", "", t) for t in texts]  # also match "1242" against "1,242"
    need = len(facts) if len(facts) <= 2 else -(-2 * len(facts) // 3)  # all of 1-2 facts, else >= 2/3 of them
    gold = []
    for cid, t in zip(task_chunks.chunk_id, texts):
        hit = sum(any(v in t for v in _variants(f)) for f in facts)
        if hit >= need:
            gold.append(cid)
    return gold if len(gold) <= 8 else []  # facts matching >8 chunks are too generic to serve as labels


async def build(n_tasks: int = 10, per_task: int = 8, seed: int = 7) -> list[dict]:
    tasks = [json.loads(l) for l in open(CFG.artifacts_dir / "tasks.jsonl")]
    chunks = pd.read_parquet(CFG.artifacts_dir / "chunks.parquet")
    rng = random.Random(seed)
    # drafting tasks grade the lawyer's own deliverable; analysis/extraction tasks test what the documents say
    tasks = [t for t in tasks if not t["task_id"].startswith("draft-")]
    tasks = rng.sample(tasks, min(n_tasks, len(tasks)))

    async def one(t):
        crit = rng.sample(t["criteria"], min(per_task, len(t["criteria"])))
        crit_txt = "\n".join(f"- id={c['id']} | {c['title']} | {c['match_criteria']}" for c in crit)
        out = await llm.complete(QA_PROMPT.format(title=t["title"], instructions=t["instructions"][:1500],
                                                  criteria=crit_txt),
                                 model=CFG.strong_llm, schema=QA_SCHEMA, purpose="qa_gen", max_tokens=8000)
        by_id = {c["id"]: c for c in crit}
        tc = chunks[chunks.task_id == t["task_id"]]
        items = []
        for it in out["items"]:
            c = by_id.get(it["criterion_id"])
            if not c or it["kind"] == "skip":
                continue
            items.append(dict(qid=f"{t['task_id']}::{c['id']}", task_id=t["task_id"], criterion_id=c["id"],
                              criterion_title=c["title"], match_criteria=c["match_criteria"],
                              question=it["question"], expected_answer=it["expected_answer"],
                              key_facts=it["key_facts"], kind=it["kind"], gold=silver_labels(it, tc)))
        return items

    res = await asyncio.gather(*[one(t) for t in tasks], return_exceptions=True)
    items = [i for r in res if not isinstance(r, Exception) for i in r]
    errors = [repr(r)[:200] for r in res if isinstance(r, Exception)]
    out_dir = CFG.artifacts_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "qa.jsonl", "w") as f:
        for i in items:
            f.write(json.dumps(i) + "\n")
    print(json.dumps(dict(n_items=len(items), n_with_gold=sum(bool(i["gold"]) for i in items),
                          by_kind={k: sum(i["kind"] == k for i in items) for k in ("fact", "analysis")},
                          errors=errors, llm=llm.STATS.snapshot()), indent=1))
    return items


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=10)
    ap.add_argument("--per-task", type=int, default=8)
    a = ap.parse_args()
    asyncio.run(build(a.tasks, a.per_task))
