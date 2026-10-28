"""C1-C2: turn Harvey rubric criteria into a QA eval set with silver retrieval labels.

Harvey LAB has no retrieval gold labels (criteria[].sources is empty), so:
  C1  Claude rewrites each sampled criterion as the question a lawyer would ask the data room, plus the expected
      answer and 0-4 verbatim key facts (figures, names, section refs) that the source document must contain.
  C2  A chunk of that task's data room is a silver gold chunk if it contains the key facts (normalised match). When
      the facts match more than 8 chunks (too generic to be a label on their own), Claude picks the chunks among
      those matches that actually answer the question (gold_source="llm"). Candidates come from string matching,
      never from our retriever, so the labels don't favour it.

  C2b audit: one Sonnet call per item checks the expected answer against the matter's documents (passages found by
      keyword search and by the key facts, never by our retriever): supported / contradicted / not_determinable /
      outside_knowledge. Rubric criteria can describe issues the documents in this corpus don't have (two test items
      fail every correct answer); the eval reports results per audit stratum.

Splits: `dev` = the 110 questions that shaped the design; `test` = questions for matters never used before.

    python -m hrag.eval.build_qa build --tasks 10 --per-task 8   # (re)build from scratch: all items are dev
    python -m hrag.eval.build_qa extend --split test             # add questions for every matter not yet covered
    python -m hrag.eval.build_qa relabel --split dev             # recompute silver labels, keep the questions
    python -m hrag.eval.build_qa audit --split test              # check expected answers against the documents
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

QA_PATH = CFG.artifacts_dir / "eval" / "qa.jsonl"
MAX_GOLD = 8              # more string matches than this = facts too generic to be a label on their own
MAX_LLM_CANDIDATES = 30   # passages shown to the labelling call

QA_PROMPT = """You are building an evaluation set for a retrieval system over one legal matter's data room.
The questions must test what the SOURCE DOCUMENTS say (facts, figures, terms, positions, issues they reveal), never
the deliverable the lawyer will write. If a criterion only grades the deliverable itself (its format, structure,
tone, citation style, length), set kind to "skip".
For EACH rubric criterion below, write:
- question: the question a lawyer would ask to satisfy the criterion. Self-contained (name the market, party,
  document or issue) and it must NOT reveal the answer. It must ask for EVERYTHING the criterion needs to PASS:
  if the criterion requires a recommendation, timeline or advice, the question must ask for it too
  (e.g. "... and what negotiation timeline should CIS follow before the June 16 deadline?").
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

LABEL_PROMPT = """You are labelling an evaluation set for document retrieval over one legal matter's data room.
Every numbered passage below mentions the question's key facts. Return the numbers of the passages that themselves
state the information needed to answer the question (at most {max_gold}, most important first). Exclude passages that
only mention the same names or figures in passing or in a different context. Return an empty list if none does.

Question: {question}
Expected answer: {expected_answer}
Key facts: {key_facts}

{passages}"""
LABEL_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["relevant"],
                "properties": {"relevant": {"type": "array", "items": {"type": "integer"}}}}


AUDIT_PROMPT = """You are auditing one item of an evaluation set for a legal document-QA system. The item was written
from a grading rubric for this matter; check it against the matter's own documents.

Question: {question}
Rubric criterion: {criterion}
Expected answer: {expected_answer}

Passages from the matter's documents (found by keyword search for the question and the expected answer, and by the
item's key facts):
{passages}

Classify the expected answer:
- "supported": the passages state it, or it follows from them directly;
- "contradicted": a passage states something incompatible with it (quote that passage);
- "not_determinable": the passages neither confirm nor contradict it (e.g. it says something is ABSENT from a
  document and the passages do not show the whole provision, or the deciding passage may not be among them);
- "outside_knowledge": it turns mainly on legal knowledge or judgment outside the documents (law, regulation,
  agency practice), not on what the documents say.
Give the deciding quote (empty if none) and a one-sentence note."""
AUDIT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["quote", "note", "verdict"],
                "properties": {"quote": {"type": "string"}, "note": {"type": "string"},
                               "verdict": {"type": "string", "enum": ["supported", "contradicted", "not_determinable",
                                                                     "outside_knowledge"]}}}
MAX_AUDIT_PASSAGES = 24


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


def match_chunks(item: dict, task_chunks: pd.DataFrame) -> list[str]:
    """Every chunk of the matter that contains the item's key facts (all of 1-2 facts, else >= 2/3 of them), the
    chunks hitting the most facts first."""
    facts = [f for f in item.get("key_facts", []) if f and len(f.strip()) >= 2]
    if not facts:
        return []
    texts = [_norm(t) for t in (task_chunks.header + " " + task_chunks.text)]
    texts = [t + " " + re.sub(r"(?<=\d),(?=\d{3})", "", t) for t in texts]  # also match "1242" against "1,242"
    need = len(facts) if len(facts) <= 2 else -(-2 * len(facts) // 3)  # all of 1-2 facts, else >= 2/3 of them
    hits = []
    for pos, (cid, t) in enumerate(zip(task_chunks.chunk_id, texts)):
        hit = sum(any(v in t for v in _variants(f)) for f in facts)
        if hit >= need:
            hits.append((-hit, pos, cid))
    return [cid for _, _, cid in sorted(hits)]


def silver_labels(item: dict, task_chunks: pd.DataFrame) -> list[str]:
    gold = match_chunks(item, task_chunks)
    pos = {c: i for i, c in enumerate(task_chunks.chunk_id)}
    gold = sorted(gold, key=pos.get)  # document order, as before
    return gold if len(gold) <= MAX_GOLD else []  # facts matching >8 chunks are too generic to serve as labels


async def llm_label(item: dict, task_chunks: pd.DataFrame, candidates: list[str]) -> list[str]:
    """Pick, among string-matched candidate chunks, the ones that answer the question (at most MAX_GOLD)."""
    rows = task_chunks.set_index("chunk_id").loc[candidates[:MAX_LLM_CANDIDATES]]
    passages = "\n\n".join(f"[{i}] ({r.header})\n{r.text[:1200]}" for i, r in enumerate(rows.itertuples(), 1))
    out = await llm.complete(LABEL_PROMPT.format(max_gold=MAX_GOLD, question=item["question"],
                                                 expected_answer=item["expected_answer"],
                                                 key_facts=", ".join(item["key_facts"]), passages=passages),
                             model=CFG.strong_llm, schema=LABEL_SCHEMA, purpose="qa_label", max_tokens=1000)
    picked = [i for i in dict.fromkeys(out.get("relevant", [])) if 1 <= i <= len(rows)][:MAX_GOLD]
    return [rows.index[i - 1] for i in picked]


async def label(item: dict, task_chunks: pd.DataFrame) -> dict:
    """gold + gold_source ("string" | "llm" | None) + n_matches for one item."""
    m = match_chunks(item, task_chunks)
    if len(m) > MAX_GOLD:
        return dict(gold=await llm_label(item, task_chunks, m), gold_source="llm", n_matches=len(m))
    gold = silver_labels(item, task_chunks)
    return dict(gold=gold, gold_source="string" if gold else None, n_matches=len(m))


def lexical_top(task_chunks: pd.DataFrame, query: str, k: int = 10) -> list[str]:
    """BM25 over one matter's chunks: a neutral candidate source for audits and the hard sets (lexical only, never the
    system's own hybrid retriever and reranker)."""
    import bm25s
    from ..index.build import bm25_tokenize
    toks = bm25_tokenize([query])
    if not toks[0] or task_chunks.empty:
        return []
    bm = bm25s.BM25()
    bm.index(bm25_tokenize((task_chunks.header + "\n" + task_chunks.text).tolist()), show_progress=False)
    idx, sc = bm.retrieve(toks, k=min(k, len(task_chunks)), show_progress=False)
    return [task_chunks.chunk_id.iat[int(i)] for i, v in zip(idx[0], sc[0]) if v > 0]


async def audit_item(item: dict, task_chunks: pd.DataFrame) -> dict:
    # deep keyword search: a clause can use other words than the rubric (e.g. "Conduct of Business Pending the
    # Closing" for "interim operating covenants") and rank ~15th
    cands = list(dict.fromkeys(list(item.get("gold") or [])
                               + lexical_top(task_chunks, item["question"] + " " + item["expected_answer"], 20)
                               + match_chunks(item, task_chunks)[:6]))
    rows = task_chunks.set_index("chunk_id").loc[cands[:MAX_AUDIT_PASSAGES]]
    passages = "\n\n".join(f"[{i}] ({r.header})\n{r.text[:3000]}" for i, r in enumerate(rows.itertuples(), 1))
    out = await llm.complete(AUDIT_PROMPT.format(question=item["question"], expected_answer=item["expected_answer"],
                                                 criterion=f"{item['criterion_title']}: {item['match_criteria']}",
                                                 passages=passages or "(none found)"),
                             model=CFG.strong_llm, schema=AUDIT_SCHEMA, purpose="qa_audit", max_tokens=1500)
    return dict(audit=out["verdict"], audit_quote=out.get("quote", ""), audit_note=out.get("note", ""))


def _gather_ok(res: list, what: str) -> list:
    """Results of gather(return_exceptions=True): a quota error stops the build; other errors are reported."""
    for r in res:
        if isinstance(r, llm.QuotaError):
            raise r
    errors = [repr(r)[:200] for r in res if isinstance(r, Exception)]
    if errors:
        print(f"{what}: {len(errors)} errors", errors)
    return [r for r in res if not isinstance(r, Exception)]


async def _items_for_task(t: dict, crit: list[dict], tc: pd.DataFrame, split: str) -> list[dict]:
    crit_txt = "\n".join(f"- id={c['id']} | {c['title']} | {c['match_criteria']}" for c in crit)
    out = await llm.complete(QA_PROMPT.format(title=t["title"], instructions=t["instructions"][:1500], criteria=crit_txt),
                             model=CFG.strong_llm, schema=QA_SCHEMA, purpose="qa_gen", max_tokens=8000)
    by_id = {c["id"]: c for c in crit}
    items = []
    for it in out["items"]:
        c = by_id.get(it["criterion_id"])
        if not c or it["kind"] == "skip":
            continue
        item = dict(qid=f"{t['task_id']}::{c['id']}", task_id=t["task_id"], split=split, criterion_id=c["id"],
                    criterion_title=c["title"], match_criteria=c["match_criteria"],
                    question=it["question"], expected_answer=it["expected_answer"],
                    key_facts=it["key_facts"], kind=it["kind"])
        items.append(item | await label(item, tc))
    return items


def _questionable(tasks: list[dict]) -> list[dict]:
    # drafting tasks grade the lawyer's own deliverable; analysis/extraction tasks test what the documents say
    return [t for t in tasks if not t["task_id"].startswith("draft-")]


def _write(items: list[dict]) -> None:
    QA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(QA_PATH, "w") as f:
        for i in items:
            f.write(json.dumps(i) + "\n")


def _report(items: list[dict], **extra) -> None:
    by = lambda key: {k: sum(i.get(key) == k for i in items) for k in sorted({i.get(key) for i in items}, key=str)}  # noqa: E731
    print(json.dumps(dict(n_items=len(items), n_with_gold=sum(bool(i["gold"]) for i in items), by_split=by("split"),
                          by_kind=by("kind"), gold_source=by("gold_source"), **extra, llm=llm.STATS.snapshot()),
                     indent=1))


def _load_inputs():
    tasks = [json.loads(l) for l in open(CFG.artifacts_dir / "tasks.jsonl")]
    chunks = pd.read_parquet(CFG.artifacts_dir / "chunks.parquet")
    return tasks, chunks


async def build(n_tasks: int = 10, per_task: int = 8, seed: int = 7) -> list[dict]:
    """Original builder (main): sample n_tasks matters, per_task criteria each. Overwrites qa.jsonl."""
    tasks, chunks = _load_inputs()
    rng = random.Random(seed)
    tasks = rng.sample(_questionable(tasks), min(n_tasks, len(_questionable(tasks))))
    jobs = [_items_for_task(t, rng.sample(t["criteria"], min(per_task, len(t["criteria"]))),
                            chunks[chunks.task_id == t["task_id"]], "dev") for t in tasks]
    items = [i for r in _gather_ok(await asyncio.gather(*jobs, return_exceptions=True), "qa_gen") for i in r]
    _write(items)
    _report(items)
    return items


async def extend(per_task: int = 8, seed: int = 7, split: str = "test") -> list[dict]:
    """Append questions for every questionable matter that has none yet (existing qids and labels untouched)."""
    tasks, chunks = _load_inputs()
    existing = [json.loads(l) for l in open(QA_PATH)] if QA_PATH.exists() else []
    existing = [dict(i, split=i.get("split", "dev")) for i in existing]  # rows from before splits existed are dev
    have = {i["task_id"] for i in existing}
    todo = [t for t in _questionable(tasks) if t["task_id"] not in have]
    print("new matters:", [t["task_id"] for t in todo], flush=True)
    jobs = []
    for t in todo:
        rng = random.Random(f"{seed}:{t['task_id']}")  # per-matter seed: independent of which matters exist
        jobs.append(_items_for_task(t, rng.sample(t["criteria"], min(per_task, len(t["criteria"]))),
                                    chunks[chunks.task_id == t["task_id"]], split))
    new = [i for r in _gather_ok(await asyncio.gather(*jobs, return_exceptions=True), "qa_gen") for i in r]
    _write(existing + new)
    _report(existing + new, added=len(new), new_matters=len(todo))
    return new


async def relabel(split: str | None = None) -> list[dict]:
    """Recompute the silver labels (string match, LLM pick when >8 matches) of every item, or of one split's items;
    questions are kept."""
    _, chunks = _load_inputs()
    items = [dict(i, split=i.get("split", "dev")) for i in map(json.loads, open(QA_PATH))]
    todo = [i for i in items if split in (None, i["split"])]
    res = await asyncio.gather(*[label(i, chunks[chunks.task_id == i["task_id"]]) for i in todo],
                               return_exceptions=True)
    for r in res:
        if isinstance(r, llm.QuotaError):
            raise r
    new = {}
    for i, r in zip(todo, res):
        if isinstance(r, Exception):  # keep the old label rather than silently dropping it
            print("relabel error", i["qid"], repr(r)[:200])
            r = dict(gold=i.get("gold", []), gold_source=i.get("gold_source"), n_matches=i.get("n_matches"))
        new[i["qid"]] = i | r
    out = [new.get(i["qid"], i) for i in items]
    _write(out)
    _report(out)
    return out


async def audit(split: str | None = "test", qids: list[str] | None = None) -> list[dict]:
    """Audit the expected answers of one split's items (or of `qids`) against the documents; questions and labels are
    kept, the verdict is stored as audit / audit_quote / audit_note."""
    _, chunks = _load_inputs()
    items = [dict(i, split=i.get("split", "dev")) for i in map(json.loads, open(QA_PATH))]
    todo = [i for i in items if (i["qid"] in set(qids) if qids else split in (None, i["split"]) and not i.get("audit"))]
    res = await asyncio.gather(*[audit_item(i, chunks[chunks.task_id == i["task_id"]]) for i in todo],
                               return_exceptions=True)
    new, stopped = {}, sum(isinstance(r, llm.QuotaError) for r in res)
    if stopped:  # results so far are still written: rerun after the reset to audit only the rest
        print(f"STOPPED at a session limit: {stopped} items left -- rerun after the reset", flush=True)
    for i, r in zip(todo, res):
        if isinstance(r, llm.QuotaError):
            continue
        if isinstance(r, Exception):
            print("audit error", i["qid"], repr(r)[:200])
            continue
        new[i["qid"]] = i | r
    out = [new.get(i["qid"], i) for i in items]
    _write(out)
    audited = [i for i in out if i.get("audit")]
    print(json.dumps({"audited": len(audited), "by_verdict": {v: sum(i["audit"] == v for i in audited)
                                                               for v in sorted({i["audit"] for i in audited})},
                      "llm": llm.STATS.snapshot()}, indent=1))
    return [new[q] for q in new]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["build", "extend", "relabel", "audit"])
    ap.add_argument("--tasks", type=int, default=10)
    ap.add_argument("--per-task", type=int, default=8)
    ap.add_argument("--split", default=None,
                    help="extend: split of the new items (default test); relabel/audit: only this split (audit: test)")
    ap.add_argument("--qids", nargs="+", default=None, help="audit: only these items")
    a = ap.parse_args()
    if a.what == "build":
        asyncio.run(build(a.tasks, a.per_task))
    elif a.what == "extend":
        asyncio.run(extend(a.per_task, split=a.split or "test"))
    elif a.what == "audit":
        asyncio.run(audit(a.split or "test", a.qids))
    else:
        asyncio.run(relabel(a.split))
