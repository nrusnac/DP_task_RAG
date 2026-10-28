"""B2-B4: hard evaluation sets with gold passages known by construction (docs/05-evaluation.md, "Hard sets").

The QA splits are written from rubric criteria and are mostly single-hop: one search finds the answer (single-shot's
top 8 held a gold chunk for 93% of labelled test questions), so they can't show what multi-step retrieval adds
(Adaptive-RAG, Jeong et al. 2024). These sets are built from the documents themselves, never from our retriever:

  bridge  two passages in different documents share an entity; the question needs a fact from each and does not
          name the entity, so the second passage can only be found after reading the first (a two-step link)
  aspect  three related passages from three documents; the question needs one fact from each
  absent  a specific question such a data room could plausibly answer, that no passage of this matter answers

A writer call proposes each item; a separate verifier call checks it (bridge/aspect: no single passage answers the
question and all of them together do; absent: no keyword-retrieved passage answers it, even partly). Matters: every
matter outside the dev split, at most 2 items per matter and kind. Items are appended to qa_hard.jsonl as they are
accepted, so a run stopped by a session limit continues where it stopped.

    python -m hrag.eval.build_hard --kinds bridge aspect absent --n 25
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import subprocess

import numpy as np
import pandas as pd

from .. import llm
from ..config import CFG
from .build_qa import QA_PATH, _load_inputs, lexical_top

HARD_PATH = CFG.artifacts_dir / "eval" / "qa_hard.jsonl"
PER_MATTER = 2       # accepted items per matter and kind
WAVE = 6             # candidates written + verified concurrently
# the link behind test question C-022 motivated the agent's link-finding step: no hard item may reuse it
EXCLUDED_LINKS = {"identify-issues-in-cfius-mandatory-declaration": ("Zhonghe", "Fang")}

# entity mentions that can link two documents: quoted defined terms, organisation names, titled person names
ENTITY_PATTERNS = [
    r'\(\s*(?:the\s+)?["“]([A-Z][\w&.,\- ]{2,40}?)["”]\s*\)',
    r"\b((?:[A-Z][\w&\-]*\.?\s+){1,5}(?:Co\.|Corp\.|Corporation|Inc\.|LLC|L\.L\.C\.|Ltd\.|Limited|LP|L\.P\.|LLP|"
    r"GmbH|S\.A\.|plc|N\.V\.|B\.V\.|AG|Fund|Partners|Holdings|Group))",
    r"\b((?:Dr|Mr|Ms|Mrs)\.\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
]
_WORD = re.compile(r"[a-z]{4,}")

_ITEM = {"question": {"type": "string"}, "expected_answer": {"type": "string"},
         "facts": {"type": "array", "items": {"type": "string"}},
         "key_facts": {"type": "array", "items": {"type": "string"}}, "skip": {"type": "boolean"}}
WRITE_SCHEMA = {"type": "object", "additionalProperties": False, "required": list(_ITEM), "properties": _ITEM}

BRIDGE_WRITE = """You are writing one evaluation question for a legal document-search system, from two passages of
the same matter's data room that both mention "{entity}".

Write a question a lawyer on this matter could ask whose answer needs a fact from Passage A AND a fact from Passage
B, linked through {entity}: from what the question names, one passage identifies {entity}; through {entity} the other
passage completes the answer. Rules:
- The question must NOT contain "{entity}" or an obvious short form of it, and must not reveal the answer.
- It must be self-contained and specific to this matter (name the people, parties or documents it starts from).
- The answer must be short and checkable (a fact, figure, party, date, or yes/no with the reason).
If no natural question of this kind exists, set skip to true.

Passage A ({doc_a}):
{text_a}

Passage B ({doc_b}):
{text_b}

Return question, expected_answer (1-2 sentences), facts (the 2-3 facts a correct answer must state, each a short
sentence), key_facts (1-4 short strings copied VERBATIM from the passages: names, figures, dates) and skip."""

ASPECT_WRITE = """You are writing one evaluation question for a legal document-search system, from three passages
of the same matter's data room (three different documents).

Write a question a lawyer on this matter could ask whose complete answer needs one specific fact from EACH of the
three passages (comparing, reconciling or combining what the documents say about one topic). Rules:
- Self-contained and specific to this matter; it must not reveal the answer.
- Each needed fact must be short and checkable (a figure, date, party, term or position).
- The three facts must serve one natural question: do not stitch unrelated facts together.
If no natural question of this kind exists, set skip to true.

{passages}

Return question, expected_answer (1-3 sentences), facts (one short sentence per passage, in passage order),
key_facts (1-2 short strings per passage copied VERBATIM: names, figures, dates) and skip."""

VERIFY_MULTI = """Check an evaluation question that should need ALL of the passages below together.

Question: {question}
Expected answer: {expected_answer}

{passages}

1. For each passage ALONE (in order): can the question be fully answered from that passage alone?
2. With ALL passages together: is the expected answer correct and fully supported?
3. Does the question itself contain or give away the expected answer?"""
VERIFY_MULTI_SCHEMA = {"type": "object", "additionalProperties": False,
                       "required": ["alone", "together_ok", "gives_away", "note"],
                       "properties": {"alone": {"type": "array", "items": {"type": "boolean"}},
                                      "together_ok": {"type": "boolean"}, "gives_away": {"type": "boolean"},
                                      "note": {"type": "string"}}}

ABSENT_WRITE = """You are writing evaluation questions for a legal document-search system. Below are the document
titles and a sample of section headings from one matter's data room.

Propose {k} specific questions a lawyer on this matter might plausibly ask about a fact this kind of data room often
contains (a figure, date, party, clause or term: e.g. a fee, a deadline, a cap, a counterparty, a notice period), but
that these documents may NOT contain. Each must be specific to this matter (name its parties or deal) and have a short
factual answer if the documents did contain it. Avoid topics the headings show the documents cover. For each, give
3-6 search terms a document stating the answer would use.

Matter: {title}
Documents: {docs}
Section headings (sample):
{headings}"""
ABSENT_WRITE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["items"], "properties": {"items": {
    "type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["question", "terms"],
                               "properties": {"question": {"type": "string"},
                                              "terms": {"type": "array", "items": {"type": "string"}}}}}}}

VERIFY_ABSENT = """Does any passage below answer the question, fully or partly? A passage answers it only if it
states the specific fact asked for, not just something on the same topic.

Question: {question}

{passages}"""
VERIFY_ABSENT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["note", "answered"],
                        "properties": {"note": {"type": "string"},
                                       "answered": {"type": "string", "enum": ["no", "partly", "yes"]}}}


# ------------------------------------------------------------------------------------------- candidates (no LLM)
def entities(text: str) -> set[str]:
    out = set()
    for pat in ENTITY_PATTERNS:
        for m in re.findall(pat, text):
            e = " ".join(m.split()).strip(" ,.")
            if len(e) >= 6 and not e.isupper():
                out.add(e)
    return out


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def bridge_candidates(tc: pd.DataFrame, rng: random.Random, max_chunks: int = 6) -> list[dict]:
    """Entities named in >= 2 documents but <= max_chunks chunks of the matter (specific enough to be a link), each with
    its least-overlapping pair of chunks from two different documents (word Jaccard < 0.25: one search is unlikely to
    return both)."""
    rows = list(tc.itertuples())
    found: dict[str, list] = {}
    for r in rows:
        for e in entities(r.text):
            found.setdefault(e, []).append(r)
    out = []
    for e, rs in found.items():
        rs = [r for r in rows if e in r.text]  # every chunk that mentions it, not only where the regex matched
        if not 2 <= len(rs) <= max_chunks or len({r.doc_id for r in rs}) < 2:
            continue
        best = None
        for i, x in enumerate(rs):
            for y in rs[i + 1:]:
                if x.doc_id == y.doc_id:
                    continue
                wx, wy = _words(x.text), _words(y.text)
                jac = len(wx & wy) / max(1, len(wx | wy))
                if best is None or jac < best[0]:
                    best = (jac, x, y)
        if best and best[0] < 0.25:
            out.append(dict(entity=e, chunks=[best[1], best[2]]))
    rng.shuffle(out)
    return out


def aspect_candidates(tc: pd.DataFrame, rng: random.Random, n_seeds: int = 6) -> list[dict]:
    """A seed chunk plus the most similar chunk of each of two OTHER documents within a TF-IDF cosine band
    (0.12-0.45: related but not near-duplicates)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    tc = tc[tc.text.str.split().str.len() >= 40]
    if tc.doc_id.nunique() < 3:
        return []
    X = TfidfVectorizer(stop_words="english", min_df=1).fit_transform(tc.text)
    sims = (X @ X.T).toarray()
    rows = list(tc.itertuples())
    out = []
    for si in rng.sample(range(len(rows)), min(n_seeds, len(rows))):
        best: dict[str, tuple[float, int]] = {}
        for j, r in enumerate(rows):
            s = sims[si, j]
            if r.doc_id != rows[si].doc_id and 0.12 <= s <= 0.45 and s > best.get(r.doc_id, (0, -1))[0]:
                best[r.doc_id] = (s, j)
        if len(best) >= 2:
            (_, j1), (_, j2) = sorted(best.values(), reverse=True)[:2]
            if sims[j1, j2] < 0.6:
                out.append(dict(chunks=[rows[si], rows[j1], rows[j2]]))
    return out


def _passage_block(rows, labels) -> str:
    return "\n\n".join(f"Passage {lab} ({r.doc_id} § {r.section or r.kind}):\n{r.text[:3000]}"
                       for lab, r in zip(labels, rows))


# ------------------------------------------------------------------------------------------- write + verify
async def _bridge(cand: dict) -> dict | None:
    a, b = cand["chunks"]
    w = await llm.complete(BRIDGE_WRITE.format(entity=cand["entity"], doc_a=f"{a.doc_id} § {a.section or a.kind}",
                                               text_a=a.text[:3000], doc_b=f"{b.doc_id} § {b.section or b.kind}",
                                               text_b=b.text[:3000]),
                           model=CFG.strong_llm, schema=WRITE_SCHEMA, purpose="hard_write")
    if w.get("skip") or cand["entity"].lower() in w["question"].lower():
        return None
    return await _verify_multi(w, [a, b], ["A", "B"], dict(entity=cand["entity"]))


async def _aspect(cand: dict) -> dict | None:
    rows = cand["chunks"]
    w = await llm.complete(ASPECT_WRITE.format(passages=_passage_block(rows, ["1", "2", "3"])),
                           model=CFG.strong_llm, schema=WRITE_SCHEMA, purpose="hard_write")
    if w.get("skip"):
        return None
    return await _verify_multi(w, rows, ["1", "2", "3"], {})


async def _verify_multi(w: dict, rows, labels, extra: dict) -> dict | None:
    v = await llm.complete(VERIFY_MULTI.format(question=w["question"], expected_answer=w["expected_answer"],
                                               passages=_passage_block(rows, labels)),
                           model=CFG.strong_llm, schema=VERIFY_MULTI_SCHEMA, purpose="hard_verify")
    ok = len(v["alone"]) == len(rows) and not any(v["alone"]) and v["together_ok"] and not v["gives_away"]
    if not ok:
        return None
    return dict(question=w["question"], expected_answer=w["expected_answer"], facts=w["facts"],
                key_facts=w["key_facts"], gold=[r.chunk_id for r in rows], verify=v, **extra)


async def _absent_questions(tc: pd.DataFrame, title: str, rng: random.Random, k: int = 3) -> list[dict]:
    heads = sorted({f"{r.doc_id} § {r.section}" for r in tc.itertuples() if r.section})
    out = await llm.complete(ABSENT_WRITE.format(k=k, title=title, docs=", ".join(sorted(tc.doc_id.unique())),
                                                 headings="\n".join(rng.sample(heads, min(15, len(heads))))),
                             model=CFG.strong_llm, schema=ABSENT_WRITE_SCHEMA, purpose="hard_write")
    return out["items"][:k]


async def _absent(cand: dict, tc: pd.DataFrame) -> dict | None:
    q, terms = cand["question"], [t for t in cand["terms"] if len(t) >= 3]
    hits = lexical_top(tc, q + " " + " ".join(terms), 20)
    term_hits = [r.chunk_id for r in tc.itertuples() if any(t.lower() in r.text.lower() for t in terms)][:10]
    rows = tc.set_index("chunk_id").loc[list(dict.fromkeys(hits + term_hits))[:24]]
    passages = "\n\n".join(f"[{i}] ({r.header})\n{r.text[:3000]}" for i, r in enumerate(rows.itertuples(), 1))
    v = await llm.complete(VERIFY_ABSENT.format(question=q, passages=passages or "(none)"),
                           model=CFG.strong_llm, schema=VERIFY_ABSENT_SCHEMA, purpose="hard_verify")
    if v["answered"] != "no":
        return None
    return dict(question=q, expected_answer="The documents provided do not establish this.", facts=[], key_facts=[],
                gold=[], verify=dict(v, n_passages_checked=len(rows), terms=terms))


# ------------------------------------------------------------------------------------------- build
def load_hard() -> list[dict]:
    return [json.loads(l) for l in open(HARD_PATH)] if HARD_PATH.exists() else []


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=CFG.artifacts_dir.parent, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


async def build(kind: str, n: int = 25, seed: int = 13, show: int = 0) -> list[dict]:
    """Append accepted `kind` items to qa_hard.jsonl until it holds n of them."""
    tasks, chunks = _load_inputs()
    titles = {t["task_id"]: t["title"] or t["task_id"] for t in tasks}
    dev = {json.loads(l)["task_id"] for l in open(QA_PATH) if json.loads(l).get("split", "dev") == "dev"}
    have = load_hard()
    per_matter: dict[str, int] = {}
    for i in have:
        if i["kind"] == kind:
            per_matter[i["task_id"]] = per_matter.get(i["task_id"], 0) + 1
    used = {tuple(i["gold"]) for i in have} | {i["question"] for i in have}
    need = n - sum(per_matter.values())
    rng = random.Random(f"{seed}:{kind}")
    matters = sorted(set(chunks.task_id) - dev)
    rng.shuffle(matters)
    commit, accepted, tried = _commit(), [], 0

    async def candidates():
        for t in matters:
            tc = chunks[chunks.task_id == t]
            mrng = random.Random(f"{seed}:{kind}:{t}")
            if kind == "bridge":
                cands = [c for c in bridge_candidates(tc, mrng)
                         if not any(x in c["entity"] for x in EXCLUDED_LINKS.get(t, ()))][:4]
            elif kind == "aspect":
                cands = aspect_candidates(tc, mrng)[:4]
            else:
                cands = await _absent_questions(tc, titles.get(t, t), mrng)
            for c in cands:
                key = tuple(r.chunk_id for r in c["chunks"]) if "chunks" in c else c["question"]
                if key not in used:
                    yield t, tc, c

    async def one(t, tc, c):
        item = await (_bridge(c) if kind == "bridge" else _aspect(c) if kind == "aspect" else _absent(c, tc))
        return t, item

    gen = candidates()
    wave: list = []
    while need > 0:
        wave.clear()
        async for t, tc, c in gen:
            if per_matter.get(t, 0) + sum(1 for x in wave if x[0] == t) < PER_MATTER:
                wave.append((t, tc, c))
            if len(wave) >= min(WAVE, need * 2):
                break
        if not wave:
            break
        tried += len(wave)
        res = await asyncio.gather(*[one(*w) for w in wave], return_exceptions=True)
        quota = next((r for r in res if isinstance(r, llm.QuotaError)), None)
        for r in res:
            if isinstance(r, Exception):
                print("error:", repr(r)[:200], flush=True)
                continue
            t, item = r
            if item is None or need <= 0 or per_matter.get(t, 0) >= PER_MATTER:
                continue
            per_matter[t] = per_matter.get(t, 0) + 1
            k = sum(1 for i in have + accepted if i["kind"] == kind)
            item = dict(qid=f"hard::{kind}::{t}::{k}", task_id=t, split="hard", kind=kind, **item, built_with=commit)
            accepted.append(item)
            with open(HARD_PATH, "a") as f:
                f.write(json.dumps(item) + "\n")
            need -= 1
        print(f"{kind}: tried {tried}, accepted {len(accepted)} (still needed {max(need, 0)})", flush=True)
        if quota:  # accepted items are already saved: rerun after the reset to continue
            raise quota
    for it in accepted[:show]:
        print(json.dumps({k: it[k] for k in ("qid", "question", "expected_answer", "facts", "gold")}, indent=1))
    print(json.dumps(dict(kind=kind, tried=tried, accepted=len(accepted),
                          acceptance=round(len(accepted) / tried, 2) if tried else None,
                          matters=len({i["task_id"] for i in accepted}), llm=llm.STATS.snapshot()), indent=1))
    return accepted


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["bridge", "aspect", "absent"])
    ap.add_argument("--n", type=int, default=25, help="items per kind")
    ap.add_argument("--show", type=int, default=0, help="print this many accepted items per kind")
    a = ap.parse_args()

    async def main():  # one event loop: llm's semaphore is bound to the loop that first uses it
        for kind in a.kinds:
            await build(kind, a.n, show=a.show)
    asyncio.run(main())
