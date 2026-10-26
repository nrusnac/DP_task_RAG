"""The agentic module: an Adaptive-RAG router + CRAG corrective loop, as a LangGraph state machine.

    route ──direct──▶ direct_answer ─────────────────────────────────────────────▶ END
      │ simple ─▶ init_simple ─┐
      └ complex ─▶ plan ───────┴▶ retrieve ─▶ grade ─▶ refine ─┬─▶ rewrite ─▶ retrieve      (CRAG corrective loop)
                                     ▲                         ├─▶ sufficiency ─┬─▶ retrieve (new sub-questions)
                                     │                         │                └─▶ generate ─▶ END
                                     └─────────────────────────┴─▶ generate ─▶ END

Every decision (route, CRAG verdict, rewrite, sufficiency) is recorded in state["log"] and traced as a span.
See docs/04-agentic-module.md for the rationale.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from functools import lru_cache
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .. import llm
from ..config import CFG
from ..obs import tracing
from ..retrieve.hybrid import HybridRetriever
from . import prompts as P


class SubQ(TypedDict, total=False):
    id: int
    q: str
    query: str          # current BM25/rerank query (original or rewritten keywords)
    hyde: str | None    # HyDE passage for the dense side after a corrective rewrite
    rounds: int         # corrective rounds used
    status: str         # pending | correct | ambiguous | incorrect | retry | done | weak | insufficient
                        # terminal: done (graded correct), weak (grader-kept chunks but never graded correct),
                        # insufficient (nothing kept, confirmed by the LLM grader in the last round)
    best: float
    hits: list
    chunks: list        # kept evidence: whole chunks, in the order they were kept


class State(TypedDict, total=False):
    question: str
    task_id: str | None
    matter: str
    policy: str          # adaptive | simple | complex (forced routes, for the ablation)
    variant: str         # index variant: raw | header | ctx (LangGraph drops undeclared keys)
    route: str
    route_reason: str
    subqs: list[SubQ]
    iteration: int
    answer: str
    evidence: list
    citations: list
    log: list[str]


@lru_cache(maxsize=2)
def retriever(variant: str = "ctx") -> HybridRetriever:
    return HybridRetriever(variant)


@lru_cache(maxsize=1)
def load_calibration() -> dict:
    """Use CRAG thresholds calibrated by `run_eval calibrate` (positives vs. cross-matter negatives) when available.
    HRAG_CRAG_HIGH / HRAG_CRAG_LOW environment variables take precedence."""
    import os
    path = CFG.artifacts_dir / "eval" / "crag_calibration.json"
    if path.exists() and not os.environ.get("HRAG_CRAG_HIGH"):
        sug = json.loads(path.read_text()).get("suggested", {})
        CFG.crag_high = float(sug.get("crag_high", CFG.crag_high))
        CFG.crag_low = float(sug.get("crag_low", CFG.crag_low))
    return dict(crag_high=CFG.crag_high, crag_low=CFG.crag_low)


@lru_cache(maxsize=1)
def _matters() -> dict[str, str]:
    out = {}
    with open(CFG.artifacts_dir / "tasks.jsonl") as f:
        for line in f:
            t = json.loads(line)
            out[t["task_id"]] = t["title"] or t["task_id"]
    return out


def _log(state: State, msg: str) -> list[str]:
    return (state.get("log") or []) + [msg]


# --------------------------------------------------------------------------------------------- nodes
async def route(state: State) -> dict:
    policy = state.get("policy", "adaptive")
    if policy != "adaptive":
        return {"route": policy, "route_reason": f"forced by policy={policy}", "log": _log(state, f"route={policy} (forced)")}
    out = await llm.complete(P.ROUTER.format(matter=state["matter"], question=state["question"]), system=P.SYSTEM,
                             model=CFG.fast_llm, schema=P.ROUTER_SCHEMA, purpose="router")
    r = out.get("route", "complex")
    return {"route": r, "route_reason": out.get("reason", ""), "log": _log(state, f"route={r}: {out.get('reason', '')}")}


def _new_subq(i: int, q: str) -> SubQ:
    return SubQ(id=i, q=q, query=q, hyde=None, rounds=0, status="pending", best=0.0, hits=[], chunks=[])


async def init_simple(state: State) -> dict:
    return {"subqs": [_new_subq(0, state["question"])], "iteration": 0}


async def plan(state: State) -> dict:
    prev = state.get("subqs") or []
    gaps = ""
    if prev:
        gaps = "\nAlready searched (do not repeat): " + "; ".join(s["q"] for s in prev)
    out = await llm.complete(P.PLAN.format(matter=state["matter"], question=state["question"], gaps=gaps),
                             system=P.SYSTEM, model=CFG.strong_llm, schema=P.PLAN_SCHEMA, purpose="plan")
    subs = [_new_subq(len(prev) + i, q) for i, q in enumerate(out["subquestions"][:5])]
    return {"subqs": prev + subs, "iteration": state.get("iteration", 0),
            "log": _log(state, "plan: " + " | ".join(s["q"] for s in subs))}


async def retrieve(state: State) -> dict:
    R = retriever(state.get("variant", "ctx"))
    subqs = [dict(s) for s in state["subqs"]]
    for s in subqs:
        if s["status"] != "pending":
            continue
        # BM25 gets the (rewritten) keywords; the dense side and the reranker get the sub-question itself
        hits = R.search(s["query"], task_id=state.get("task_id"), dense_query=s.get("hyde"), question=s["q"])
        s["hits"] = hits
        s["best"] = max((h.score for h in hits), default=0.0)  # with rank fusion, hits[0] isn't always the max
        s["status"] = "retrieved"
    return {"subqs": subqs}


def _passages(hits, n=5, chars=2500) -> str:  # ~a whole chunk: at 900 chars the grader missed clauses further down
    return "\n\n".join(f"[{i}] ({h.doc_label()})\n{h.meta['chunk_text'][:chars]}" for i, h in enumerate(hits[:n], 1))


async def grade(state: State) -> dict:
    """CRAG retrieval evaluator: reranker-score thresholds; Claude for the ambiguous band AND for a last round that
    scores below tau_low, so a sub-question is never given up on a score alone (on the test split, 3 answerable
    questions were abstained on with the answer in the top hits at scores 0.47-0.71: long multi-part questions score
    low against the passage that answers them)."""
    subqs = [dict(s) for s in state["subqs"]]
    log = list(state.get("log") or [])
    todo = []
    for s in subqs:
        if s["status"] != "retrieved":
            continue
        last = s["rounds"] >= CFG.max_corrective_rounds
        if s["best"] >= CFG.crag_high:  # confident: keep all top hits, exactly what single-shot would pass on
            s["status"], s["keep"] = "correct", s["hits"]
        elif s["best"] < CFG.crag_low and not last:
            s["status"], s["keep"] = "incorrect", []
        else:
            todo.append(s)
    if todo:
        outs = await asyncio.gather(*[llm.complete(P.GRADE.format(question=s["q"], passages=_passages(s["hits"])),
                                                   system=P.SYSTEM, model=CFG.fast_llm, schema=P.GRADE_SCHEMA,
                                                   purpose="crag_grade") for s in todo], return_exceptions=True)
        for s, o in zip(todo, outs):
            if isinstance(o, llm.QuotaError):
                raise o
            rel = [] if isinstance(o, Exception) else [i for i in o.get("relevant", []) if 1 <= i <= len(s["hits"][:5])]
            s["keep"] = [s["hits"][i - 1] for i in rel]
            s["status"] = "ambiguous" if s["keep"] else "incorrect"
    for s in subqs:
        if s["status"] in ("correct", "ambiguous", "incorrect") and s.get("_logged_round") != s["rounds"]:
            log.append(f"crag[{s['id']}] round {s['rounds']}: {s['status']} (best rerank {s['best']:.3f})")
            s["_logged_round"] = s["rounds"]
    return {"subqs": subqs, "log": log}


async def refine(state: State) -> dict:
    """Knowledge refinement at chunk level: keep the hits that passed the evaluator (all top hits when correct, the
    grader's picks when ambiguous) as WHOLE chunks; decide retry vs. a terminal status for the rest. Only a
    sub-question graded correct in some round ends 'done'; one that runs out of rounds with only grader-kept chunks
    ends 'weak'. Sentence strips were dropped: they cut the clause or table context the answer needs (docs/04)."""
    subqs = [dict(s) for s in state["subqs"]]
    for s in subqs:
        st = s["status"]
        if st in ("correct", "ambiguous") and s.get("keep"):
            seen = {x["chunk_id"] for x in s["chunks"]}
            s["chunks"] = s["chunks"] + [dict(chunk_id=h.chunk_id, doc=h.doc_label(), text=h.meta["chunk_text"],
                                              score=h.score) for h in s["keep"] if h.chunk_id not in seen]
        can_retry = s["rounds"] < CFG.max_corrective_rounds
        if st == "correct":
            s["status"] = "done"
        elif st == "ambiguous":
            s["status"] = "retry" if can_retry else "weak"
        elif st == "incorrect":
            s["status"] = "retry" if can_retry else ("weak" if s["chunks"] else "insufficient")
    return {"subqs": subqs}


async def rewrite(state: State) -> dict:
    subqs = [dict(s) for s in state["subqs"]]
    todo = [s for s in subqs if s["status"] == "retry"]
    outs = await asyncio.gather(*[llm.complete(
        P.REWRITE.format(question=s["q"], query=s["query"],
                         seen="\n".join(f"- {h.doc_label()} (score {h.score:.2f})" for h in s["hits"][:4]) or "- none"),
        system=P.SYSTEM, model=CFG.fast_llm, schema=P.REWRITE_SCHEMA, purpose="crag_rewrite") for s in todo],
        return_exceptions=True)
    log = list(state.get("log") or [])
    for s, o in zip(todo, outs):
        if isinstance(o, llm.QuotaError):
            raise o
        s["rounds"] += 1
        if isinstance(o, Exception):
            s["status"] = "weak" if s["chunks"] else "insufficient"
            continue
        s["query"], s["hyde"], s["status"] = o["keywords"], o["hyde"], "pending"
        log.append(f"rewrite[{s['id']}] round {s['rounds']}: '{o['keywords']}' + HyDE")
    return {"subqs": subqs, "log": log}


def _evidence_by_subq(subqs) -> str:
    parts = []
    for s in subqs:
        body = "\n".join(f"  - {x['doc']}: {x['text'][:500]}" for x in s["chunks"][:5]) or "  (no evidence found)"
        parts.append(f"Sub-question: {s['q']} [{s['status']}]\n{body}")
    return "\n\n".join(parts)


async def sufficiency(state: State) -> dict:
    it = state.get("iteration", 0) + 1
    out = await llm.complete(P.SUFFICIENCY.format(question=state["question"], evidence=_evidence_by_subq(state["subqs"])),
                             system=P.SYSTEM, model=CFG.fast_llm, schema=P.SUFFICIENCY_SCHEMA, purpose="sufficiency")
    missing = [] if out.get("sufficient") else out.get("missing", [])[:2]
    known = {s["q"].lower() for s in state["subqs"]}
    new = [_new_subq(len(state["subqs"]) + i, q) for i, q in enumerate(m for m in missing if m.lower() not in known)]
    msg = f"sufficiency iter {it}: " + ("sufficient" if not new else "missing → " + " | ".join(s["q"] for s in new))
    return {"subqs": list(state["subqs"]) + new, "iteration": it, "log": _log(state, msg)}


def _select_evidence(subqs, per_subq: int | None = None, max_total: int | None = None) -> list[dict]:
    """Evidence passed to generation: up to `per_subq` kept chunks of each sub-question (in the order kept, i.e. the
    retriever's order), taken round-robin across sub-questions up to `max_total`, each chunk once. A simple question
    graded correct thus gets the same top-8 chunks as single-shot. Grouped by sub-question."""
    per_subq = per_subq or CFG.gen_chunks_per_subq
    max_total = max_total or CFG.gen_max_evidence
    picked, seen = [], set()
    for rnd in range(per_subq):
        for si, s in enumerate(subqs):
            if rnd < len(s["chunks"]) and len(picked) < max_total and s["chunks"][rnd]["chunk_id"] not in seen:
                seen.add(s["chunks"][rnd]["chunk_id"])
                picked.append((si, rnd, s["chunks"][rnd]))
    return [x for _, _, x in sorted(picked, key=lambda p: (p[0], p[1]))]


def abstain_answer(subqs) -> str:
    return ("The documents provided do not establish the answer: the search of this matter's data room found no "
            "relevant evidence for:\n" + "\n".join(f"- {s['q']}" for s in subqs))


async def generate(state: State) -> dict:
    subqs = state["subqs"]
    if subqs and all(s["status"] == "insufficient" for s in subqs):  # forced abstain: nothing to answer from
        return {"answer": abstain_answer(subqs), "evidence": [], "citations": [],
                "log": _log(state, "generate: all sub-questions insufficient → abstain (no LLM call)")}
    evidence = _select_evidence(subqs)
    ev_txt = "\n\n".join(f"[{i}] ({x['doc']})\n{x['text']}" for i, x in enumerate(evidence, 1)) or "(no evidence found)"
    # 'weak' sub-questions get no warning: their chunks were judged relevant by the grader, and the earlier "indirect
    # evidence" note made the model decline answers its evidence supported
    insuff = [s["q"] for s in subqs if s["status"] == "insufficient"]
    notes = ""
    if insuff:
        notes += ("\nThe search of the documents found NO evidence for these sub-questions: say the documents provided "
                  "do not establish them and do not fill them from evidence about something else (general legal "
                  "knowledge may still go under Analysis:, labelled as such):\n"
                  + "\n".join(f"- {q}" for q in insuff) + "\n")
    answer = await llm.complete(P.GENERATE.format(insufficient=notes, matter=state["matter"],
                                                  question=state["question"], evidence=ev_txt),
                                system=P.GENERATE_SYSTEM, model=CFG.strong_llm, purpose="generate", thinking=False)
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer or "") if 1 <= int(n) <= len(evidence)})
    return {"answer": answer, "evidence": evidence,
            "citations": [dict(n=n, chunk_id=evidence[n - 1]["chunk_id"], doc=evidence[n - 1]["doc"]) for n in cited]}


async def direct_answer(state: State) -> dict:
    ans = await llm.complete(P.DIRECT.format(question=state["question"]), system=P.SYSTEM, model=CFG.fast_llm,
                             purpose="direct")
    return {"answer": ans, "evidence": [], "citations": [], "subqs": []}


# --------------------------------------------------------------------------------------------- edges
def after_route(state: State) -> str:
    return {"direct": "direct_answer", "simple": "init_simple"}.get(state["route"], "plan")


def after_refine(state: State) -> str:
    if any(s["status"] == "retry" for s in state["subqs"]):
        return "rewrite"
    if state["route"] == "complex" and state.get("iteration", 0) < CFG.max_sufficiency_iters:
        return "sufficiency"
    return "generate"


def after_sufficiency(state: State) -> str:
    return "retrieve" if any(s["status"] == "pending" for s in state["subqs"]) else "generate"


@lru_cache(maxsize=1)
def build_graph():
    g = StateGraph(State)
    for name, fn in [("route", route), ("init_simple", init_simple), ("plan", plan), ("retrieve", retrieve),
                     ("grade", grade), ("refine", refine), ("rewrite", rewrite), ("sufficiency", sufficiency),
                     ("generate", generate), ("direct_answer", direct_answer)]:
        g.add_node(name, fn)
    g.add_edge(START, "route")
    g.add_conditional_edges("route", after_route, ["direct_answer", "init_simple", "plan"])
    g.add_edge("init_simple", "retrieve")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_edge("grade", "refine")
    g.add_conditional_edges("refine", after_refine, ["rewrite", "sufficiency", "generate"])
    g.add_edge("rewrite", "retrieve")
    g.add_conditional_edges("sufficiency", after_sufficiency, ["retrieve", "generate"])
    g.add_edge("generate", END)
    g.add_edge("direct_answer", END)
    return g.compile()


# --------------------------------------------------------------------------------------------- public API
async def answer(question: str, task_id: str | None = None, policy: str = "adaptive", variant: str = "ctx") -> dict[str, Any]:
    t0 = time.perf_counter()
    ctr = llm.start_counter()
    load_calibration()
    state = State(question=question, task_id=task_id, matter=_matters().get(task_id, "whole corpus"),
                  policy=policy, variant=variant, log=[], subqs=[], iteration=0)
    with tracing.span("agent.answer", kind="AGENT", **{"input.value": question, "task_id": task_id or "corpus",
                                                       "policy": policy, "variant": variant}) as sp:
        out = await build_graph().ainvoke(state, {"recursion_limit": 60})
        res = dict(question=question, task_id=task_id, policy=policy, route=out.get("route"),
                   route_reason=out.get("route_reason"), answer=out.get("answer", ""), citations=out.get("citations", []),
                   evidence=out.get("evidence", []), log=out.get("log", []),
                   subqs=[dict(q=s["q"], status=s["status"], rounds=s["rounds"], best=round(s["best"], 3),
                               n_chunks=len(s["chunks"])) for s in out.get("subqs", [])],
                   llm_calls=ctr["calls"], seconds=round(time.perf_counter() - t0, 1))
        tracing.set_attrs(sp, **{"output.value": res["answer"], "route": res["route"], "llm_calls": res["llm_calls"]})
        return res


def answer_sync(*a, **kw):
    return llm.run(answer(*a, **kw))
