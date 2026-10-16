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
    status: str         # pending | correct | ambiguous | incorrect | retry | insufficient | done
    best: float
    hits: list
    strips: list


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
    if r == "direct" and state.get("task_id"):
        pass  # allowed: e.g. "what can you do?"
    return {"route": r, "route_reason": out.get("reason", ""), "log": _log(state, f"route={r}: {out.get('reason', '')}")}


def _new_subq(i: int, q: str) -> SubQ:
    return SubQ(id=i, q=q, query=q, hyde=None, rounds=0, status="pending", best=0.0, hits=[], strips=[])


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
        hits = R.search(s["query"], task_id=state.get("task_id"), dense_query=s.get("hyde"))
        s["hits"] = hits
        s["best"] = hits[0].score if hits else 0.0
        s["status"] = "retrieved"
    return {"subqs": subqs}


def _passages(hits, n=5, chars=900) -> str:
    return "\n\n".join(f"[{i}] ({h.doc_label()})\n{h.meta['chunk_text'][:chars]}" for i, h in enumerate(hits[:n], 1))


async def grade(state: State) -> dict:
    """CRAG retrieval evaluator: reranker-score thresholds; Claude only for the ambiguous band."""
    subqs = [dict(s) for s in state["subqs"]]
    log = list(state.get("log") or [])
    todo = []
    for s in subqs:
        if s["status"] != "retrieved":
            continue
        if s["best"] >= CFG.crag_high:
            s["status"], s["keep"] = "correct", [h for h in s["hits"] if h.score >= CFG.crag_low]
        elif s["best"] < CFG.crag_low:
            s["status"], s["keep"] = "incorrect", []
        else:
            todo.append(s)
    if todo:
        outs = await asyncio.gather(*[llm.complete(P.GRADE.format(question=s["q"], passages=_passages(s["hits"])),
                                                   system=P.SYSTEM, model=CFG.fast_llm, schema=P.GRADE_SCHEMA,
                                                   purpose="crag_grade") for s in todo], return_exceptions=True)
        for s, o in zip(todo, outs):
            rel = [] if isinstance(o, Exception) else [i for i in o.get("relevant", []) if 1 <= i <= len(s["hits"][:5])]
            s["keep"] = [s["hits"][i - 1] for i in rel]
            s["status"] = "ambiguous" if s["keep"] else "incorrect"
    for s in subqs:
        if s["status"] in ("correct", "ambiguous", "incorrect") and s.get("_logged_round") != s["rounds"]:
            log.append(f"crag[{s['id']}] round {s['rounds']}: {s['status']} (best rerank {s['best']:.3f})")
            s["_logged_round"] = s["rounds"]
    return {"subqs": subqs, "log": log}


async def refine(state: State) -> dict:
    """Knowledge refinement for correct/ambiguous; decide retry vs. insufficient for the rest."""
    R = retriever(state.get("variant", "ctx"))
    subqs = [dict(s) for s in state["subqs"]]
    for s in subqs:
        st = s["status"]
        if st in ("correct", "ambiguous") and s.get("keep"):
            new = R.strips(s["q"], s["keep"])
            seen = {(x["chunk_id"], x["text"]) for x in s["strips"]}
            s["strips"] = s["strips"] + [x for x in new if (x["chunk_id"], x["text"]) not in seen]
        can_retry = s["rounds"] < CFG.max_corrective_rounds
        if st == "correct":
            s["status"] = "done"
        elif st == "ambiguous":
            s["status"] = "retry" if can_retry else "done"
        elif st == "incorrect":
            s["status"] = "retry" if can_retry else ("done" if s["strips"] else "insufficient")
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
        s["rounds"] += 1
        if isinstance(o, Exception):
            s["status"] = "done" if s["strips"] else "insufficient"
            continue
        s["query"], s["hyde"], s["status"] = o["keywords"], o["hyde"], "pending"
        log.append(f"rewrite[{s['id']}] round {s['rounds']}: '{o['keywords']}' + HyDE")
    return {"subqs": subqs, "log": log}


def _evidence_by_subq(subqs) -> str:
    parts = []
    for s in subqs:
        body = "\n".join(f"  - {x['doc']}: {x['text'][:400]}" for x in s["strips"]) or "  (no evidence found)"
        parts.append(f"Sub-question: {s['q']} [{s['status']}]\n{body}")
    return "\n\n".join(parts)


async def sufficiency(state: State) -> dict:
    it = state.get("iteration", 0) + 1
    out = await llm.complete(P.SUFFICIENCY.format(question=state["question"], evidence=_evidence_by_subq(state["subqs"])),
                             system=P.SYSTEM, model=CFG.fast_llm, schema=P.SUFFICIENCY_SCHEMA, purpose="sufficiency")
    missing = [] if out.get("sufficient") else out.get("missing", [])[:3]
    known = {s["q"].lower() for s in state["subqs"]}
    new = [_new_subq(len(state["subqs"]) + i, q) for i, q in enumerate(m for m in missing if m.lower() not in known)]
    msg = f"sufficiency iter {it}: " + ("sufficient" if not new else "missing → " + " | ".join(s["q"] for s in new))
    return {"subqs": list(state["subqs"]) + new, "iteration": it, "log": _log(state, msg)}


async def generate(state: State) -> dict:
    evidence, index = [], {}
    for s in state["subqs"]:
        for x in s["strips"]:
            key = (x["chunk_id"], x["text"])
            if key not in index:
                index[key] = len(evidence) + 1
                evidence.append(x)
    ev_txt = "\n\n".join(f"[{i}] ({x['doc']})\n{x['text']}" for i, x in enumerate(evidence, 1)) or "(no evidence found)"
    insuff = [s["q"] for s in state["subqs"] if s["status"] == "insufficient"]
    insuff_txt = ("\nThe search found NO supporting evidence for these sub-questions; state that explicitly:\n"
                  + "\n".join(f"- {q}" for q in insuff) + "\n") if insuff else ""
    answer = await llm.complete(P.GENERATE.format(insufficient=insuff_txt, matter=state["matter"],
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
    calls0 = llm.STATS.calls
    state = State(question=question, task_id=task_id, matter=_matters().get(task_id, "whole antitrust corpus"),
                  policy=policy, variant=variant, log=[], subqs=[], iteration=0)
    with tracing.span("agent.answer", kind="AGENT", **{"input.value": question, "task_id": task_id or "corpus",
                                                       "policy": policy, "variant": variant}) as sp:
        out = await build_graph().ainvoke(state, {"recursion_limit": 60})
        res = dict(question=question, task_id=task_id, policy=policy, route=out.get("route"),
                   route_reason=out.get("route_reason"), answer=out.get("answer", ""), citations=out.get("citations", []),
                   evidence=out.get("evidence", []), log=out.get("log", []),
                   subqs=[dict(q=s["q"], status=s["status"], rounds=s["rounds"], best=round(s["best"], 3),
                               n_strips=len(s["strips"])) for s in out.get("subqs", [])],
                   llm_calls=llm.STATS.calls - calls0, seconds=round(time.perf_counter() - t0, 1))
        tracing.set_attrs(sp, **{"output.value": res["answer"], "route": res["route"], "llm_calls": res["llm_calls"]})
        return res


def answer_sync(*a, **kw):
    return llm.run(answer(*a, **kw))
