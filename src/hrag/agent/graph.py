"""The agentic module: an Adaptive-RAG router + CRAG corrective loop, as a LangGraph state machine.

    route ──direct──▶ direct_answer ─────────────────────────────────────────────▶ END
      │ simple ─▶ init_simple ─┐
      └ complex ─▶ plan ───────┴▶ retrieve ─▶ grade ─▶ refine ─┬─▶ rewrite ─▶ retrieve      (CRAG corrective loop)
                                     ▲                         ├─▶ sufficiency ─┬─▶ retrieve (new sub-questions)
                                     │                         │                └─▶ generate ─▶ END
                                     └─────────────────────────┴─▶ generate ─▶ END

On both routes sub-question 0 is the question itself (on the complex route it is searched before planning, and its
hits ground the plan), so the agent's evidence contains single-shot's whenever that search is usable.
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
    seen: list          # chunk ids the LLM grader judged or that were kept: a rewrite round searches past them


class State(TypedDict, total=False):
    question: str
    task_id: str | None
    matter: str
    policy: str          # adaptive | simple | complex (forced routes, for the ablation)
    variant: str         # index variant: raw | header | ctx (LangGraph drops undeclared keys)
    reuse: dict | None   # eval only: single-shot's {evidence_ids, answer} for the same question (see generate)
    route: str
    route_reason: str
    subqs: list[SubQ]
    iteration: int
    answer: str
    evidence: list
    citations: list
    reused: bool
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
    return SubQ(id=i, q=q, query=q, hyde=None, rounds=0, status="pending", best=0.0, hits=[], chunks=[], seen=[])


async def init_simple(state: State) -> dict:
    return {"subqs": [_new_subq(0, state["question"])], "iteration": 0}


def _first_look(hits) -> str:
    return "\n".join(f"- {h.doc_label()}: {' '.join(h.meta['chunk_text'][:200].split())}" for h in hits) or "- (nothing)"


async def plan(state: State) -> dict:
    """Sub-question 0 is the question itself, searched before planning (retrieval only, no LLM call). Its hits show the
    planner what this data room holds, so the plan uses the matter's names and documents; and its top chunks go to
    generation first (_select_evidence). On test_v2 the planner, writing blind, produced textbook sub-questions and
    never searched the question, and lost a question single-shot passed (docs/05, "Second test run")."""
    question = state["question"]
    s0 = _new_subq(0, question)
    hits = retriever(state.get("variant", "ctx")).search(question, task_id=state.get("task_id"))
    s0.update(hits=hits, best=max((h.score for h in hits), default=0.0), status="retrieved")
    out = await llm.complete(P.PLAN.format(matter=state["matter"], question=question, found=_first_look(hits)),
                             system=P.SYSTEM, model=CFG.strong_llm, schema=P.PLAN_SCHEMA, purpose="plan")
    qs = [q for q in out["subquestions"] if q.strip().lower() != question.strip().lower()][:4]
    subs = [_new_subq(i, q) for i, q in enumerate(qs, 1)]
    return {"subqs": [s0] + subs, "iteration": state.get("iteration", 0),
            "log": _log(state, "plan: (the question itself) | " + " | ".join(s["q"] for s in subs))}


async def retrieve(state: State) -> dict:
    R = retriever(state.get("variant", "ctx"))
    subqs = [dict(s) for s in state["subqs"]]
    for s in subqs:
        if s["status"] != "pending":
            continue
        # BM25 gets the (rewritten) keywords; the dense side and the reranker get the sub-question itself. A rewrite
        # round searches past the chunks already judged or kept, to find NEW evidence: with the same chunks back, 27 of
        # 33 retried sub-questions on test_v2 got the identical best score in every round
        hits = R.search(s["query"], task_id=state.get("task_id"), dense_query=s.get("hyde"), question=s["q"],
                        exclude=set(s.get("seen") or []))
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
    low against the passage that answers them).

    In round 0 the grader is a gate, not a filter: when it finds anything relevant, the sub-question keeps all its top
    hits, as when it is confident (keeping only the grader's picks left the agent with less evidence than single-shot,
    e.g. 481 vs 1,835 words). A rewrite round sees only new chunks and adds just the ones the grader accepts."""
    subqs = [dict(s) for s in state["subqs"]]
    log = list(state.get("log") or [])
    todo = []
    for s in subqs:
        if s["status"] != "retrieved":
            continue
        last = s["rounds"] >= CFG.max_corrective_rounds
        if s["rounds"] == 0 and s["best"] >= CFG.crag_high:  # confident: keep all top hits, as single-shot does
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
            s["seen"] = list(dict.fromkeys((s.get("seen") or []) + [h.chunk_id for h in s["hits"][:5]]))
            if not rel:
                s["status"], s["keep"] = "incorrect", []
            elif s["rounds"] == 0:
                s["status"], s["keep"] = "ambiguous", s["hits"]
            else:
                s["status"] = "correct" if s["best"] >= CFG.crag_high else "ambiguous"
                s["keep"] = [s["hits"][i - 1] for i in rel]
    for s in subqs:
        if s["status"] in ("correct", "ambiguous", "incorrect") and s.get("_logged_round") != s["rounds"]:
            log.append(f"crag[{s['id']}] round {s['rounds']}: {s['status']} (best rerank {s['best']:.3f}"
                       f"{', new chunks only' if s['rounds'] else ''})")
            s["_logged_round"] = s["rounds"]
    return {"subqs": subqs, "log": log}


async def refine(state: State) -> dict:
    """Knowledge refinement at chunk level: keep the hits that passed the evaluator as WHOLE chunks (see grade); decide
    retry vs. a terminal status for the rest. Only a sub-question graded correct in some round ends 'done'; one that
    runs out of rounds with kept chunks but never graded correct ends 'weak'. Sentence strips were dropped: they cut
    the clause or table context the answer needs (docs/04)."""
    subqs = [dict(s) for s in state["subqs"]]
    for s in subqs:
        st = s["status"]
        if st in ("correct", "ambiguous") and s.get("keep"):
            have = {x["chunk_id"] for x in s["chunks"]}
            s["chunks"] = s["chunks"] + [dict(chunk_id=h.chunk_id, doc=h.doc_label(), text=h.meta["chunk_text"],
                                              score=h.score) for h in s["keep"] if h.chunk_id not in have]
            s["seen"] = list(dict.fromkeys((s.get("seen") or []) + [x["chunk_id"] for x in s["chunks"]]))
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


def _evidence_by_subq(subqs, n: int = 4, chars: int = 1200) -> str:
    parts = []
    for s in subqs:
        body = "\n".join(f"  - {x['doc']}: {x['text'][:chars]}" for x in s["chunks"][:n]) or "  (no evidence found)"
        parts.append(f"Sub-question: {s['q']} [{s['status']}]\n{body}")
    return "\n\n".join(parts)


async def sufficiency(state: State) -> dict:
    """Missing parts AND unstated links (an entity named in one passage whose relation to another is needed). Sonnet
    with ~1,200 characters per chunk: on test_v2, Haiku with 500-character excerpts called the evidence sufficient
    although the link the answer turned on (a nominee's former employer -> a fund investor) was never searched."""
    it = state.get("iteration", 0) + 1
    out = await llm.complete(P.SUFFICIENCY.format(question=state["question"], evidence=_evidence_by_subq(state["subqs"])),
                             system=P.SYSTEM, model=CFG.strong_llm, schema=P.SUFFICIENCY_SCHEMA, purpose="sufficiency")
    missing = [] if out.get("sufficient") else out.get("missing", [])[:2]
    known = {s["q"].lower() for s in state["subqs"]}
    new = [_new_subq(len(state["subqs"]) + i, q) for i, q in enumerate(m for m in missing if m.lower() not in known)]
    msg = f"sufficiency iter {it}: " + ("sufficient" if not new else "missing → " + " | ".join(s["q"] for s in new))
    return {"subqs": list(state["subqs"]) + new, "iteration": it, "log": _log(state, msg)}


def _select_evidence(subqs, per_subq: int | None = None, max_total: int | None = None,
                     lead: int | None = None) -> list[dict]:
    """Evidence passed to generation, grouped by sub-question, each chunk once. Sub-question 0 is the question itself
    on both routes: its first `lead` kept chunks (= single-shot's top 8) come first, so the agent's evidence contains
    single-shot's whenever that search is usable. Then up to `per_subq` more chunks of each sub-question (sub-question
    0's later chunks, from a retry, included) are taken round-robin in the order kept, up to `max_total`."""
    per_subq = per_subq or CFG.gen_chunks_per_subq
    max_total = max_total or CFG.gen_max_evidence
    lead = CFG.rerank_top if lead is None else lead
    picked, seen = [], set()

    def take(si: int, rank: int, x: dict) -> None:
        if len(picked) < max_total and x["chunk_id"] not in seen:
            seen.add(x["chunk_id"])
            picked.append((si, rank, x))

    if not subqs:
        return []
    for rank, x in enumerate(subqs[0]["chunks"][:lead]):
        take(0, rank, x)
    queues = [(0, lead, subqs[0]["chunks"][lead:lead + per_subq])] + \
             [(si, 0, s["chunks"][:per_subq]) for si, s in enumerate(subqs[1:], 1)]
    for rnd in range(per_subq):
        for si, offset, q in queues:
            if rnd < len(q):
                take(si, offset + rnd, q[rnd])
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
    # Eval only (run_eval, docs/05 "Shared-path reuse"): when the evidence is single-shot's, chunk for chunk, and no note
    # is added, the generation prompt is byte-identical to single-shot's, so its answer is a sample of exactly this
    # call. Reusing it removes pure sampling noise from the paired comparison and saves the call.
    reuse = state.get("reuse") or {}
    reused = bool(evidence and not notes and reuse.get("answer")
                  and [x["chunk_id"] for x in evidence] == list(reuse.get("evidence_ids") or []))
    if reused:
        answer = reuse["answer"]
    else:
        answer = await llm.complete(P.GENERATE.format(insufficient=notes, matter=state["matter"],
                                                      question=state["question"], evidence=ev_txt),
                                    system=P.GENERATE_SYSTEM, model=CFG.strong_llm, purpose="generate", thinking=False)
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer or "") if 1 <= int(n) <= len(evidence)})
    out = {"answer": answer, "evidence": evidence, "reused": reused,
           "citations": [dict(n=n, chunk_id=evidence[n - 1]["chunk_id"], doc=evidence[n - 1]["doc"]) for n in cited]}
    if reused:
        out["log"] = _log(state, "generate: same evidence and prompt as single-shot -> its answer reused (eval)")
    return out


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
async def answer(question: str, task_id: str | None = None, policy: str = "adaptive", variant: str = "ctx",
                 reuse: dict | None = None) -> dict[str, Any]:
    """reuse (eval only): single-shot's {"evidence_ids", "answer"} for this question, see generate()."""
    t0 = time.perf_counter()
    ctr = llm.start_counter()
    load_calibration()
    state = State(question=question, task_id=task_id, matter=_matters().get(task_id, "whole corpus"),
                  policy=policy, variant=variant, reuse=reuse, log=[], subqs=[], iteration=0)
    with tracing.span("agent.answer", kind="AGENT", **{"input.value": question, "task_id": task_id or "corpus",
                                                       "policy": policy, "variant": variant}) as sp:
        out = await build_graph().ainvoke(state, {"recursion_limit": 60})
        res = dict(question=question, task_id=task_id, policy=policy, route=out.get("route"),
                   route_reason=out.get("route_reason"), answer=out.get("answer", ""), citations=out.get("citations", []),
                   evidence=out.get("evidence", []), reused=bool(out.get("reused")), log=out.get("log", []),
                   subqs=[dict(q=s["q"], status=s["status"], rounds=s["rounds"], best=round(s["best"], 3),
                               n_chunks=len(s["chunks"])) for s in out.get("subqs", [])],
                   llm_calls=ctr["calls"], seconds=round(time.perf_counter() - t0, 1))
        tracing.set_attrs(sp, **{"output.value": res["answer"], "route": res["route"], "llm_calls": res["llm_calls"]})
        return res


def answer_sync(*a, **kw):
    return llm.run(answer(*a, **kw))
