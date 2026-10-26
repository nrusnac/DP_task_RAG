"""C3-C5: retrieval ablations, CRAG threshold calibration, and generation eval (agent vs. baselines).

qa.jsonl has two splits (docs/05-evaluation.md): `dev` (the 110 questions that shaped the design) and `test` (matters
never used while building it). Calibration uses dev only; the generation eval defaults to test.

    python -m hrag.eval.run_eval retrieval                       # no LLM calls: every retrieval configuration, per split
    python -m hrag.eval.run_eval calibrate                       # CRAG thresholds from dev rerank-score distributions
    python -m hrag.eval.run_eval generation --tag test_v2 --resume          # whole test split: single_shot vs. agent
    python -m hrag.eval.run_eval generation --n 20 --systems agent:simple agent:complex agent   # Adaptive-RAG ablation
    python -m hrag.eval.run_eval negatives --n 40 --systems single_shot agent                   # unanswerable questions

Long runs stop cleanly at a subscription session limit (llm.QuotaError); rerun with --resume after the reset.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from .. import llm
from ..config import CFG
from ..obs import tracing
from .metrics import summarize

EVAL_DIR = CFG.artifacts_dir / "eval"
SPLITS = ("dev", "test")


def load_qa(path: Path | None = None, split: str | None = None) -> list[dict]:
    items = [json.loads(l) for l in open(path or EVAL_DIR / "qa.jsonl")]
    return [i for i in items if i.get("split", "dev") == split] if split else items


def available_variants() -> list[str]:
    return [v for v in ("raw", "header", "ctx") if (CFG.index_dir / f"dense_{v}.npy").exists()]


@lru_cache(maxsize=1)
def _areas() -> dict[str, str]:
    """task_id -> practice area (tasks.jsonl rows written before the corpus grew have no field: the main area)."""
    out = {}
    with open(CFG.artifacts_dir / "tasks.jsonl") as f:
        for line in f:
            t = json.loads(line)
            out[t["task_id"]] = t.get("practice_area", CFG.practice_area)
    return out


def other_matter(rng: random.Random, task_id: str, tasks: list[str]) -> str:
    """A different matter for a cross-matter negative, from the SAME practice area when possible: an antitrust
    question asked in another antitrust data room finds look-alike evidence, which is the hard case."""
    area = _areas()
    same = [t for t in tasks if t != task_id and area.get(t) == area.get(task_id)]
    return rng.choice(same or [t for t in tasks if t != task_id])


# ------------------------------------------------------------------------------------------- C3 retrieval
def retrieval(variants: list[str] | None = None) -> dict:
    from ..agent.graph import retriever
    labelled = [i for i in load_qa() if i["gold"]]
    # (mode, rerank, scoped, fusion): scoped = search only the matter's data room; unscoped = whole corpus (harder, and
    # shows cross-matter contamination: the share of top-5 hits that belong to OTHER matters). fusion: how the reranker
    # order is combined with the hybrid order ("replace" = reranker only, "rrf" = rank fusion of both).
    configs = [("bm25", False, True, None), ("dense", False, True, None), ("hybrid", False, True, None),
               ("hybrid", True, True, "replace"), ("hybrid", True, True, "rrf"),
               ("dense", False, False, None), ("hybrid", True, False, "rrf")]
    results = {}
    for v in variants or available_variants():
        R = retriever(v)
        for split in SPLITS:
            items = [i for i in labelled if i.get("split", "dev") == split]
            if not items:
                continue
            for mode, rr, scoped, fusion in configs:
                name = (f"{split} · {v}/{mode}{'+rerank' if rr else ''}{'(rrf)' if fusion == 'rrf' else ''}"
                        f"{'' if scoped else ' (corpus-wide)'}")
                t0 = time.perf_counter()
                rows, foreign = [], []
                for it in items:
                    hits = R.search(it["question"], task_id=it["task_id"] if scoped else None, mode=mode, rerank=rr,
                                    top=10, fusion=fusion)
                    rows.append(dict(ranked=[h.chunk_id for h in hits], gold=set(it["gold"]),
                                     top_score=max((h.score for h in hits), default=0.0)))
                    foreign.append(sum(h.meta["task_id"] != it["task_id"] for h in hits[:5]) / 5)
                results[name] = summarize(rows) | {"foreign@5": round(float(np.mean(foreign)), 3),
                                                   "ms_per_query": round((time.perf_counter() - t0) * 1000 / len(items))}
                print(name, results[name], flush=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "retrieval_results.json").write_text(json.dumps(results, indent=1))
    return results


# ------------------------------------------------------------------------------------------- CRAG calibration
def thresholds(pos: np.ndarray, neg: np.ndarray, tail: float | None = None) -> tuple[float, float]:
    """(tau_low, tau_high) from top rerank scores of answerable (pos) and unanswerable (neg) queries.

    tau_low  = the `tail` quantile of the positives: at most `tail` of answerable questions are graded incorrect
               outright (they still get the corrective rewrites before abstaining);
    tau_high = the 1-`tail` quantile of the negatives: at most `tail` of unanswerable ones are graded correct.
    The band in between goes to the LLM grader. Quantiles, not min/max: on main, a single positive at 0.095 set
    tau_low to 0.09, which let near-zero retrievals through to the grader."""
    tail = CFG.crag_tail if tail is None else tail
    tau_low = float(np.floor(np.quantile(pos, tail) * 100) / 100) if len(pos) else CFG.crag_low
    tau_high = float(np.ceil(np.quantile(neg, 1 - tail) * 100) / 100) if len(neg) else CFG.crag_high
    if tau_low > tau_high:  # classes don't overlap: no ambiguous band needed
        tau_low = tau_high = round((tau_low + tau_high) / 2, 2)
    return tau_low, tau_high


def calibrate(variant: str | None = None, seed: int = 11, split: str = "dev") -> dict:
    """Calibrate CRAG's reranker thresholds with POSITIVES and cross-matter NEGATIVES (dev split only, so the test
    split stays untouched by every tuning decision).

    The QA set only has answerable questions, and their top rerank scores saturate (most >= 0.95), so it can't tell
    a good retrieval from a bad one. Negatives come free: ask each question against a DIFFERENT matter's data room
    (same practice area when possible), where the answer cannot exist. Thresholds: see `thresholds`.
    """
    from ..agent.graph import retriever
    v = variant or available_variants()[-1]
    R = retriever(v)
    rng = random.Random(seed)
    tasks = sorted(set(R.task_ids))
    pos, neg = [], []
    for it in [i for i in load_qa(split=split) if i["gold"]]:
        hits = R.search(it["question"], task_id=it["task_id"], top=CFG.rerank_top)
        pos.append((max((h.score for h in hits), default=0.0), any(h.chunk_id in set(it["gold"]) for h in hits)))
        hits = R.search(it["question"], task_id=other_matter(rng, it["task_id"], tasks), top=CFG.rerank_top)
        neg.append(max((h.score for h in hits), default=0.0))
    ps = np.array([p[0] for p in pos]); ok = np.array([p[1] for p in pos]); ns = np.array(neg)
    good = ps[ok]  # positives where retrieval actually found the gold chunk
    tau_low, tau_high = thresholds(ps, ns)
    q = lambda a: [round(float(x), 3) for x in np.quantile(a, [0.1, 0.25, 0.5, 0.75, 0.9])] if len(a) else []  # noqa: E731
    buckets = []
    for lo, hi in [(0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, .95), (.95, .99), (.99, 1.01)]:
        mp, mn = (ps >= lo) & (ps < hi), (ns >= lo) & (ns < hi)
        buckets.append(dict(range=f"[{lo:.2f},{min(hi, 1):.2f})", positives=int(mp.sum()), negatives=int(mn.sum()),
                            gold_retrieved=round(float(ok[mp].mean()), 2) if mp.any() else None))
    out = dict(variant=v, split=split, tail=CFG.crag_tail, n_pos=len(pos), n_neg=len(neg),
               pos_gold_retrieved=round(float(ok.mean()), 3),
               pos_quantiles_p10_p90=q(ps), neg_quantiles_p10_p90=q(ns), buckets=buckets,
               suggested=dict(crag_high=tau_high, crag_low=tau_low),
               raw=dict(pos=[round(float(x), 4) for x in ps], pos_ok=[bool(x) for x in ok],
                        neg=[round(float(x), 4) for x in ns]),
               coverage=dict(correct_pos=round(float((good >= tau_high).mean()), 3) if len(good) else None,
                             incorrect_pos=round(float((ps < tau_low).mean()), 3),
                             correct_neg=round(float((ns >= tau_high).mean()), 3),
                             incorrect_neg=round(float((ns < tau_low).mean()), 3),
                             ambiguous_share=round(float(np.mean(np.r_[(ps >= tau_low) & (ps < tau_high),
                                                                       (ns >= tau_low) & (ns < tau_high)])), 3)),
               previous=dict(crag_high=CFG.crag_high, crag_low=CFG.crag_low))
    CFG.crag_high, CFG.crag_low = tau_high, tau_low  # apply immediately for the rest of this session
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "crag_calibration.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "raw"}, indent=1))
    return out


# ------------------------------------------------------------------------------------------- C4-C5 generation
ABSTAIN_MARKERS = ("do not establish", "does not establish", "not found in", "no evidence", "do not contain",
                   "does not contain", "not addressed in")
_CITATION = re.compile(r"\[\d+\]")


def abstained(answer: str | None) -> bool:
    """A whole-answer decline: opens with a decline AND cites nothing. ("Short answer: the documents do not establish
    X. They show Y [2]" is a partial answer, not an abstention: on the first test run 48 of 51 "abstentions" cited
    evidence.)"""
    a = answer or ""
    return any(m in a[:250].lower() for m in ABSTAIN_MARKERS) and not _CITATION.search(a)


def hedged(answer: str | None) -> bool:
    """Says somewhere that the documents don't establish something (tracks over-hedging)."""
    return any(m in (answer or "").lower() for m in ("do not establish", "does not establish"))


async def _run_system(system: str, it: dict, variant: str) -> dict:
    from ..agent import baseline
    from ..agent.graph import answer
    if system == "naive":
        return await baseline.naive(it["question"], it["task_id"])
    if system == "single_shot":
        return await baseline.single_shot(it["question"], it["task_id"], variant=variant)
    policy = system.split(":", 1)[1] if ":" in system else "adaptive"
    return await answer(it["question"], task_id=it["task_id"], policy=policy, variant=variant)


def _load_done(out_path: Path, resume: bool) -> dict[tuple[str, str], dict]:
    """resume=True: keep the successful records of an earlier (interrupted) run and drop its errors, so they are
    retried; resume=False: start a fresh file."""
    done = {}
    if resume and out_path.exists():
        for line in open(out_path):
            r = json.loads(line)
            if "error" not in r:
                done[(r["qid"], r["system"])] = r
        with open(out_path, "w") as f:
            for r in done.values():
                f.write(json.dumps(r, default=str) + "\n")
    else:
        out_path.unlink(missing_ok=True)
    return done


async def _run_pairs(pairs, work, out_path: Path, concurrency: int, n_total: int, done: int) -> int:
    """Run `work(it, system) -> record` over (item, system) pairs, appending each record to out_path. A QuotaError
    stops scheduling new pairs and is NOT recorded, so a resumed run picks those pairs up. Returns #pairs left."""
    sem, stop, n = asyncio.Semaphore(concurrency), asyncio.Event(), [done]
    left = [len(pairs)]

    async def one(it, system):
        async with sem:
            if stop.is_set():
                return
            try:
                rec = await work(it, system)
            except llm.QuotaError as e:
                if not stop.is_set():
                    print(f"STOPPED: {e} -- rerun with resume=True after the limit resets", flush=True)
                stop.set()
                return
            except Exception as e:  # noqa: BLE001
                rec = dict(qid=it["qid"], system=system, error=repr(e)[:300])
            n[0] += 1
            left[0] -= "error" not in rec
            print(f"[{n[0]}/{n_total}] {system} {it['qid']} -> {rec.get('verdict', rec.get('declined', rec.get('error')))}",
                  flush=True)
            with open(out_path, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")

    await asyncio.gather(*[one(it, s) for it, s in pairs])
    return left[0]


async def generation(n: int | None = None, systems=("single_shot", "agent"), seed: int = 3, variant: str | None = None,
                     concurrency: int = 3, judge_faith: bool = True, qids: list[str] | None = None,
                     tag: str | None = None, split: str | None = "test", resume: bool = False) -> dict:
    """n=None: the whole split. With split="dev" and n=30 this is the 30-question set used on main."""
    from .judge import judge_correctness, judge_faithfulness
    variant = variant or available_variants()[-1]
    items = load_qa(split=split)
    random.Random(seed).shuffle(items)
    items = [i for i in items if i["qid"] in set(qids)] if qids else items[:n] if n else items
    out_path = EVAL_DIR / f"gen_{tag or '_'.join(s.replace(':', '-') for s in systems)}.jsonl"
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    done = _load_done(out_path, resume)
    pairs = [(it, s) for it in items for s in systems if (it["qid"], s) not in done]

    async def work(it, system):
        with tracing.span("eval.item", kind="CHAIN", **{"eval.system": system, "eval.qid": it["qid"],
                                                        "input.value": it["question"]}):
            res = await _run_system(system, it, variant)
            corr = await judge_correctness(it, res["answer"])
            faith = await judge_faithfulness(res["answer"], res["evidence"]) if judge_faith else {}
            cited = {c["chunk_id"] for c in res.get("citations", [])} or {e["chunk_id"] for e in res["evidence"]}
            return dict(qid=it["qid"], system=system, kind=it["kind"], route=res.get("route"),
                        verdict=corr["verdict"], judge_reasoning=corr["reasoning"],
                        faithful=faith.get("faithful"), unsupported=faith.get("unsupported"),
                        abstained=abstained(res["answer"]), hedged=hedged(res["answer"]),
                        evidence_hits_gold=bool(set(it["gold"]) & {e["chunk_id"] for e in res["evidence"]})
                        if it["gold"] else None,
                        cited_gold=bool(set(it["gold"]) & cited) if it["gold"] else None,
                        n_evidence=len(res["evidence"]),
                        evidence_words=sum(len(e["text"].split()) for e in res["evidence"]),
                        llm_calls=res.get("llm_calls"), seconds=res.get("seconds"),
                        subqs=res.get("subqs"), log=res.get("log"), answer=res["answer"])

    print(f"{len(items)} questions x {len(systems)} systems: {len(done)} done earlier, {len(pairs)} to run", flush=True)
    left = await _run_pairs(pairs, work, out_path, concurrency, len(items) * len(systems), len(done))
    wanted = {(it["qid"], s) for it in items for s in systems}
    records = [r for r in map(json.loads, open(out_path)) if (r["qid"], r["system"]) in wanted] \
        if out_path.exists() else []
    summary = summarize_generation(records)
    (EVAL_DIR / f"summary_{out_path.stem}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    if left:
        print(f"INCOMPLETE: {left} of {len(wanted)} pairs still missing -- rerun with resume=True", flush=True)
    return summary


UNANSWERABLE_JUDGE = """The question below was asked against a legal matter's data room that does NOT contain the
answer (it belongs to a different matter). Did the answer correctly decline, i.e. state that the documents do not
establish the requested facts, without asserting specific facts, figures or terms as the answer?

Question: {question}

Answer:
{answer}"""
UNANSWERABLE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["declined", "reason"],
                       "properties": {"declined": {"type": "boolean"}, "reason": {"type": "string"}}}


async def negatives(n: int = 40, systems=("single_shot", "agent"), seed: int = 21, variant: str | None = None,
                    concurrency: int = 4, split: str | None = "test", tag: str | None = None,
                    resume: bool = False) -> dict:
    """Unanswerable questions: each QA question is asked against ANOTHER matter's data room (same practice area when
    possible). A correct system declines; an incorrect one answers from look-alike evidence (a hallucination with
    citations)."""
    from ..agent.graph import _matters
    variant = variant or available_variants()[-1]
    rng = random.Random(seed)
    items = [i for i in load_qa(split=split) if i["gold"]]
    rng.shuffle(items)
    tasks = sorted(_matters())
    items = [dict(i, task_id=other_matter(rng, i["task_id"], tasks)) for i in items[:n]]
    out_path = EVAL_DIR / f"unanswerable_{tag or variant}.jsonl"
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    done = _load_done(out_path, resume)
    pairs = [(it, s) for it in items for s in systems if (it["qid"], s) not in done]

    async def work(it, system):
        res = await _run_system(system, it, variant)
        j = await llm.complete(UNANSWERABLE_JUDGE.format(question=it["question"], answer=res["answer"]),
                               model=CFG.fast_llm, schema=UNANSWERABLE_SCHEMA, purpose="judge_unanswerable")
        return dict(qid=it["qid"], asked_in=it["task_id"], system=system, declined=bool(j["declined"]),
                    reason=j["reason"], llm_calls=res.get("llm_calls"), route=res.get("route"),
                    log=res.get("log"), answer=res["answer"])

    left = await _run_pairs(pairs, work, out_path, concurrency, len(items) * len(systems), len(done))
    wanted = {(it["qid"], s) for it in items for s in systems}
    recs = [r for r in map(json.loads, open(out_path)) if (r["qid"], r["system"]) in wanted] if out_path.exists() else []
    out = summarize_negatives(recs, systems)
    (EVAL_DIR / f"summary_{out_path.stem}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    if left:
        print(f"INCOMPLETE: {left} of {len(wanted)} pairs still missing -- rerun with resume=True", flush=True)
    return out


def summarize_negatives(recs: list[dict], systems) -> dict:
    out = {}
    for s in systems:
        rs = [r for r in recs if r["system"] == s and "error" not in r]
        if not rs:
            continue
        declined = float(np.mean([r["declined"] for r in rs]))
        calls = [r["llm_calls"] for r in rs if r.get("llm_calls") is not None]
        out[s] = dict(n=len(rs), errors=sum(1 for r in recs if r["system"] == s and "error" in r),
                      declined_rate=round(declined, 3), hallucination_rate=round(1 - declined, 3),
                      avg_llm_calls=round(float(np.mean(calls)), 2) if calls else None)
    return out


def summarize_generation(records: list[dict]) -> dict:
    out = {}
    for system in sorted({r["system"] for r in records}):
        rs = [r for r in records if r["system"] == system and "error" not in r]
        if not rs:
            continue
        mean = lambda xs: round(float(np.mean(xs)), 3) if xs else None  # noqa: E731
        out[system] = dict(
            n=len(rs), errors=sum(1 for r in records if r["system"] == system and "error" in r),
            pass_rate=mean([r["verdict"] == "pass" for r in rs]),
            pass_rate_fact=mean([r["verdict"] == "pass" for r in rs if r["kind"] == "fact"]),
            pass_rate_analysis=mean([r["verdict"] == "pass" for r in rs if r["kind"] == "analysis"]),
            faithful_rate=mean([bool(r["faithful"]) for r in rs if r["faithful"] is not None]),
            abstain_rate=mean([abstained(r["answer"]) for r in rs]),  # recomputed: older records used a looser rule
            hedge_rate=mean([hedged(r["answer"]) for r in rs]),
            evidence_has_gold=mean([r["evidence_hits_gold"] for r in rs if r["evidence_hits_gold"] is not None]),
            avg_evidence_words=mean([r["evidence_words"] for r in rs]),
            avg_llm_calls=mean([r["llm_calls"] for r in rs if r["llm_calls"] is not None]),
            avg_seconds=mean([r["seconds"] for r in rs if r["seconds"] is not None]),
            routes={k: sum(r["route"] == k for r in rs) for k in {r["route"] for r in rs}})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["retrieval", "calibrate", "generation", "negatives"])
    ap.add_argument("--n", type=int, default=None, help="questions (default: whole split; negatives: 40)")
    ap.add_argument("--systems", nargs="+", default=["single_shot", "agent"])
    ap.add_argument("--variant", default=None)
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--split", default=None, help="dev | test (calibrate: dev; generation/negatives: test)")
    ap.add_argument("--tag", default=None, help="output file tag")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run instead of starting over")
    a = ap.parse_args()
    if a.what == "retrieval":
        retrieval(a.variants)
    elif a.what == "calibrate":
        calibrate(a.variant, split=a.split or "dev")
    elif a.what == "negatives":
        asyncio.run(negatives(a.n or 40, a.systems, variant=a.variant, split=a.split or "test", tag=a.tag,
                              resume=a.resume))
    else:
        asyncio.run(generation(a.n, a.systems, variant=a.variant, split=a.split or "test", tag=a.tag,
                               resume=a.resume))
    print("llm:", llm.STATS.snapshot())
