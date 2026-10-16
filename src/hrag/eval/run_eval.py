"""C3-C5: retrieval ablations, CRAG threshold calibration, and generation eval (agent vs. baselines).

    python -m hrag.eval.run_eval retrieval                       # no LLM calls: every retrieval configuration
    python -m hrag.eval.run_eval calibrate                       # CRAG thresholds from rerank-score distributions
    python -m hrag.eval.run_eval generation --n 40 --systems naive single_shot agent
    python -m hrag.eval.run_eval generation --n 20 --systems agent:simple agent:complex agent   # Adaptive-RAG ablation
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import numpy as np

from .. import llm
from ..config import CFG
from ..obs import tracing
from .metrics import summarize

EVAL_DIR = CFG.artifacts_dir / "eval"


def load_qa(path: Path | None = None) -> list[dict]:
    return [json.loads(l) for l in open(path or EVAL_DIR / "qa.jsonl")]


def available_variants() -> list[str]:
    return [v for v in ("raw", "header", "ctx") if (CFG.index_dir / f"dense_{v}.npy").exists()]


# ------------------------------------------------------------------------------------------- C3 retrieval
def retrieval(variants: list[str] | None = None) -> dict:
    from ..agent.graph import retriever
    items = [i for i in load_qa() if i["gold"]]
    # (mode, rerank, scoped): scoped = search only the matter's data room; unscoped = whole corpus (harder, and shows
    # cross-matter contamination: the share of top-5 hits that belong to OTHER matters)
    configs = [("bm25", False, True), ("dense", False, True), ("hybrid", False, True), ("hybrid", True, True),
               ("dense", False, False), ("hybrid", True, False)]
    results = {}
    for v in variants or available_variants():
        R = retriever(v)
        for mode, rr, scoped in configs:
            name = f"{v}/{mode}{'+rerank' if rr else ''}{'' if scoped else ' (corpus-wide)'}"
            t0 = time.perf_counter()
            rows, foreign = [], []
            for it in items:
                hits = R.search(it["question"], task_id=it["task_id"] if scoped else None, mode=mode, rerank=rr, top=10)
                rows.append(dict(ranked=[h.chunk_id for h in hits], gold=set(it["gold"]),
                                 top_score=hits[0].score if hits else 0.0))
                foreign.append(sum(h.meta["task_id"] != it["task_id"] for h in hits[:5]) / 5)
            results[name] = summarize(rows) | {"foreign@5": round(float(np.mean(foreign)), 3),
                                               "ms_per_query": round((time.perf_counter() - t0) * 1000 / len(items))}
            print(name, results[name], flush=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "retrieval_results.json").write_text(json.dumps(results, indent=1))
    return results


# ------------------------------------------------------------------------------------------- CRAG calibration
def calibrate(variant: str | None = None, seed: int = 11) -> dict:
    """Calibrate CRAG's reranker thresholds with POSITIVES and cross-matter NEGATIVES.

    The QA set only has answerable questions, and their top rerank scores saturate (most >= 0.95), so it can't tell
    a good retrieval from a bad one. Negatives come free: ask each question against a DIFFERENT matter's data room,
    where the answer cannot exist. Then:
      tau_high = lowest score at which a 'correct' verdict is >= 90% precise (positives with gold retrieved)
      tau_low  = highest score below which >= 90% of queries are negatives
    and the band in between goes to the LLM grader.
    """
    from ..agent.graph import retriever
    v = variant or available_variants()[-1]
    R = retriever(v)
    rng = random.Random(seed)
    tasks = sorted(set(R.task_ids))
    pos, neg = [], []
    for it in [i for i in load_qa() if i["gold"]]:
        hits = R.search(it["question"], task_id=it["task_id"], top=CFG.rerank_top)
        pos.append((hits[0].score if hits else 0.0, any(h.chunk_id in set(it["gold"]) for h in hits)))
        other = rng.choice([t for t in tasks if t != it["task_id"]])
        hits = R.search(it["question"], task_id=other, top=CFG.rerank_top)
        neg.append(hits[0].score if hits else 0.0)
    ps = np.array([p[0] for p in pos]); ok = np.array([p[1] for p in pos]); ns = np.array(neg)
    good = ps[ok]  # positives where retrieval actually found the gold chunk
    # Error-free band edges on the calibration set: below tau_low there is no answerable question (incorrect is
    # safe), above tau_high there is no unanswerable one (correct is safe); in between, the LLM grader decides.
    tau_low = float(np.floor(ps.min() * 100) / 100) if len(ps) else CFG.crag_low
    tau_high = float(np.ceil((ns.max() + 1e-9) * 100) / 100) if len(ns) else CFG.crag_high
    if tau_low > tau_high:  # classes don't overlap: no ambiguous band needed
        tau_low = tau_high = round((tau_low + tau_high) / 2, 2)
    q = lambda a: [round(float(x), 3) for x in np.quantile(a, [0.1, 0.25, 0.5, 0.75, 0.9])] if len(a) else []  # noqa: E731
    buckets = []
    for lo, hi in [(0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, .95), (.95, .99), (.99, 1.01)]:
        mp, mn = (ps >= lo) & (ps < hi), (ns >= lo) & (ns < hi)
        buckets.append(dict(range=f"[{lo:.2f},{min(hi, 1):.2f})", positives=int(mp.sum()), negatives=int(mn.sum()),
                            gold_retrieved=round(float(ok[mp].mean()), 2) if mp.any() else None))
    out = dict(variant=v, n_pos=len(pos), n_neg=len(neg), pos_gold_retrieved=round(float(ok.mean()), 3),
               pos_quantiles_p10_p90=q(ps), neg_quantiles_p10_p90=q(ns), buckets=buckets,
               suggested=dict(crag_high=tau_high, crag_low=tau_low),
               raw=dict(pos=[round(float(x), 4) for x in ps], pos_ok=[bool(x) for x in ok],
                        neg=[round(float(x), 4) for x in ns]),
               coverage=dict(correct_pos=round(float((good >= tau_high).mean()), 3) if len(good) else None,
                             incorrect_neg=round(float((ns < tau_low).mean()), 3),
                             ambiguous_share=round(float(np.mean(np.r_[(ps >= tau_low) & (ps < tau_high),
                                                                       (ns >= tau_low) & (ns < tau_high)])), 3)),
               previous=dict(crag_high=CFG.crag_high, crag_low=CFG.crag_low))
    CFG.crag_high, CFG.crag_low = tau_high, tau_low  # apply immediately for the rest of this session
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "crag_calibration.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------------------------------- C4-C5 generation
ABSTAIN_MARKERS = ("do not establish", "does not establish", "not found in", "no evidence", "do not contain",
                   "does not contain", "not addressed in")


async def _run_system(system: str, it: dict, variant: str) -> dict:
    from ..agent import baseline
    from ..agent.graph import answer
    if system == "naive":
        return await baseline.naive(it["question"], it["task_id"])
    if system == "single_shot":
        return await baseline.single_shot(it["question"], it["task_id"], variant=variant)
    policy = system.split(":", 1)[1] if ":" in system else "adaptive"
    return await answer(it["question"], task_id=it["task_id"], policy=policy, variant=variant)


async def generation(n: int = 40, systems=("single_shot", "agent"), seed: int = 3, variant: str | None = None,
                     concurrency: int = 3, judge_faith: bool = True) -> dict:
    from .judge import judge_correctness, judge_faithfulness
    variant = variant or available_variants()[-1]
    items = load_qa()
    random.Random(seed).shuffle(items)
    items = items[:n]
    sem = asyncio.Semaphore(concurrency)
    out_path = EVAL_DIR / f"gen_{'_'.join(s.replace(':', '-') for s in systems)}.jsonl"
    records = []

    async def one(it, system):
        async with sem:
            with tracing.span("eval.item", kind="CHAIN", **{"eval.system": system, "eval.qid": it["qid"],
                                                            "input.value": it["question"]}):
                try:
                    res = await _run_system(system, it, variant)
                    corr = await judge_correctness(it, res["answer"])
                    faith = await judge_faithfulness(res["answer"], res["evidence"]) if judge_faith else {}
                    cited = {c["chunk_id"] for c in res.get("citations", [])} or {e["chunk_id"] for e in res["evidence"]}
                    rec = dict(qid=it["qid"], system=system, kind=it["kind"], route=res.get("route"),
                               verdict=corr["verdict"], judge_reasoning=corr["reasoning"],
                               faithful=faith.get("faithful"), unsupported=faith.get("unsupported"),
                               abstained=any(m in (res["answer"] or "")[:250].lower() for m in ABSTAIN_MARKERS),
                               evidence_hits_gold=bool(set(it["gold"]) & {e["chunk_id"] for e in res["evidence"]})
                               if it["gold"] else None,
                               cited_gold=bool(set(it["gold"]) & cited) if it["gold"] else None,
                               n_evidence=len(res["evidence"]),
                               evidence_words=sum(len(e["text"].split()) for e in res["evidence"]),
                               llm_calls=res.get("llm_calls"), seconds=res.get("seconds"),
                               subqs=res.get("subqs"), log=res.get("log"), answer=res["answer"])
                except Exception as e:  # noqa: BLE001
                    rec = dict(qid=it["qid"], system=system, error=repr(e)[:300])
                records.append(rec)
                print(f"[{len(records)}/{len(items) * len(systems)}] {system} {it['qid']} -> "
                      f"{rec.get('verdict', rec.get('error'))}", flush=True)
                with open(out_path, "a") as f:
                    f.write(json.dumps(rec, default=str) + "\n")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path.unlink(missing_ok=True)
    await asyncio.gather(*[one(it, s) for it in items for s in systems])
    summary = summarize_generation(records)
    (EVAL_DIR / f"summary_{out_path.stem}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return summary


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
            abstain_rate=mean([r["abstained"] for r in rs]),
            evidence_has_gold=mean([r["evidence_hits_gold"] for r in rs if r["evidence_hits_gold"] is not None]),
            avg_evidence_words=mean([r["evidence_words"] for r in rs]),
            avg_llm_calls=mean([r["llm_calls"] for r in rs if r["llm_calls"] is not None]),
            avg_seconds=mean([r["seconds"] for r in rs if r["seconds"] is not None]),
            routes={k: sum(r["route"] == k for r in rs) for k in {r["route"] for r in rs}})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["retrieval", "calibrate", "generation"])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--systems", nargs="+", default=["single_shot", "agent"])
    ap.add_argument("--variant", default=None)
    ap.add_argument("--variants", nargs="+", default=None)
    a = ap.parse_args()
    if a.what == "retrieval":
        retrieval(a.variants)
    elif a.what == "calibrate":
        calibrate(a.variant)
    else:
        asyncio.run(generation(a.n, a.systems, variant=a.variant))
    print("llm:", llm.STATS.snapshot())
