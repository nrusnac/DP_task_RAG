"""C3-C5: retrieval ablations, CRAG threshold calibration, and generation eval (agent vs. baselines).

qa.jsonl has two splits (docs/05-evaluation.md): `dev` (the 110 questions that shaped the original design) and `test`
(now a regression set after its failures motivated fixes). Calibration uses dev; generation defaults to test.
qa_hard.jsonl holds the post-freeze two-step-link and multi-document hard sets (build_hard.py).

    python -m hrag.eval.run_eval retrieval                       # no LLM calls: every retrieval configuration, per split
    python -m hrag.eval.run_eval calibrate                       # CRAG thresholds from dev rerank-score distributions
    python -m hrag.eval.run_eval generation --tag test_v3 --resume          # whole test split: single_shot vs. agent
    python -m hrag.eval.run_eval hard --tag hard_v2 --resume                # strict hard sets
    python -m hrag.eval.run_eval report --tags test_v3 hard_v2              # comparison tables, no LLM calls
    python -m hrag.eval.run_eval generation --n 20 --systems agent:simple agent:complex agent   # Adaptive-RAG ablation
    python -m hrag.eval.run_eval negatives --n 40 --systems single_shot agent                   # cross-matter negatives

Long runs stop cleanly at a subscription session limit (llm.QuotaError); rerun with --resume after the reset.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
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
HARD_KINDS = ("bridge", "aspect")
HARD_TARGET_PER_KIND = 25


def load_qa(path: Path | None = None, split: str | None = None) -> list[dict]:
    items = [json.loads(l) for l in open(path or EVAL_DIR / "qa.jsonl")]
    return [i for i in items if i.get("split", "dev") == split] if split else items


def available_variants() -> list[str]:
    return [v for v in ("raw", "header", "ctx") if (CFG.index_dir / f"dense_{v}.npy").exists()]


def default_variant() -> str:
    """The branch is evaluated on header (decision #44). Fall back only when that index was not built."""
    variants = available_variants()
    if not variants:
        raise FileNotFoundError(f"no indexes found in {CFG.index_dir}; run hrag.index.build first")
    return CFG.default_variant if CFG.default_variant in variants else variants[-1]


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
    v = variant or default_variant()
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
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "crag_calibration.json").write_text(json.dumps(out, indent=1))
    from ..agent.graph import load_calibration
    load_calibration.cache_clear()
    print(json.dumps({k: v for k, v in out.items() if k != "raw"}, indent=1))
    return out


# ------------------------------------------------------------------------------------------- C4-C5 generation
ABSTAIN_MARKERS = ("do not establish", "does not establish", "not found in", "no evidence", "do not contain",
                   "does not contain", "not addressed in")
_CITATION = re.compile(r"\[(\d+)\]")


def abstained(answer: str | None) -> bool:
    """A whole-answer decline: opens with a decline AND cites nothing. ("Short answer: the documents do not establish
    X. They show Y [2]" is a partial answer, not an abstention: on the first test run 48 of 51 "abstentions" cited
    evidence.)"""
    a = answer or ""
    return any(m in a[:250].lower() for m in ABSTAIN_MARKERS) and not _CITATION.search(a)


def hedged(answer: str | None) -> bool:
    """Says somewhere that the documents don't establish something (tracks over-hedging)."""
    return any(m in (answer or "").lower() for m in ("do not establish", "does not establish"))


def key_fact_recall(answer: str | None, facts: list[str] | None) -> float | None:
    """Share of the item's verbatim key facts that appear in the answer (normalised as in the silver labels, so
    "1,242" matches "1242"): a free, reproducible correctness signal. None when the item has no key facts."""
    from .build_qa import _norm, _variants
    facts = [f for f in (facts or []) if f and len(f.strip()) >= 2]
    if not facts:
        return None
    a = _norm(answer or "")
    a += " " + re.sub(r"(?<=\d),(?=\d{3})", "", a)
    return round(sum(any(v in a for v in _variants(f)) for f in facts) / len(facts), 3)


@lru_cache(maxsize=1)
def _chunk_lookup() -> dict[str, dict]:
    """chunk_id -> {chunk_id, doc, text}, labelled exactly as the systems show chunks to the generator."""
    import pandas as pd
    df = pd.read_parquet(CFG.artifacts_dir / "chunks.parquet", columns=["chunk_id", "doc_id", "section", "kind", "text"])
    return {r.chunk_id: dict(chunk_id=r.chunk_id, doc=f"{r.doc_id} § {r.section or r.kind}", text=r.text)
            for r in df.itertuples()}


def passages(chunk_ids: list[str]) -> list[dict]:
    lookup = _chunk_lookup()
    return [lookup[c] for c in dict.fromkeys(chunk_ids) if c in lookup]


async def _run_system(system: str, it: dict, variant: str, reuse: dict | None = None) -> dict:
    from ..agent import baseline
    from ..agent.graph import answer
    if system == "naive":
        return await baseline.naive(it["question"], it["task_id"])
    if system == "single_shot":
        return await baseline.single_shot(it["question"], it["task_id"], variant=variant)
    policy = system.split(":", 1)[1] if ":" in system else "adaptive"
    return await answer(it["question"], task_id=it["task_id"], policy=policy, variant=variant, reuse=reuse)


def _sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


@lru_cache(maxsize=1)
def _source_sha() -> str:
    """Fingerprint the implementation that produces and judges answers, including uncommitted edits."""
    root = Path(__file__).resolve().parents[2]
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def _file_set_sha(paths: list[Path]) -> str:
    """Hash named files/directories with portable relative names, so moved runs can still be resumed safely."""
    h = hashlib.sha256()
    for root in paths:
        h.update(root.name.encode())
        files = sorted(p for p in root.rglob("*") if p.is_file()) if root.is_dir() else [root]
        for path in files:
            name = str(path.relative_to(root)) if root.is_dir() else path.name
            h.update(name.encode())
            if not path.exists():
                h.update(b"<missing>")
                continue
            with open(path, "rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(block)
        if not root.exists():
            h.update(b"<missing>")
    return h.hexdigest()


def _retriever_sha(variant: str) -> str:
    """Fingerprint the corpus and exact dense/sparse indexes consumed by HybridRetriever."""
    inputs = [CFG.artifacts_dir / "chunks.parquet", CFG.index_dir / f"dense_{variant}.npy",
              CFG.index_dir / f"bm25_{variant}"]
    if variant == "ctx":
        inputs.append(CFG.artifacts_dir / "contexts.jsonl")
    return _file_set_sha(inputs)


def _manifest_path(tag: str) -> Path:
    return EVAL_DIR / f"manifest_{tag}.json"


def _prepare_run(tag: str, kind: str, variant: str, systems: list[str], items: list[dict], resume: bool,
                 post: bool = True) -> None:
    """Bind resumable outputs to their inputs and implementation. A fresh run clears every derived sidecar; a resume
    refuses to combine answers or judgments produced under another configuration."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = _manifest_path(tag)
    gen_path = EVAL_DIR / f"gen_{tag}.jsonl"
    from ..agent.graph import load_calibration
    calibration = load_calibration(variant)
    comparison_required = {"agent", "single_shot"} <= set(systems)
    spec = dict(version=2, tag=tag, kind=kind, variant=variant, systems=systems,
                items_sha256=_sha(items), n_items=len(items), source_sha256=_source_sha(),
                retriever_sha256=_retriever_sha(variant), embed_model=CFG.embed_model,
                rerank_model=CFG.rerank_model, fast_llm=CFG.fast_llm, strong_llm=CFG.strong_llm,
                crag_high=calibration["crag_high"], crag_low=calibration["crag_low"],
                dense_k=CFG.dense_k, bm25_k=CFG.bm25_k, rrf_k=CFG.rrf_k,
                rerank_candidates=CFG.rerank_candidates, rerank_top=CFG.rerank_top,
                rerank_fusion=CFG.rerank_fusion, post_required=comparison_required)
    derived = [gen_path, EVAL_DIR / f"rejudge_{tag}.jsonl", EVAL_DIR / f"pairwise_{tag}.jsonl",
               EVAL_DIR / f"summary_gen_{tag}.json", EVAL_DIR / f"compare_{tag}.json"]
    if not resume:
        for path in derived + [manifest_path]:
            path.unlink(missing_ok=True)
    elif any(path.exists() for path in derived) and not manifest_path.exists():
        legacy = [path.name for path in derived if path.exists()]
        raise RuntimeError(f"cannot safely resume {tag}: {legacy} have no run manifest; use a new tag")
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        mismatch = {k: (old.get(k), v) for k, v in spec.items() if old.get(k) != v}
        if mismatch:
            raise RuntimeError(f"cannot resume {tag}: run configuration changed: {mismatch}; use a new tag")
        old["post_requested"] = bool(old.get("post_requested") or post)
        manifest_path.write_text(json.dumps(old, indent=1))
    else:
        manifest_path.write_text(json.dumps(spec | {"post_requested": post, "post_complete": False,
                                                      "complete": False, "stage": "answers"}, indent=1))


def _update_manifest(tag: str, **state) -> None:
    path = _manifest_path(tag)
    manifest = json.loads(path.read_text())
    manifest.update(state)
    path.write_text(json.dumps(manifest, indent=1))


def _completion_state(left: int, post: bool, systems: list[str], rejudge_left: int | None,
                      pairwise_left: int | None) -> tuple[bool, bool, str]:
    """Return (run complete, post complete, stage); comparisons always require both post-passes."""
    comparison_required = {"agent", "single_shot"} <= set(systems)
    post_complete = comparison_required and post and not rejudge_left and not pairwise_left
    complete = not left and (not comparison_required or post_complete)
    stage = "complete" if complete else "post_skipped" if not left and comparison_required and not post else "incomplete"
    return complete, post_complete, stage


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


def _ordered(systems) -> list[str]:
    """single_shot first: on each question the agent then runs with single-shot's answer available for reuse."""
    return sorted(systems, key=lambda s: s != "single_shot")


async def _run_items(items: list[dict], systems: list[str], work, out_path: Path, done: dict,
                     concurrency: int) -> int:
    """Run each item through `systems` in order, `work(it, system, prev) -> record`, where prev holds this item's
    earlier records by system (from this run or a resumed one). Each record is appended to out_path. A QuotaError stops
    scheduling and is NOT recorded, so a resumed run picks the pair up. Returns #pairs left."""
    sem, stop = asyncio.Semaphore(concurrency), asyncio.Event()
    n_total = len(items) * len(systems)
    n, left = [len(done)], [sum((it["qid"], s) not in done for it in items for s in systems)]

    async def one(it):
        async with sem:
            prev = {}
            for system in systems:
                if stop.is_set():
                    return
                if (it["qid"], system) in done:
                    prev[system] = done[(it["qid"], system)]
                    continue
                try:
                    rec = await work(it, system, prev)
                except llm.QuotaError as e:
                    if not stop.is_set():
                        print(f"STOPPED: {e} -- rerun with resume=True after the limit resets", flush=True)
                    stop.set()
                    return
                except Exception as e:  # noqa: BLE001
                    rec = dict(qid=it["qid"], system=system, error=repr(e)[:300])
                prev[system] = rec
                n[0] += 1
                left[0] -= "error" not in rec
                shown = next((rec[k] for k in ("verdict", "reference_verdict", "declined", "error") if k in rec), "")
                print(f"[{n[0]}/{n_total}] {system} {it['qid']} -> {shown}{' (reused)' if rec.get('reused') else ''}",
                      flush=True)
                with open(out_path, "a") as f:
                    f.write(json.dumps(rec, default=str) + "\n")

    await asyncio.gather(*[one(it) for it in items])
    return left[0]


async def _append_all(jobs: list, out_path: Path, concurrency: int = 3) -> int:
    """Run zero-argument coroutine functions, appending each returned record to out_path; a QuotaError stops
    scheduling (rerun to continue), other errors are printed and retried on the next run. Returns #jobs left."""
    sem, stop, left = asyncio.Semaphore(concurrency), asyncio.Event(), [len(jobs)]

    async def one(job):
        async with sem:
            if stop.is_set():
                return
            try:
                rec = await job()
            except llm.QuotaError as e:
                if not stop.is_set():
                    print(f"STOPPED: {e} -- rerun after the limit resets", flush=True)
                stop.set()
                return
            except Exception as e:  # noqa: BLE001
                print("error:", repr(e)[:200], flush=True)
                return
            left[0] -= 1
            with open(out_path, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")

    await asyncio.gather(*[one(j) for j in jobs])
    return left[0]


# judged fields: when the agent reuses single-shot's answer it gets the same judgments
JUDGED = ("verdict", "judge_reasoning", "claims", "n_claims", "claim_support", "material_errors", "faithful",
          "unsupported", "reference_verdict", "reference_reasoning", "fact_coverage", "declined", "decline_reason")


def _with_reused(rec: dict, ss: dict) -> dict:
    """A reused answer IS single-shot's answer, so it gets single-shot's judgments; its cost still counts the generation
    call the agent makes in production (reuse is an eval device, not a saving)."""
    return rec | {k: ss[k] for k in JUDGED if k in ss} | dict(
        llm_calls=(rec.get("llm_calls") or 0) + 1,
        seconds=round((rec.get("seconds") or 0) + (ss.get("gen_seconds") or 0), 1))


def _reuse(prev: dict) -> dict | None:
    ss = prev.get("single_shot")
    if ss and "error" not in ss and ss.get("evidence_ids"):
        return dict(evidence_ids=ss["evidence_ids"], answer=ss["answer"])
    return None


def _record(it: dict, system: str, res: dict) -> dict:
    ev_ids = [e["chunk_id"] for e in res["evidence"]]
    gold = set(it.get("gold") or [])
    cited = {c["chunk_id"] for c in res.get("citations", [])}
    cited |= {ev_ids[int(n) - 1] for n in _CITATION.findall(res.get("answer") or "")
              if 1 <= int(n) <= len(ev_ids)}
    return dict(qid=it["qid"], task_id=it["task_id"], system=system, kind=it["kind"], route=res.get("route"),
                abstained=abstained(res["answer"]), hedged=hedged(res["answer"]),
                key_fact_recall=key_fact_recall(res["answer"], it.get("key_facts")),
                evidence_hits_gold=bool(gold & set(ev_ids)) if gold else None,
                all_gold_in_evidence=gold <= set(ev_ids) if gold and it.get("split") == "hard" else None,
                cited_gold=bool(gold & cited) if gold else None,
                evidence_ids=ev_ids, n_evidence=len(ev_ids),
                evidence_words=sum(len(e["text"].split()) for e in res["evidence"]),
                llm_calls=res.get("llm_calls"), seconds=res.get("seconds"), gen_seconds=res.get("gen_seconds"),
                reused=bool(res.get("reused")),
                subqs=res.get("subqs"), log=res.get("log"), answer=res["answer"])


async def _claims(res: dict) -> dict:
    """Claim-level groundedness; a whole-answer decline makes no claims, so it is not sent to the judge."""
    from .judge import judge_claims
    if abstained(res["answer"]):
        return dict(claims=[], n_claims=0, claim_support=None, material_errors=0, faithful=True, unsupported=[])
    return await judge_claims(res["answer"], res["evidence"])


async def generation(n: int | None = None, systems=("single_shot", "agent"), seed: int = 3, variant: str | None = None,
                     concurrency: int = 3, judge_faith: bool = True, qids: list[str] | None = None,
                     tag: str | None = None, split: str | None = "test", resume: bool = False,
                     post: bool = True) -> dict:
    """n=None: the whole split. With split="dev" and n=30 this is the 30-question set used on main.

    Single-shot runs first on each question; the agent reuses its answer when its evidence is single-shot's, chunk
    for chunk (docs/05, "Shared-path reuse"). Judges: Harvey rubric verdict + claim-level groundedness. post: once
    every pair is done, discordant rubric verdicts are re-judged (majority of 3) and answers that differ get a
    pairwise judgment; the comparison with confidence intervals is written to compare_<tag>.json."""
    from .judge import judge_correctness
    variant = variant or default_variant()
    items = load_qa(split=split)
    random.Random(seed).shuffle(items)
    items = [i for i in items if i["qid"] in set(qids)] if qids else items[:n] if n else items
    systems = _ordered(systems)
    tag = tag or "_".join(s.replace(":", "-") for s in systems)
    out_path = EVAL_DIR / f"gen_{tag}.jsonl"
    _prepare_run(tag, "generation", variant, systems, items, resume, post)
    done = _load_done(out_path, resume)

    async def work(it, system, prev):
        with tracing.span("eval.item", kind="CHAIN", **{"eval.system": system, "eval.qid": it["qid"],
                                                        "input.value": it["question"]}):
            res = await _run_system(system, it, variant, _reuse(prev) if system.startswith("agent") else None)
            rec = _record(it, system, res)
            if rec["reused"]:
                return _with_reused(rec, prev["single_shot"])
            corr = await judge_correctness(it, res["answer"])
            rec.update(verdict=corr["verdict"], judge_reasoning=corr["reasoning"])
            if judge_faith:
                rec.update(await _claims(res))
            return rec

    todo = sum((it["qid"], s) not in done for it in items for s in systems)
    print(f"{len(items)} questions x {len(systems)} systems: {len(done)} done earlier, {todo} to run", flush=True)
    left = await _run_items(items, systems, work, out_path, done, concurrency)
    rejudge_left = pairwise_left = None
    if left:
        print(f"INCOMPLETE: {left} pairs still missing -- rerun with resume=True", flush=True)
    elif post and {"agent", "single_shot"} <= set(systems):
        by_qid = {it["qid"]: it for it in items}
        rejudge_left = await rejudge(tag, by_qid, "verdict",
                                    lambda it, r: judge_correctness(it, r["answer"]), "pass")
        if not rejudge_left:
            pairwise_left = await pairwise(
                tag, by_qid, lambda it: "Reference answer (written from the matter's grading rubric; it may be "
                                        f"imperfect): {it['expected_answer']}\n")
    complete, post_complete, stage = _completion_state(left, post, systems, rejudge_left, pairwise_left)
    _update_manifest(tag, complete=complete, stage=stage, post_complete=post_complete,
                     answer_pairs_left=left, rejudge_answers_left=rejudge_left,
                     pairwise_questions_left=pairwise_left)
    if not complete and not left:
        print("INCOMPLETE: acceptance comparison requires post-pass judgments; rerun with post=True, resume=True",
              flush=True)
    return _finish(tag, items, systems)


def load_hard(path: Path | None = None) -> list[dict]:
    path = path or EVAL_DIR / "qa_hard.jsonl"
    return [json.loads(l) for l in open(path)] if path.exists() else []


def validate_hard_set(items: list[dict], kinds=HARD_KINDS, target: int = HARD_TARGET_PER_KIND) -> None:
    """Refuse a headline hard-set run until every selected kind is complete and passed the current verifier."""
    current = [i for i in items if i.get("kind") in kinds and i.get("verification_version") == 2]
    counts = {kind: sum(i.get("kind") == kind for i in current) for kind in kinds}
    unverified = [i.get("qid") for i in items if i.get("kind") in kinds and i.get("verification_version") != 2]
    missing = {kind: target - counts[kind] for kind in kinds if counts[kind] < target}
    qids, questions = [i.get("qid") for i in current], [i.get("question") for i in current]
    malformed = []
    per_matter = {}
    for i in current:
        key = (i.get("kind"), i.get("task_id"))
        per_matter[key] = per_matter.get(key, 0) + 1
        n = 2 if i.get("kind") == "bridge" else 3
        verify = i.get("verify") or {}
        if (len(i.get("gold") or []) != n or len(verify.get("alone") or []) != n
                or len(verify.get("without_each") or []) != n or any(verify.get("alone") or [])
                or any(verify.get("without_each") or []) or not verify.get("together_ok")
                or not verify.get("link_hidden") or verify.get("gives_away")
                or not i.get("facts")
                or len(verify.get("facts_supported") or []) != len(i.get("facts") or [])
                or not all(verify.get("facts_supported") or [])):
            malformed.append(i.get("qid"))
    duplicates = len(qids) != len(set(qids)) or len(questions) != len(set(questions))
    over_cap = [f"{kind}/{task}" for (kind, task), count in per_matter.items() if count > 2]
    if missing or malformed or duplicates or over_cap:
        raise RuntimeError("hard set is not ready: "
                           f"validated counts={counts}, missing={missing}, malformed={malformed}, "
                           f"duplicates={duplicates}, over_cap={over_cap}, unverified_ignored={len(unverified)}; "
                           "run `python -m hrag.eval.build_hard --kinds bridge aspect --n 25`")


async def hard(kinds=HARD_KINDS, systems=("single_shot", "agent"), n: int | None = None, variant: str | None = None,
               concurrency: int = 3, tag: str = "hard_v2", resume: bool = True, post: bool = True,
               require_complete: bool = True) -> dict:
    """Strict hard sets (build_hard.py, docs/05): two-step links (bridge) and questions needing one necessary fact from
    each of three documents (aspect). Answers are judged against the selected source passages and expected answer;
    `all_gold_in_evidence` means all selected source chunks were retrieved, not that equivalent evidence cannot exist."""
    from .judge import judge_reference
    variant = variant or default_variant()
    all_items = load_hard()
    if require_complete:
        validate_hard_set(all_items, kinds)
    items = [i for i in all_items if i["kind"] in kinds and i.get("verification_version") == 2]
    items = items[:n] if n else items
    systems = _ordered(systems)
    out_path = EVAL_DIR / f"gen_{tag}.jsonl"
    _prepare_run(tag, "hard", variant, systems, items, resume, post)
    done = _load_done(out_path, resume)

    async def work(it, system, prev):
        res = await _run_system(system, it, variant, _reuse(prev) if system.startswith("agent") else None)
        rec = _record(it, system, res)
        if rec["reused"]:
            return _with_reused(rec, prev["single_shot"])
        if it["kind"] == "absent":
            j = await llm.complete(UNANSWERABLE_JUDGE.format(question=it["question"], answer=res["answer"]),
                                   model=CFG.fast_llm, schema=UNANSWERABLE_SCHEMA, purpose="judge_unanswerable")
            return rec | dict(declined=bool(j["declined"]), decline_reason=j["reason"])
        rec.update(await judge_reference(it, res["answer"], passages(it["gold"])))
        return rec | await _claims(res)

    todo = sum((it["qid"], s) not in done for it in items for s in systems)
    print(f"{len(items)} hard questions x {len(systems)} systems: {len(done)} done earlier, {todo} to run", flush=True)
    left = await _run_items(items, systems, work, out_path, done, concurrency)
    rejudge_left = pairwise_left = None
    if left:
        print(f"INCOMPLETE: {left} pairs still missing -- rerun with resume=True", flush=True)
    elif post and {"agent", "single_shot"} <= set(systems):
        by_qid = {it["qid"]: it for it in items if it["kind"] != "absent"}
        rejudge_left = await rejudge(tag, by_qid, "reference_verdict",
                                    lambda it, r: judge_reference(it, r["answer"], passages(it["gold"])), "correct")
        if not rejudge_left:
            pairwise_left = await pairwise(tag, by_qid, lambda it: f"Reference answer: {it['expected_answer']}\n")
    complete, post_complete, stage = _completion_state(left, post, systems, rejudge_left, pairwise_left)
    _update_manifest(tag, complete=complete, stage=stage, post_complete=post_complete,
                     answer_pairs_left=left, rejudge_answers_left=rejudge_left,
                     pairwise_questions_left=pairwise_left)
    if not complete and not left:
        print("INCOMPLETE: acceptance comparison requires post-pass judgments; rerun with post=True, resume=True",
              flush=True)
    return _finish(tag, items, systems)


# ------------------------------------------------------------------------------------------- post-passes
async def rejudge(tag: str, items: dict[str, dict], field: str, judge, success: str, a: str = "agent",
                  b: str = "single_shot", votes: int = 3) -> int:
    """Where one system's `field` verdict is `success` and the other's is not, judge both answers again until each has
    `votes` verdicts; the majority goes to rejudge_<tag>.jsonl and is merged as <field>_majority (a single judge flip
    can no longer decide a comparison). Reused answers can't disagree. Returns #answers left (rerun to continue after
    a session limit)."""
    from .stats import by_question
    path = EVAL_DIR / f"rejudge_{tag}.jsonl"
    have = {(r["qid"], r["system"], r["field"]) for r in map(json.loads, open(path))} if path.exists() else set()
    pairs = by_question([json.loads(l) for l in open(EVAL_DIR / f"gen_{tag}.jsonl")])
    jobs = []
    for q, d in pairs.items():
        if q not in items or a not in d or b not in d or (d[a].get(field) == success) == (d[b].get(field) == success):
            continue
        for s in (a, b):
            if (q, s, field) in have:
                continue

            async def job(q=q, s=s, r=d[s]):
                extra = [(await judge(items[q], r))[field] for _ in range(votes - 1)]
                vs = [r[field]] + extra
                best = max(vs, key=lambda v: (vs.count(v), v == r[field]))  # ties keep the original verdict
                return dict(qid=q, system=s, field=field, votes=vs, majority=best)
            jobs.append(job)
    print(f"re-judging {len(jobs)} answers of discordant pairs ({field})", flush=True)
    return await _append_all(jobs, path)


async def pairwise(tag: str, items: dict[str, dict], reference, a: str = "agent", b: str = "single_shot") -> int:
    """One pairwise judgment per question where the agent did NOT reuse single-shot's answer (elsewhere the answers are
    identical), shown in a seeded random order; written to pairwise_<tag>.jsonl. Returns #questions left."""
    from .judge import judge_pairwise
    from .stats import by_question
    path = EVAL_DIR / f"pairwise_{tag}.jsonl"
    have = {r["qid"] for r in map(json.loads, open(path))} if path.exists() else set()
    pairs = by_question([json.loads(l) for l in open(EVAL_DIR / f"gen_{tag}.jsonl")])
    jobs = []
    for q, d in pairs.items():
        if q not in items or q in have or a not in d or b not in d or d[a].get("reused") \
                or d[a]["answer"] == d[b]["answer"]:
            continue
        agent_first = random.Random(f"{tag}:{q}").random() < 0.5
        first, second = (d[a], d[b]) if agent_first else (d[b], d[a])

        async def job(q=q, first=first, second=second, agent_first=agent_first):
            first_ev, second_ev = passages(first.get("evidence_ids", [])), passages(second.get("evidence_ids", []))
            j = await judge_pairwise(items[q]["question"], reference(items[q]), first["answer"], second["answer"],
                                     first_ev, second_ev)
            winner = {"1": first["system"], "2": second["system"]}.get(j["winner"], "tie")
            return dict(qid=q, task_id=items[q]["task_id"], winner=winner, agent_first=agent_first,
                        reasoning=j["reasoning"])
        jobs.append(job)
    print(f"pairwise judgments to run: {len(jobs)}", flush=True)
    return await _append_all(jobs, path)


# ------------------------------------------------------------------------------------------- results
def load_records(tag: str) -> tuple[list[dict], list[dict]]:
    """Records of gen_<tag>.jsonl with the re-judged majorities merged in, and the pairwise preferences."""
    path = EVAL_DIR / f"gen_{tag}.jsonl"
    records = [json.loads(l) for l in open(path)] if path.exists() else []
    rj = EVAL_DIR / f"rejudge_{tag}.jsonl"
    if rj.exists():
        maj = {(r["qid"], r["system"], r["field"]): r["majority"] for r in map(json.loads, open(rj))}
        for r in records:
            for f in ("verdict", "reference_verdict"):
                if (r["qid"], r["system"], f) in maj:
                    r[f"{f}_majority"] = maj[(r["qid"], r["system"], f)]
    pw = EVAL_DIR / f"pairwise_{tag}.jsonl"
    prefs = [json.loads(l) for l in open(pw)] if pw.exists() else []
    return records, prefs


def compare_tag(tag: str, a: str = "agent", b: str = "single_shot") -> dict:
    """Paired comparison of a run (stats.compare) on all questions, where the agent took its own path, per question
    kind and per audit stratum of the rubric; written to compare_<tag>.json."""
    from . import stats
    manifest = _manifest_path(tag)
    if not manifest.exists():
        raise RuntimeError(f"run {tag} has no manifest and cannot be verified; rerun it with the current evaluator")
    if not json.loads(manifest.read_text()).get("complete"):
        raise RuntimeError(f"run {tag} is incomplete; resume it before comparing")
    records, prefs = load_records(tag)
    audit = {i["qid"]: i.get("audit") for i in load_qa() + load_hard()}
    subsets = {"all": lambda ra, rb: True, "agent took its own path": lambda ra, rb: not ra.get("reused")}
    for k in sorted({r["kind"] for r in records if "kind" in r}):
        subsets[f"kind: {k}"] = lambda ra, rb, k=k: ra["kind"] == k
    if any(audit.get(r["qid"]) for r in records):
        subsets["rubric not contradicted by the documents (audit)"] = \
            lambda ra, rb: audit.get(ra["qid"]) != "contradicted"
        for v in ("supported", "outside_knowledge", "not_determinable", "contradicted"):
            subsets[f"audit: {v}"] = lambda ra, rb, v=v: audit.get(ra["qid"]) == v
    comp = stats.compare(records, a, b, prefs=prefs, subsets=subsets)
    (EVAL_DIR / f"compare_{tag}.json").write_text(json.dumps(comp, indent=1))
    return comp


def _finish(tag: str, items: list[dict], systems: list[str]) -> dict:
    from . import stats
    wanted = {(it["qid"], s) for it in items for s in systems}
    records = [r for r in load_records(tag)[0] if (r["qid"], r["system"]) in wanted]
    summary = summarize_generation(records)
    (EVAL_DIR / f"summary_gen_{tag}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    manifest = json.loads(_manifest_path(tag).read_text()) if _manifest_path(tag).exists() else {}
    if {"agent", "single_shot"} <= set(systems) and manifest.get("complete"):
        print(stats.format_table(compare_tag(tag)))
    elif {"agent", "single_shot"} <= set(systems):
        print(f"PARTIAL RUN: {tag} is not complete; no acceptance comparison is reported", flush=True)
    return summary


def report(tags: list[str]) -> None:
    """Print the comparison tables of finished runs (recomputed from the files: no LLM calls)."""
    from . import stats
    for tag in tags:
        manifest = _manifest_path(tag)
        if not manifest.exists():
            raise RuntimeError(f"run {tag} has no manifest and cannot be verified; rerun it with the current evaluator")
        if not json.loads(manifest.read_text()).get("complete"):
            raise RuntimeError(f"run {tag} is incomplete; resume it before reporting")
        print(f"\n######## {tag}")
        print(stats.format_table(compare_tag(tag)))


UNANSWERABLE_JUDGE = """The question below was asked against a legal matter's data room that does NOT contain the
answer (no document in it answers the question). Did the answer correctly decline, i.e. state that the documents do
not establish the requested facts, without asserting specific facts, figures or terms as the answer?

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
    citations). The experimental inside-matter `absent` builder is not part of the headline hard run."""
    from ..agent.graph import _matters
    variant = variant or default_variant()
    rng = random.Random(seed)
    items = [i for i in load_qa(split=split) if i["gold"]]
    rng.shuffle(items)
    tasks = sorted(_matters())
    items = [dict(i, task_id=other_matter(rng, i["task_id"], tasks)) for i in items[:n]]
    systems = _ordered(systems)
    out_path = EVAL_DIR / f"unanswerable_{tag or variant}.jsonl"
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    done = _load_done(out_path, resume)

    async def work(it, system, prev):
        res = await _run_system(system, it, variant)
        j = await llm.complete(UNANSWERABLE_JUDGE.format(question=it["question"], answer=res["answer"]),
                               model=CFG.fast_llm, schema=UNANSWERABLE_SCHEMA, purpose="judge_unanswerable")
        return dict(qid=it["qid"], asked_in=it["task_id"], system=system, declined=bool(j["declined"]),
                    reason=j["reason"], llm_calls=res.get("llm_calls"), route=res.get("route"),
                    log=res.get("log"), answer=res["answer"])

    left = await _run_items(items, systems, work, out_path, done, concurrency)
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
    """Per-system means. Keys appear only where they apply (rubric verdicts on the QA splits, reference verdicts and
    decline rates on the hard sets); older records (Haiku faithfulness, no claims) still summarize."""
    out = {}
    for system in sorted({r["system"] for r in records}):
        rs = [r for r in records if r["system"] == system and "error" not in r]
        if not rs:
            continue
        mean = lambda xs: round(float(np.mean(xs)), 3) if len(xs) else None  # noqa: E731
        vals = lambda key: [r[key] for r in rs if r.get(key) is not None]  # noqa: E731
        kinds = sorted({r["kind"] for r in rs})
        s = dict(n=len(rs), errors=sum(1 for r in records if r["system"] == system and "error" in r))
        voted = [r for r in rs if r.get("verdict")]
        verdict = lambda r: (r.get("verdict_majority") or r["verdict"]) == "pass"  # noqa: E731
        s["pass_rate"] = mean([verdict(r) for r in voted])
        for k in kinds:
            s[f"pass_rate_{k}"] = mean([verdict(r) for r in voted if r["kind"] == k])
        refd = [r for r in rs if r.get("reference_verdict")]
        correct = lambda r: (r.get("reference_verdict_majority") or r["reference_verdict"]) == "correct"  # noqa: E731
        s["correct_rate"] = mean([correct(r) for r in refd])
        for k in kinds:
            s[f"correct_rate_{k}"] = mean([correct(r) for r in refd if r["kind"] == k])
        s.update(fact_coverage=mean(vals("fact_coverage")), declined_rate=mean(vals("declined")),
                 faithful_rate=mean([bool(r["faithful"]) for r in rs if r.get("faithful") is not None]),
                 claim_support=mean(vals("claim_support")), material_errors=mean(vals("material_errors")),
                 key_fact_recall=mean(vals("key_fact_recall")),
                 abstain_rate=mean([abstained(r["answer"]) for r in rs]),  # recomputed: older records used a looser rule
                 hedge_rate=mean([hedged(r["answer"]) for r in rs]),
                 evidence_has_gold=mean(vals("evidence_hits_gold")),
                 all_gold_in_evidence=mean(vals("all_gold_in_evidence")),
                 reused_rate=mean([bool(r.get("reused")) for r in rs]) if system.startswith("agent") else None,
                 avg_evidence_words=mean(vals("evidence_words")), avg_llm_calls=mean(vals("llm_calls")),
                 avg_seconds=mean(vals("seconds")),
                 routes={k: sum(r["route"] == k for r in rs) for k in {r["route"] for r in rs}})
        out[system] = {k: v for k, v in s.items() if v is not None}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["retrieval", "calibrate", "generation", "hard", "negatives", "report"])
    ap.add_argument("--n", type=int, default=None, help="questions (default: whole split / hard set; negatives: 40)")
    ap.add_argument("--systems", nargs="+", default=["single_shot", "agent"])
    ap.add_argument("--variant", default=None)
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--split", default=None, help="dev | test (calibrate: dev; generation/negatives: test)")
    ap.add_argument("--tag", default=None, help="output file tag")
    ap.add_argument("--tags", nargs="+", default=None, help="report: tags of finished runs")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run instead of starting over")
    a = ap.parse_args()
    if a.what == "retrieval":
        retrieval(a.variants)
    elif a.what == "calibrate":
        calibrate(a.variant, split=a.split or "dev")
    elif a.what == "negatives":
        asyncio.run(negatives(a.n or 40, a.systems, variant=a.variant, split=a.split or "test", tag=a.tag,
                              resume=a.resume))
    elif a.what == "hard":
        asyncio.run(hard(systems=a.systems, n=a.n, variant=a.variant, tag=a.tag or "hard_v2", resume=a.resume))
    elif a.what == "report":
        report(a.tags or ["test_v3", "hard_v2"])
    else:
        asyncio.run(generation(a.n, a.systems, variant=a.variant, split=a.split or "test", tag=a.tag,
                               resume=a.resume))
    print("llm:", llm.STATS.snapshot())
