"""Paired statistics for comparing two systems on the same questions (docs/05-evaluation.md, "How sure are we?").

Both systems answer the same questions, so each question gives a paired difference. Questions come in clusters (up to
8 per matter, written from one matter's rubric), so they are not independent: the confidence interval comes from a
bootstrap that resamples whole matters, and binary metrics get a paired randomization test that flips the signs of
whole matters together. No LLM calls.

    python -m hrag.eval.stats artifacts/eval/gen_naive_single_shot_agent.jsonl                  # agent vs. single-shot
    python -m hrag.eval.stats artifacts/eval/gen_naive_single_shot_agent.jsonl --a single_shot --b naive
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import defaultdict
from typing import Callable


def cluster_bootstrap(diffs: dict[str, list[float]], n_boot: int = 4000, seed: int = 0,
                      alpha: float = 0.05) -> tuple[float, float]:
    """(lo, hi) percentile CI of the mean per-question difference, resampling clusters (matters) with replacement."""
    keys = sorted(diffs)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        xs = [x for k in (rng.choice(keys) for _ in keys) for x in diffs[k]]
        means.append(sum(xs) / len(xs))
    means.sort()
    return means[int(n_boot * alpha / 2)], means[int(n_boot * (1 - alpha / 2)) - 1]


def cluster_randomization_p(diffs: dict[str, list[float]], n_perm: int = 20000, seed: int = 0) -> float:
    """Two-sided paired randomization test that flips all question differences in a matter together (exact when there
    are at most 16 matters). Unlike a question-level McNemar test, it keeps the within-matter dependence."""
    sums = [sum(diffs[k]) for k in sorted(diffs)]
    observed = abs(sum(sums))
    if not observed:
        return 1.0
    if len(sums) <= 16:
        totals = [abs(sum(s * sign for s, sign in zip(sums, signs)))
                  for signs in itertools.product((-1, 1), repeat=len(sums))]
        return sum(t >= observed - 1e-12 for t in totals) / len(totals)
    rng = random.Random(seed)
    extreme = sum(abs(sum(s * rng.choice((-1, 1)) for s in sums)) >= observed - 1e-12 for _ in range(n_perm))
    return (extreme + 1) / (n_perm + 1)


def by_question(records: list[dict]) -> dict[str, dict[str, dict]]:
    """qid -> system -> record (records with an error are left out)."""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in records:
        if "error" not in r:
            out[r["qid"]][r["system"]] = r
    return out


def _cluster(qid: str, rec: dict) -> str:
    return rec.get("task_id") or qid.split("::")[0]


def paired(pairs: dict[str, dict[str, dict]], metric: Callable[[dict], float | bool | None], a: str, b: str,
           binary: bool = False, n_boot: int = 4000) -> dict:
    """metric(a) - metric(b) over the questions both systems answered and the metric is defined for."""
    rows = []
    for q, d in pairs.items():
        if a in d and b in d:
            x, y = metric(d[a]), metric(d[b])
            if x is not None and y is not None:
                rows.append((_cluster(q, d[a]), float(x), float(y)))
    if not rows:
        return {}
    diffs: dict[str, list[float]] = defaultdict(list)
    for k, x, y in rows:
        diffs[k].append(x - y)
    lo, hi = cluster_bootstrap(diffs, n_boot)
    n = len(rows)
    out = dict(n=n, n_clusters=len(diffs), a=round(sum(x for _, x, _ in rows) / n, 3),
               b=round(sum(y for _, _, y in rows) / n, 3),
               diff=round(sum(x - y for _, x, y in rows) / n, 3), ci95=[round(lo, 3), round(hi, 3)])
    if binary:
        out.update(only_a=sum(1 for _, x, y in rows if x and not y), only_b=sum(1 for _, x, y in rows if y and not x),
                   p=round(cluster_randomization_p(diffs), 3))
    return out


# (name, metric, binary). A metric returns None where it does not apply (e.g. no gold label), and that question is
# then left out of that metric's comparison only.
METRICS: list[tuple[str, Callable[[dict], float | bool | None], bool]] = [
    ("pass (Harvey rubric judge)", lambda r: None if r.get("verdict") is None else r["verdict"] == "pass", True),
    ("faithful (Haiku judge)", lambda r: r.get("faithful"), True),
    ("evidence has gold chunk", lambda r: r.get("evidence_hits_gold"), True),
    ("evidence words", lambda r: r.get("evidence_words"), False),
    ("LLM calls", lambda r: r.get("llm_calls"), False),
    ("seconds", lambda r: r.get("seconds"), False),
]


def compare(records: list[dict], a: str = "agent", b: str = "single_shot", n_boot: int = 4000) -> dict:
    """{"n_questions", "n_matters", "metrics": {metric: paired stats}} for system a minus system b."""
    pairs = {q: d for q, d in by_question(records).items() if a in d and b in d}
    metrics = {m: s for m, f, binary in METRICS if (s := paired(pairs, f, a, b, binary, n_boot))}
    return dict(a=a, b=b, n_questions=len(pairs), n_matters=len({_cluster(q, d[a]) for q, d in pairs.items()}),
                metrics=metrics)


def format_table(comp: dict) -> str:
    """Plain-text table of compare() output: metric | b | a | diff [95% CI] | p."""
    a, b = comp["a"], comp["b"]
    lines = [f"{a} vs. {b}: {comp['n_questions']} paired questions from {comp['n_matters']} matters",
             f"{'metric':28} {'n':>4} {b:>12} {a:>12}  {'diff [95% CI]':28} {'p':>6}"]
    for m, s in comp["metrics"].items():
        ci = f"{s['diff']:+.3f} [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]"
        p = f"{s['p']:.3f}" if "p" in s else ""
        lines.append(f"{m:28} {s['n']:>4} {s['b']:>12.3f} {s['a']:>12.3f}  {ci:28} {p:>6}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="a gen_*.jsonl file written by run_eval generation")
    ap.add_argument("--a", default="agent")
    ap.add_argument("--b", default="single_shot")
    args = ap.parse_args()
    recs = [json.loads(line) for line in open(args.path)]
    print(format_table(compare(recs, args.a, args.b)))
