"""Paired statistics for comparing two systems on the same questions (docs/05-evaluation.md, "Statistics").

Both systems answer the same questions, so each question gives a paired difference; questions come in clusters (about
8 per matter, written from one matter's rubric), so the confidence interval comes from a bootstrap that resamples whole
matters. Binary metrics and pairwise preferences use paired randomization that flips whole matter clusters.

    python -m hrag.eval.stats artifacts/eval/gen_test_v2.jsonl      # offline: no LLM calls
"""
from __future__ import annotations

import json
import itertools
import random
import sys
from collections import defaultdict
from math import comb
from typing import Callable


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts (b: only the first system succeeds, c: only the
    second). Also the two-sided sign test for wins vs. losses."""
    n = b + c
    if not n:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


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
    """Two-sided paired randomization test that flips all question differences in a matter together. This preserves
    within-matter dependence; question-level McNemar/sign tests do not."""
    sums = [sum(diffs[k]) for k in sorted(diffs)]
    observed = abs(sum(sums))
    if not observed:
        return 1.0
    if len(sums) <= 16:
        totals = [abs(sum(s * sign for s, sign in zip(sums, signs)))
                  for signs in itertools.product((-1, 1), repeat=len(sums))]
        return sum(t >= observed - 1e-12 for t in totals) / len(totals)
    rng = random.Random(seed)
    extreme = sum(abs(sum(s * rng.choice((-1, 1)) for s in sums)) >= observed - 1e-12
                  for _ in range(n_perm))
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
        only_a = sum(1 for _, x, y in rows if x and not y)
        only_b = sum(1 for _, x, y in rows if y and not x)
        out.update(only_a=only_a, only_b=only_b, p=round(cluster_randomization_p(diffs), 3))
    return out


def _verdict(r: dict) -> bool | None:
    v = r.get("verdict_majority") or r.get("verdict")
    return None if v is None else v == "pass"


# (name, metric, binary). A metric returns None where it does not apply (e.g. no key facts), and that question is
# then left out of that metric's comparison only.
METRICS: list[tuple[str, Callable[[dict], float | bool | None], bool]] = [
    ("pass (rubric)", _verdict, True),
    ("correct (reference)", lambda r: None if not r.get("reference_verdict")
     else (r.get("reference_verdict_majority") or r["reference_verdict"]) == "correct", True),
    ("fact coverage (reference)", lambda r: r.get("fact_coverage"), False),
    ("declined (unanswerable)", lambda r: r.get("declined"), True),
    ("faithful (no material error)", lambda r: r.get("faithful"), True),
    ("claim support", lambda r: r.get("claim_support"), False),
    ("material errors / answer", lambda r: r.get("material_errors"), False),
    ("key-fact recall", lambda r: r.get("key_fact_recall"), False),
    ("all gold chunks in evidence", lambda r: r.get("all_gold_in_evidence"), True),
    ("any gold chunk in evidence", lambda r: r.get("evidence_hits_gold"), True),
    ("LLM calls", lambda r: r.get("llm_calls"), False),
    ("seconds", lambda r: r.get("seconds"), False),
]


def pairwise_summary(prefs: list[dict], a: str = "agent", b: str = "single_shot") -> dict:
    """Win/tie/loss of `a` against `b`, with a matter-cluster randomization test and a position-bias check."""
    if not prefs:
        return {}
    w = sum(p["winner"] == a for p in prefs)
    l_ = sum(p["winner"] == b for p in prefs)
    decided = [p for p in prefs if p["winner"] in (a, b)]
    first_won = sum((p["winner"] == a) == bool(p["agent_first"]) for p in decided)
    diffs: dict[str, list[float]] = defaultdict(list)
    for p in prefs:
        diffs[p.get("task_id") or p["qid"]].append(float(p["winner"] == a) - float(p["winner"] == b))
    return dict(n=len(prefs), n_clusters=len(diffs), wins=w, ties=len(prefs) - w - l_, losses=l_,
                p=round(cluster_randomization_p(diffs), 3),
                first_shown_won=round(first_won / len(decided), 3) if decided else None)


def compare(records: list[dict], a: str = "agent", b: str = "single_shot", prefs: list[dict] | None = None,
            subsets: dict[str, Callable[[dict, dict], bool]] | None = None, n_boot: int = 4000) -> dict:
    """{subset: {metric: paired stats}} (+ "pairwise" per subset). Default subsets: all questions, and the questions
    where the agent's answer was NOT reused from single-shot (its own path; elsewhere the difference is 0 by
    construction)."""
    pairs = by_question(records)
    subsets = subsets or {"all": lambda ra, rb: True, "agent took its own path": lambda ra, rb: not ra.get("reused")}
    out = {}
    for name, keep in subsets.items():
        sub = {q: d for q, d in pairs.items() if a in d and b in d and keep(d[a], d[b])}
        if not sub:
            continue
        res = {m: s for m, f, binary in METRICS if (s := paired(sub, f, a, b, binary, n_boot))}
        if prefs:
            res["pairwise preference"] = pairwise_summary([p for p in prefs if p["qid"] in sub], a, b)
        out[name] = dict(n_questions=len(sub), metrics=res)
    return out


def format_table(comp: dict) -> str:
    """Plain-text table of compare() output: metric | single-shot | agent | diff [95% CI] | p."""
    lines = []
    for name, block in comp.items():
        lines.append(f"\n== {name} ({block['n_questions']} questions)")
        lines.append(f"{'metric':34} {'n':>4} {'single-shot':>11} {'agent':>8}  {'diff [95% CI]':24} {'p':>6}")
        for m, s in block["metrics"].items():
            if m == "pairwise preference":
                if s:
                    lines.append(f"{'pairwise: agent wins/ties/losses':34} {s['n']:>4} {'':>11} {'':>8}  "
                                 f"{s['wins']}/{s['ties']}/{s['losses']:<18} {s['p']:>6}")
                continue
            ci = f"{s['diff']:+.3f} [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]"
            p = f"{s['p']:.3f}" if "p" in s else ""
            lines.append(f"{m:34} {s['n']:>4} {s['b']:>11.3f} {s['a']:>8.3f}  {ci:24} {p:>6}")
    return "\n".join(lines)


if __name__ == "__main__":
    recs = [json.loads(line) for line in open(sys.argv[1])]
    print(format_table(compare(recs)))
