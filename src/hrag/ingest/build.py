"""A1-A3: acquire the Harvey LAB subset, parse every document, chunk it. Writes artifacts/{tasks.jsonl,chunks.parquet}.

    python -m hrag.ingest.build            # sparse-checkout + parse + chunk
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
import time
from pathlib import Path

import pandas as pd

from ..config import CFG
from .chunking import chunk_document
from .parsers import parse


def acquire(cfg=CFG) -> Path:
    repo = cfg.raw_dir / "harvey-labs"
    if not repo.exists():
        cfg.raw_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse", cfg.harvey_repo,
                        str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "sparse-checkout", "set", f"tasks/{cfg.practice_area}"], check=True)
    return cfg.area_dir


def load_tasks(area_dir: Path) -> list[dict]:
    tasks = []
    for tj in sorted(area_dir.glob("*/task.json")):
        d = json.loads(tj.read_text())
        tasks.append(dict(task_id=tj.parent.name, title=d.get("title"), work_type=d.get("work_type"),
                          tags=d.get("tags", []), instructions=d.get("instructions", ""),
                          deliverables=d.get("deliverables", {}), criteria=d.get("criteria", []),
                          n_docs=len(list((tj.parent / "documents").glob("*")))))
    return tasks


_BANNER = re.compile(r"confidential|privileged|restricted|business secrets|work product|draft|attorney|"
                     r"\bllp\b|\bsuite\b|avenue|street|^from:|^to:|this document", re.I)


def doc_title(path: Path, detected: str | None) -> str:
    """Filenames in this corpus are descriptive ('market-analysis-report.docx'); detected titles are often
    confidentiality banners or letterheads, so use the filename and append the detected title only if clean."""
    base = path.stem.replace("-", " ").replace("_", " ").strip().title()
    det = (detected or "").strip()
    if det and len(det) <= 90 and not _BANNER.search(det) and det.lower() != base.lower():
        return f"{base} — {det}"
    return base


def build(cfg=CFG) -> dict:
    t0 = time.perf_counter()
    area = acquire(cfg)
    tasks = [t for t in load_tasks(area) if t["n_docs"] > 0]
    chunks, failures = [], []
    for t in tasks:
        for path in sorted((area / t["task_id"] / "documents").glob("*")):
            try:
                title, elements = parse(path)
                chunks += chunk_document(t["task_id"], path.name, path.suffix.lstrip(".").lower(),
                                         doc_title(path, title), elements)
            except Exception as e:  # noqa: BLE001  -- one broken file must not kill the index
                failures.append((t["task_id"], path.name, repr(e)[:200]))
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.artifacts_dir / "tasks.jsonl", "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    df = pd.DataFrame([c.to_dict() for c in chunks])
    df.to_parquet(cfg.artifacts_dir / "chunks.parquet", index=False)
    stats = dict(n_tasks=len(tasks), n_docs=int(df.groupby(["task_id", "doc_id"]).ngroups), n_chunks=len(df),
                 tokens_total=int(df.n_tokens.sum()), tokens_median=int(df.n_tokens.median()),
                 tokens_p95=int(df.n_tokens.quantile(0.95)), tokens_max=int(df.n_tokens.max()),
                 by_kind=dict(collections.Counter(df.kind)), by_doc_type=dict(collections.Counter(df.doc_type)),
                 tracked_change_chunks=int(df.text.str.contains(r"\[INSERTED:|\[DELETED:").sum()),
                 failures=failures, seconds=round(time.perf_counter() - t0, 1))
    (cfg.artifacts_dir / "ingest_stats.json").write_text(json.dumps(stats, indent=1))
    return stats


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(build(), indent=1))
