"""A4: Contextual Retrieval, per document (see docs/03-rag-techniques.md §2). Model: config.context_model.

Per-chunk prompting (Anthropic's original recipe) measured 2.62 s/chunk ≈ 2.9 h on a T4, because the whole document
is re-sent for every chunk. Instead, each prompt carries the document ONCE with inline [[c1]]..[[cN]] markers and
asks for one context line per chunk (measured 0.98 s/chunk, 99.6% coverage).

    python -m hrag.ingest.contextualize prepare     # main env  -> data/index/ctx_input.json
    <vllm venv>/bin/python scripts/contextualize_vllm.py data/index/ctx_input.json artifacts/contexts.jsonl
"""
from __future__ import annotations

import json
import re
import sys

import pandas as pd

from ..config import CFG

WINDOW_TOKENS = 10_000  # documents longer than this are processed in consecutive windows

INSTRUCTION = (
    "The document above has been split into consecutive chunks; each chunk starts with a marker like [[c1]], [[c2]], ...\n"
    "For EACH chunk, write one short, succinct context (1-2 sentences, at most 50 words) that situates the chunk "
    "within the overall document, to improve search retrieval of the chunk: name the parties or matter and what "
    "that part of the document is about.\n"
    "Answer with exactly one line per chunk, in order, formatted as:\n[c1] <context>\n[c2] <context>\n...\n"
    "Output nothing else.")


def prepare(chunks: pd.DataFrame, out_path) -> dict:
    jobs = []
    for (task_id, doc_id), g in chunks.sort_values("order").groupby(["task_id", "doc_id"], sort=False):
        rows = g.to_dict("records")
        windows, cur, cur_t = [], [], 0
        for r in rows:
            if cur and cur_t + r["n_tokens"] > WINDOW_TOKENS:
                windows.append(cur); cur, cur_t = [], 0
            cur.append(r); cur_t += r["n_tokens"]
        if cur:
            windows.append(cur)
        intro = rows[0]["text"][:2500]
        for wi, win in enumerate(windows):
            body = "\n".join(f"[[c{i}]] {r['header']}\n{r['text']}" for i, r in enumerate(win, 1))
            prefix = "" if wi == 0 else f"(Document start, for orientation)\n{intro}\n...\n(Window {wi + 1} of {len(windows)})\n"
            jobs.append(dict(job_id=f"{task_id}::{doc_id}::w{wi}", chunk_ids=[r["chunk_id"] for r in win],
                             prompt=f"<document title=\"{rows[0]['doc_title']}\">\n{prefix}{body}\n</document>\n\n{INSTRUCTION}"))
    with open(out_path, "w") as f:
        json.dump(dict(model=CFG.context_model, jobs=jobs), f)
    return dict(n_jobs=len(jobs), n_chunks=int(len(chunks)), multi_window_docs=sum(j["job_id"].endswith("w1") for j in jobs))


_CLEAN = [(re.compile(r"^\s*(context\s*:\s*)", re.I), ""), (re.compile(r"^[<\"']+|[>\"']+$"), "")]


def clean_context(text: str | None) -> str:
    if not text:
        return ""
    t = text.strip()
    for rx, rep in _CLEAN:
        t = rx.sub(rep, t).strip()
    return t


def context_models(path) -> dict[str, int]:
    """Which model wrote the contexts (rows from before the script recorded it count as "unrecorded")."""
    out: dict[str, int] = {}
    try:
        with open(path) as f:
            for line in f:
                m = json.loads(line).get("model") or "unrecorded"
                out[m] = out.get(m, 0) + 1
    except FileNotFoundError:
        pass
    return out


def load_contexts(path) -> dict[str, str]:
    out = {}
    try:
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                if d.get("context"):
                    out[d["chunk_id"]] = clean_context(d["context"])
    except FileNotFoundError:
        pass
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "prepare":
        CFG.index_dir.mkdir(parents=True, exist_ok=True)
        df = pd.read_parquet(CFG.artifacts_dir / "chunks.parquet")
        print(prepare(df, CFG.index_dir / "ctx_input.json"))
    else:
        print(__doc__)
