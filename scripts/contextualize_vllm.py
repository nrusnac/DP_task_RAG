"""Standalone vLLM job for per-document Contextual Retrieval. Runs in its own venv (vLLM pins its own torch).

    /content/vllm_env/bin/python scripts/contextualize_vllm.py data/index/ctx_input.json artifacts/contexts.jsonl
    ... --model Qwen/Qwen3-4B-Instruct-2507        # override the model written by `contextualize prepare`
    ... --limit 2                                  # smoke test: first 2 documents only

Resumable: chunk ids already present in the output file are skipped, and results are appended batch by batch,
so the job can be watched (`wc -l artifacts/contexts.jsonl`) and restarted after a Colab disconnect. Each row records
the model that wrote it, so contexts from two models are never mixed silently.

Qwen3.5 notes: it thinks by default (disabled through the chat template), it is a vision-language model (no image
inputs are allowed, so vLLM reserves no memory for the vision encoder), and a T4 has no bf16 (fp16 is used).
"""
import argparse
import json
import re
import sys
import time

BATCH = 24  # prompts per vLLM call; the KV cache (~5 GB on a T4) holds ~15 long documents at once anyway


def load_llm(model, max_model_len, enforce_eager, max_num_seqs):
    from vllm import LLM
    kw = dict(model=model, dtype="half", max_model_len=max_model_len, gpu_memory_utilization=0.85,
              enforce_eager=enforce_eager, max_num_seqs=max_num_seqs)
    try:
        return LLM(**kw, limit_mm_per_prompt={"image": 0, "video": 0})
    except (ValueError, TypeError) as e:  # text-only models reject multimodal limits
        if "multimodal" not in str(e).lower() and "limit_mm" not in str(e).lower():
            raise
        return LLM(**kw)


def main(inp, out, model=None, limit=None, max_model_len=16384, enforce_eager=False, max_num_seqs=BATCH):
    from vllm import SamplingParams

    cfg = json.load(open(inp))
    model = model or cfg["model"]
    done = set()
    try:
        with open(out) as f:
            rows = [json.loads(line) for line in f]
        other = {r.get("model") for r in rows} - {model, None}
        if other:
            sys.exit(f"{out} already holds contexts from {other}; write {model} to a different file")
        done = {r["chunk_id"] for r in rows}
    except FileNotFoundError:
        pass
    jobs = [j for j in cfg["jobs"] if not all(c in done for c in j["chunk_ids"])][:limit]
    print(f"model={model} | {len(jobs)} jobs to run ({len(done)} chunks already done)", flush=True)
    if not jobs:
        return
    llm = load_llm(model, max_model_len, enforce_eager, max_num_seqs)
    tok = llm.get_tokenizer()
    t0, n_done, n_parsed = time.perf_counter(), 0, 0
    for b in range(0, len(jobs), BATCH):
        batch = jobs[b:b + BATCH]
        msgs, params = [], []
        for j in batch:
            n_prompt = len(tok(j["prompt"])["input_ids"]) + 64
            max_new = min(90 * len(j["chunk_ids"]) + 50, max_model_len - n_prompt)
            msgs.append([{"role": "user", "content": j["prompt"]}])
            params.append(SamplingParams(temperature=0, max_tokens=max(64, max_new)))
        outs = llm.chat(msgs, params, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
        with open(out, "a") as f:
            for j, o in zip(batch, outs):
                got = {int(m.group(1)): m.group(2).strip()
                       for m in re.finditer(r"^\s*\[c(\d+)\]\s*(.+)$", o.outputs[0].text, re.M)}
                for i, cid in enumerate(j["chunk_ids"], 1):
                    f.write(json.dumps({"chunk_id": cid, "context": got.get(i), "model": model}) + "\n")
                    n_parsed += i in got
                n_done += len(j["chunk_ids"])
        el = time.perf_counter() - t0
        print(f"[{b + len(batch)}/{len(jobs)} jobs] chunks={n_done} coverage={n_parsed / max(n_done, 1):.3f} "
              f"elapsed={el / 60:.1f} min  s/chunk={el / max(n_done, 1):.2f}", flush=True)
    if limit:  # smoke test: show what the model wrote
        print("sample:", outs[0].outputs[0].text[:600], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--enforce-eager", action="store_true", help="skip CUDA graphs (less GPU memory, slower)")
    ap.add_argument("--max-num-seqs", type=int, default=BATCH)
    a = ap.parse_args()
    main(a.inp, a.out, a.model, a.limit, a.max_model_len, a.enforce_eager, a.max_num_seqs)
