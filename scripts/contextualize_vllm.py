"""Standalone vLLM job for per-document Contextual Retrieval. Runs in its own venv (vLLM pins its own torch).

    /content/vllm_env/bin/python scripts/contextualize_vllm.py data/index/ctx_input.json artifacts/contexts.jsonl

Resumable: chunk ids already present in the output file are skipped, and results are appended batch by batch,
so the job can be watched (`wc -l artifacts/contexts.jsonl`) and restarted after a Colab disconnect.
"""
import json
import re
import sys
import time

BATCH = 24  # prompts per vLLM call; the KV cache (~5 GB on a T4) holds ~15 long documents at once anyway


def main(inp, out, max_model_len=16384):
    from vllm import LLM, SamplingParams

    cfg = json.load(open(inp))
    done = set()
    try:
        with open(out) as f:
            done = {json.loads(line)["chunk_id"] for line in f}
    except FileNotFoundError:
        pass
    jobs = [j for j in cfg["jobs"] if not all(c in done for c in j["chunk_ids"])]
    print(f"{len(jobs)} jobs to run ({len(done)} chunks already done)", flush=True)
    if not jobs:
        return
    llm = LLM(model=cfg["model"], dtype="half", max_model_len=max_model_len, gpu_memory_utilization=0.85)
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
        outs = llm.chat(msgs, params, use_tqdm=False)
        with open(out, "a") as f:
            for j, o in zip(batch, outs):
                got = {int(m.group(1)): m.group(2).strip()
                       for m in re.finditer(r"^\s*\[c(\d+)\]\s*(.+)$", o.outputs[0].text, re.M)}
                for i, cid in enumerate(j["chunk_ids"], 1):
                    f.write(json.dumps({"chunk_id": cid, "context": got.get(i)}) + "\n")
                    n_parsed += i in got
                n_done += len(j["chunk_ids"])
        el = time.perf_counter() - t0
        print(f"[{b + len(batch)}/{len(jobs)} jobs] chunks={n_done} coverage={n_parsed / max(n_done, 1):.3f} "
              f"elapsed={el / 60:.1f} min  s/chunk={el / max(n_done, 1):.2f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
