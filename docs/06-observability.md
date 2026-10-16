# 06 · Observability: Arize Phoenix + OpenTelemetry

**Why Phoenix:**
- open source
- runs **inside the Colab notebook** (`px.launch_app()`), with no account or keys
- speaks OpenTelemetry with the OpenInference conventions, so retriever and LLM spans render as such (documents with scores, prompts with token counts)

**Setup** (`src/hrag/obs/tracing.py`):
```python
from hrag.obs import tracing
url = tracing.setup(project="dp-task-rag")   # launches Phoenix, registers OTLP/HTTP exporter, auto-instruments LangChain/LangGraph
```
If `setup()` is never called, every `tracing.span(...)` is a no-op, so the pipeline runs without Phoenix.

## What is traced
| Span | Kind | How | Attributes |
|---|---|---|---|
| `agent.answer` | AGENT | manual | question, task_id, policy, variant, route, llm_calls, answer |
| LangGraph nodes (`route`, `plan`, `retrieve`, `grade`, `refine`, `rewrite`, `sufficiency`, `generate`) | CHAIN | **auto** (`openinference-instrumentation-langchain`) | node input/output state |
| `retrieve.hybrid` | RETRIEVER | manual | query, mode, scope, variant, retrieved documents with scores, latency |
| `rerank` | RERANKER | manual | reranked documents with cross-encoder scores |
| `crag.refine` | RERANKER | manual | strips in vs. kept, words in vs. kept |
| `llm.<purpose>` (router, crag_grade, crag_rewrite, plan, sufficiency, generate, judge…) | LLM | **manual**: the Agent SDK runs Claude in a CLI subprocess that no instrumentor sees | model, backend, prompt, output, token counts, errors |
| `eval.item` | CHAIN | manual | system, qid, so a failing eval item opens as its trace |

## How we use it
- **Debugging a failed eval item:** open its `eval.item` trace and check, in order:
  - wrong route?
  - gold chunk not in the `retrieve.hybrid` documents? (a retrieval miss)
  - CRAG marked the sub-question incorrect and abstained?
  - evidence present but the generator ignored it?
- **Cost and latency:** `llm.*` spans carry token counts and durations. `hrag.llm.STATS` aggregates calls, tokens and seconds by purpose (printed at the end of the notebook).
- **Decision audit:** route, CRAG verdict and best rerank score, rewrites, and sufficiency iterations are also returned in `result["log"]`, so the notebook demo shows the agent's reasoning without opening Phoenix.

The prep smoke test validated this setup on Colab: a 7-span trace across the LangGraph nodes plus manual retriever and LLM spans.
