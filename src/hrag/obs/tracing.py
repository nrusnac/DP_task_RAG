"""Observability: Arize Phoenix + OpenTelemetry (OpenInference conventions).

- `setup()` launches Phoenix in-process (Colab/Jupyter) or exports to PHOENIX_COLLECTOR_ENDPOINT, registers the
  tracer provider and auto-instruments LangChain/LangGraph.
- `span(name, kind=..., **attrs)` is a context manager for manual spans (RETRIEVER, RERANKER, LLM, CHAIN...).
  It is a no-op if tracing was never set up, so the pipeline runs without Phoenix installed.
"""
from __future__ import annotations

import contextlib
import json
import os

_TRACER = None
SESSION_URL = None


def setup(project: str = "dp-task-rag", launch: bool = True) -> str | None:
    global _TRACER, SESSION_URL
    from phoenix.otel import register

    endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")
    if launch and not endpoint:
        endpoint = "http://127.0.0.1:6006"
        if not _launch_phoenix(endpoint):
            print("[tracing] Phoenix did not start; continuing without tracing")
            return None
        SESSION_URL = endpoint
        try:  # Colab: expose the UI through the kernel proxy
            from google.colab.output import eval_js  # type: ignore
            SESSION_URL = eval_js("google.colab.kernel.proxyPort(6006)")
        except Exception:
            pass
    tp = register(project_name=project, endpoint=endpoint.rstrip("/") + "/v1/traces",
                  protocol="http/protobuf", auto_instrument=True, batch=False)
    _TRACER = tp.get_tracer("hrag")
    return SESSION_URL


def _up(url: str) -> bool:
    import urllib.request
    try:
        urllib.request.urlopen(url, timeout=2)
        return True
    except Exception:
        return False


def _launch_phoenix(url: str, wait_s: int = 90) -> bool:
    """In-process launch_app() first (notebooks); fall back to a `phoenix serve` subprocess."""
    import subprocess
    import sys
    import time
    if _up(url):
        return True
    try:
        import phoenix as px
        px.launch_app()
        return True
    except Exception as e:  # e.g. "server took too long to start" on a busy machine
        print(f"[tracing] launch_app failed ({e!r:.120}); starting `phoenix serve` subprocess")
    subprocess.Popen([sys.executable, "-m", "phoenix.server.main", "serve"], stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if _up(url):
            return True
        time.sleep(1)
    return False


def _clean(v):
    if isinstance(v, (str, bool, int, float)):
        return v
    return json.dumps(v, default=str)[:20000]


@contextlib.contextmanager
def span(name: str, kind: str = "CHAIN", **attrs):
    if _TRACER is None:
        yield None
        return
    with _TRACER.start_as_current_span(name) as sp:
        sp.set_attribute("openinference.span.kind", kind)
        set_attrs(sp, **attrs)
        yield sp


def set_attrs(sp, **attrs):
    if sp is None:
        return
    for k, v in attrs.items():
        if v is not None:
            sp.set_attribute(k, _clean(v))


def set_documents(sp, docs: list[dict], prefix: str = "retrieval.documents"):
    """docs: [{"id":..., "content":..., "score":...}] in OpenInference retriever format."""
    if sp is None:
        return
    for i, d in enumerate(docs):
        sp.set_attribute(f"{prefix}.{i}.document.id", str(d.get("id")))
        sp.set_attribute(f"{prefix}.{i}.document.content", str(d.get("content", ""))[:3000])
        if d.get("score") is not None:
            sp.set_attribute(f"{prefix}.{i}.document.score", float(d["score"]))
