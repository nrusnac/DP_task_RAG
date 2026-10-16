"""One interface for every Claude call: `await complete(prompt, ...)`.

Backends
- "sdk" (primary): claude_agent_sdk, which drives the bundled Claude Code CLI. Auth is the subscription OAuth token
  (CLAUDE_CODE_OAUTH_TOKEN, from Colab Secrets) or, locally, an existing `claude` login. Tools are disabled and
  filesystem settings are isolated, so it behaves as a plain LLM call.
- "api" (fallback): Anthropic Messages API with ANTHROPIC_API_KEY, for reviewers without a subscription.

Credentials are never printed or logged. ANTHROPIC_API_KEY is blanked in the SDK subprocess on purpose: if present
it takes precedence over OAuth and would silently bill the API instead of the subscription.
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import os
import time
from dataclasses import dataclass, field

from .config import CFG
from .obs import tracing

API_MODELS = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5"}


def load_colab_secrets(names=("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY")) -> dict[str, bool]:
    """Copy Colab Secrets into this process's env (never echoed). Returns which ones were found."""
    found = {}
    try:
        from google.colab import userdata  # type: ignore
    except ImportError:
        return {n: bool(os.environ.get(n)) for n in names}
    for n in names:
        try:
            val = userdata.get(n)
        except Exception:
            val = None
        if val:
            os.environ[n] = val
        found[n] = bool(val)
    return found


def backend() -> str:
    choice = os.environ.get("HRAG_LLM_BACKEND", "auto")
    if choice != "auto":
        return choice
    if os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        return "api"
    return "sdk"


@dataclass
class LLMStats:
    calls: int = 0
    errors: int = 0
    seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    by_purpose: dict = field(default_factory=dict)

    def add(self, purpose: str, secs: float, usage: dict, ok: bool):
        self.calls += 1
        self.errors += 0 if ok else 1
        self.seconds += secs
        self.input_tokens += int(usage.get("input_tokens", 0)) + int(usage.get("cache_read_input_tokens", 0)) \
            + int(usage.get("cache_creation_input_tokens", 0))
        self.output_tokens += int(usage.get("output_tokens", 0))
        self.by_purpose[purpose] = self.by_purpose.get(purpose, 0) + 1

    def snapshot(self) -> dict:
        return dict(calls=self.calls, errors=self.errors, seconds=round(self.seconds, 1),
                    input_tokens=self.input_tokens, output_tokens=self.output_tokens, by_purpose=dict(self.by_purpose))


STATS = LLMStats()
# Per-run call counter. Global STATS deltas are wrong under concurrency (other questions' calls leak in), so each
# answer() / baseline run sets its own counter; child asyncio tasks inherit it by reference.
_COUNTER: contextvars.ContextVar[dict | None] = contextvars.ContextVar("hrag_llm_counter", default=None)


def start_counter() -> dict:
    c = {"calls": 0}
    _COUNTER.set(c)
    return c
_SEM: asyncio.Semaphore | None = None


def _sem() -> asyncio.Semaphore:
    global _SEM
    if _SEM is None:
        _SEM = asyncio.Semaphore(CFG.llm_concurrency)
    return _SEM


class LLMError(RuntimeError):
    pass


async def _sdk_call(prompt, system, model, schema, thinking):
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

    env = {"ANTHROPIC_API_KEY": ""}
    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        env["CLAUDE_CODE_OAUTH_TOKEN"] = os.environ["CLAUDE_CODE_OAUTH_TOKEN"]
    opts = dict(model=model, tools=[], setting_sources=[], system_prompt=system, env=env,
                max_turns=2 if schema else 1)  # structured output costs one extra internal turn (measured)
    if not thinking:
        opts["thinking"] = {"type": "disabled"}
    if schema:
        opts["output_format"] = {"type": "json_schema", "schema": schema}
    res = None
    async for m in query(prompt=prompt, options=ClaudeAgentOptions(**opts)):
        if isinstance(m, ResultMessage):
            res = m
    if res is None or res.is_error:
        raise LLMError(f"SDK call failed: {getattr(res, 'subtype', 'no result')} {getattr(res, 'errors', '')}")
    out = res.structured_output if schema else res.result
    if schema and out is None:  # tolerate a model that answered with JSON text instead
        out = _parse_json(res.result or "")
    return out, {k: v for k, v in (res.usage or {}).items() if isinstance(v, int)}


async def _api_call(prompt, system, model, schema, thinking, max_tokens):
    import anthropic

    client = anthropic.AsyncAnthropic()
    kw = dict(model=API_MODELS.get(model, model), max_tokens=max_tokens, system=system,
              messages=[{"role": "user", "content": prompt}])
    if schema:  # forced tool call = reliable structured output
        kw["tools"] = [{"name": "emit", "description": "Return the answer.", "input_schema": schema}]
        kw["tool_choice"] = {"type": "tool", "name": "emit"}
    r = await client.messages.create(**kw)
    usage = {"input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens}
    if schema:
        block = next(b for b in r.content if b.type == "tool_use")
        return block.input, usage
    return "".join(b.text for b in r.content if b.type == "text"), usage


def _parse_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[1] if "\n" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start == -1:
        raise LLMError("no JSON object in model output")
    return json.loads(text[start:end + 1])


async def complete(prompt: str, *, system: str = "You are a precise legal research assistant.",
                   model: str | None = None, schema: dict | None = None, thinking: bool = False,
                   purpose: str = "llm", max_tokens: int = 4096, retries: int = 2):
    """Return text (or a dict if `schema` is given). Traced as an OpenInference LLM span."""
    model = model or CFG.fast_llm
    last_err = None
    for attempt in range(retries + 1):
        async with _sem():
            t0 = time.perf_counter()
            ctr = _COUNTER.get()
            if ctr is not None:
                ctr["calls"] += 1
            with tracing.span(f"llm.{purpose}", kind="LLM", **{
                    "llm.model_name": model, "input.value": prompt[:20000], "llm.backend": backend()}) as sp:
                try:
                    if backend() == "api":
                        out, usage = await _api_call(prompt, system, model, schema, thinking, max_tokens)
                    else:
                        out, usage = await _sdk_call(prompt, system, model, schema, thinking)
                    secs = time.perf_counter() - t0
                    STATS.add(purpose, secs, usage, True)
                    tracing.set_attrs(sp, **{"output.value": json.dumps(out) if schema else str(out)[:20000],
                                             "llm.token_count.prompt": usage.get("input_tokens", 0),
                                             "llm.token_count.completion": usage.get("output_tokens", 0)})
                    return out
                except Exception as e:  # noqa: BLE001
                    STATS.add(purpose, time.perf_counter() - t0, {}, False)
                    tracing.set_attrs(sp, **{"error": repr(e)[:500]})
                    last_err = e
        await asyncio.sleep(1.5 * (attempt + 1))
    raise LLMError(f"{purpose}: failed after {retries + 1} attempts: {last_err!r}")


def run(coro):
    """Run a coroutine from sync code; works inside Jupyter/Colab (which already has a running loop)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(1) as ex:
        return ex.submit(asyncio.run, coro).result()
