"""LLM-as-judge: Harvey LAB's own rubric prompt (correctness) + a faithfulness check (groundedness).

The correctness prompt is copied verbatim from harveyai/harvey-labs (MIT):
lab_core/evaluation/prompts/rubric_criterion.txt. The verdict schema matches lab_core/evaluation/judge.py
(reasoning first, then verdict).
"""
from __future__ import annotations

from .. import llm
from ..config import CFG

HARVEY_RUBRIC_PROMPT = """You are evaluating a legal AI agent's work product against a specific quality criterion.

## Task
{task_description}

## Agent's Output
{agent_output}

## Criterion
**{criterion_title}**

{match_criteria}

## Instructions
Evaluate the agent's output against the criterion above.
- **PASS**: The agent's output satisfies the criterion as described
- **FAIL**: The agent's output does not satisfy the criterion as described

Respond with JSON only:

```json
{{
  "reasoning": "Brief explanation",
  "verdict": "pass" | "fail"
}}
```"""
VERDICT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["reasoning", "verdict"],
                  "properties": {"reasoning": {"type": "string"}, "verdict": {"type": "string", "enum": ["pass", "fail"]}}}

FAITHFUL_PROMPT = """Check whether the answer is grounded in the evidence. A claim is unsupported if the evidence does
not state it (general legal background that is clearly marked as such is fine; an explicit "the documents do not
establish X" is fine).

Evidence:
{evidence}

Answer:
{answer}

List the unsupported factual claims (empty if none) and give a verdict."""
FAITHFUL_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["unsupported", "faithful"],
                   "properties": {"unsupported": {"type": "array", "items": {"type": "string"}},
                                  "faithful": {"type": "boolean"}}}


async def judge_correctness(item: dict, answer: str, model: str | None = None) -> dict:
    task = (f"Answer this question about the matter, using the matter's data room: {item['question']}\n"
            "(The criterion below was written for a full memo on this matter; judge whether THIS answer satisfies "
            "the substance of the criterion for the question asked.)")
    out = await llm.complete(HARVEY_RUBRIC_PROMPT.format(task_description=task, agent_output=answer or "(empty)",
                                                         criterion_title=item["criterion_title"],
                                                         match_criteria=item["match_criteria"]),
                             system="You are a strict, fair grader of legal work product.",
                             model=model or CFG.strong_llm, schema=VERDICT_SCHEMA, purpose="judge")
    return dict(verdict=out.get("verdict"), reasoning=out.get("reasoning", ""))


async def judge_faithfulness(answer: str, evidence: list[dict], model: str | None = None) -> dict:
    ev = "\n\n".join(f"[{i}] {e.get('doc', '')}: {e['text']}" for i, e in enumerate(evidence, 1)) or "(none)"
    out = await llm.complete(FAITHFUL_PROMPT.format(evidence=ev[:60000], answer=answer or "(empty)"),
                             model=model or CFG.fast_llm, schema=FAITHFUL_SCHEMA, purpose="judge_faithful")
    return dict(faithful=bool(out.get("faithful")), unsupported=out.get("unsupported", []))
