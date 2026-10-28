"""LLM-as-judge (docs/05-evaluation.md, "Metrics").

- judge_correctness: Harvey LAB's own rubric prompt, copied verbatim from harveyai/harvey-labs (MIT):
  lab_core/evaluation/prompts/rubric_criterion.txt. The verdict schema matches lab_core/evaluation/judge.py
  (reasoning first, then verdict).
- judge_claims: claim-level groundedness against the evidence the answer was given. It replaces a binary Haiku verdict
  that, on the 78 test_v2 questions where both systems had identical input, disagreed with itself 44% of the time.
- judge_reference: correctness against gold passages and an expected answer known by construction (hard sets).
- judge_pairwise: which of two answers is better, shown in a seeded random order.
"""
from __future__ import annotations

from .. import llm
from ..config import CFG

JUDGE_SYSTEM = "You are a strict, fair grader of legal work product."


def numbered(passages: list[dict], limit: int = 80000) -> str:
    """[n] doc: text, numbered from 1 as the answer cites them."""
    return "\n\n".join(f"[{i}] {p.get('doc', '')}: {p['text']}" for i, p in enumerate(passages, 1))[:limit] or "(none)"

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


async def judge_correctness(item: dict, answer: str, model: str | None = None) -> dict:
    task = (f"Answer this question about the matter, using the matter's data room: {item['question']}\n"
            "(The criterion below was written for a full memo on this matter; judge whether THIS answer satisfies "
            "the substance of the criterion for the question asked.)")
    out = await llm.complete(HARVEY_RUBRIC_PROMPT.format(task_description=task, agent_output=answer or "(empty)",
                                                         criterion_title=item["criterion_title"],
                                                         match_criteria=item["match_criteria"]),
                             system=JUDGE_SYSTEM, model=model or CFG.strong_llm, schema=VERDICT_SCHEMA,
                             purpose="judge")
    return dict(verdict=out.get("verdict"), reasoning=out.get("reasoning", ""))


CLAIMS_PROMPT = """You are checking a legal research answer against the numbered evidence it was given.

List every factual statement the answer makes about THIS MATTER (facts, figures, dates, parties, what a document
says, positions, execution status). Skip reasoning or advice in a section or sentence labelled "Analysis:", general
legal background labelled as such, and statements that the documents do not establish something. Merge repetitions.

For each statement give:
- claim: the statement (shortened if long)
- cites: the evidence numbers the answer cites for it ([] if none)
- verdict: "supported" (the evidence states it, or it follows by simple arithmetic or direct restatement),
  "unsupported" (the evidence does not state it) or "contradicted" (the evidence states otherwise)
- material: true if the error would mislead a lawyer on the question asked: a wrong or invented figure, date, party,
  clause content or execution status, or a claim the answer's conclusion relies on; false for wording, labels,
  characterisations and attributions that do not change the answer

Evidence:
{evidence}

Answer:
{answer}"""
CLAIMS_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["claims"], "properties": {"claims": {
    "type": "array", "items": {"type": "object", "additionalProperties": False,
                               "required": ["claim", "cites", "verdict", "material"],
                               "properties": {"claim": {"type": "string"},
                                              "cites": {"type": "array", "items": {"type": "integer"}},
                                              "verdict": {"type": "string",
                                                          "enum": ["supported", "unsupported", "contradicted"]},
                                              "material": {"type": "boolean"}}}}}}


async def judge_claims(answer: str, evidence: list[dict], model: str | None = None) -> dict:
    """claim_support = supported / all claims; faithful = no MATERIAL unsupported or contradicted claim (a mislabelled
    title or heading no longer flips the whole answer)."""
    out = await llm.complete(CLAIMS_PROMPT.format(evidence=numbered(evidence), answer=answer or "(empty)"),
                             system=JUDGE_SYSTEM, model=model or CFG.strong_llm, schema=CLAIMS_SCHEMA,
                             purpose="judge_claims", max_tokens=8000)
    claims = out.get("claims", [])
    bad = [c for c in claims if c.get("verdict") != "supported"]
    material = [c for c in bad if c.get("material")]
    return dict(claims=claims, n_claims=len(claims),
                claim_support=round(1 - len(bad) / len(claims), 3) if claims else None,
                material_errors=len(material), faithful=not material, unsupported=[c.get("claim", "") for c in bad])


REFERENCE_PROMPT = """You are grading an answer to a question about a legal matter. The reference answer and the
passages it was written from are correct by construction.

Question: {question}

Reference answer: {expected}

Facts a correct answer needs:
{facts}

Passages:
{passages}

Answer to grade:
{answer}

For each needed fact, say whether the answer states it (same meaning; exact wording not required). Then give a
verdict: "correct" (states every needed fact and reaches the reference's conclusion, and asserts nothing the passages
contradict), "partial" (misses some needed facts or the conclusion, but asserts nothing the passages contradict), or
"incorrect" (misses the point, or asserts something the passages contradict)."""
REFERENCE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["reasoning", "facts", "verdict"],
                    "properties": {"reasoning": {"type": "string"},
                                   "facts": {"type": "array", "items": {
                                       "type": "object", "additionalProperties": False, "required": ["n", "present"],
                                       "properties": {"n": {"type": "integer"}, "present": {"type": "boolean"}}}},
                                   "verdict": {"type": "string", "enum": ["correct", "partial", "incorrect"]}}}


async def judge_reference(item: dict, answer: str, passages: list[dict], model: str | None = None) -> dict:
    facts = item.get("facts") or []
    out = await llm.complete(REFERENCE_PROMPT.format(
        question=item["question"], expected=item["expected_answer"],
        facts="\n".join(f"{i}. {f}" for i, f in enumerate(facts, 1)) or "(those in the reference answer)",
        passages=numbered(passages), answer=answer or "(empty)"),
        system=JUDGE_SYSTEM, model=model or CFG.strong_llm, schema=REFERENCE_SCHEMA, purpose="judge_reference")
    present = {f["n"] for f in out.get("facts", []) if f.get("present") and 1 <= f.get("n", 0) <= len(facts)}
    return dict(reference_verdict=out.get("verdict"), reference_reasoning=out.get("reasoning", ""),
                fact_coverage=round(len(present) / len(facts), 3) if facts else None)


PAIRWISE_PROMPT = """Two research systems answered the same question about a legal matter. Compare the answers on:
1. correctness against the documents below (a wrong, invented or unsupported fact counts heavily);
2. whether they answer everything the question asks;
3. usefulness to a lawyer (a clear, committed answer where the documents allow one).
Ignore length and formatting unless they hide or omit the answer.

Question: {question}
{reference}
Documents (everything either system retrieved):
{evidence}

Answer 1:
{first}

Answer 2:
{second}

Which answer is better overall: "1", "2", or "tie" (only when neither is better in a way that matters)?"""
PAIRWISE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["reasoning", "winner"],
                   "properties": {"reasoning": {"type": "string"},
                                  "winner": {"type": "string", "enum": ["1", "2", "tie"]}}}


async def judge_pairwise(question: str, reference: str, first: str, second: str, evidence: list[dict],
                         model: str | None = None) -> dict:
    """reference: a line such as 'Reference answer: ...' (may be empty). Returns {"winner": "1"|"2"|"tie", ...}."""
    out = await llm.complete(PAIRWISE_PROMPT.format(question=question, reference=reference,
                                                    evidence=numbered(evidence), first=first or "(empty)",
                                                    second=second or "(empty)"),
                             system=JUDGE_SYSTEM, model=model or CFG.strong_llm, schema=PAIRWISE_SCHEMA,
                             purpose="judge_pairwise")
    return dict(winner=out.get("winner"), reasoning=out.get("reasoning", ""))
