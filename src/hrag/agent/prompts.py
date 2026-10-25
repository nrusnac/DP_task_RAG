"""Prompts and JSON schemas for the agentic retrieval controller."""

SYSTEM = ("You are a meticulous legal research assistant working inside one client matter's data room "
          "(deal documents, memos, e-mails, spreadsheets, board decks). Follow the output format exactly.")

ROUTER = """Classify how much retrieval this question needs (Adaptive-RAG routing).

- direct: answerable WITHOUT this matter's documents (greetings, questions about you, generic definitions or general
  law not tied to this matter's facts).
- simple: about ONE topic that one search can cover: a figure, date, party, clause, policy or position, including
  2-3 closely related facts and a short recommendation based on them.
- complex: needs facts on SEVERAL distinct topics or documents that one search cannot cover, or a broad
  synthesis (comparing markets/parties/positions, a full issues analysis or strategy across topics).
When in doubt, choose simple.

Matter: {matter}
Question: {question}"""
ROUTER_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["route", "reason"],
                 "properties": {"route": {"type": "string", "enum": ["direct", "simple", "complex"]},
                                "reason": {"type": "string"}}}

PLAN = """Decompose the question into 2-5 self-contained sub-questions for document search. Each sub-question must be
answerable from a single passage, name the specific entities, markets, documents or terms involved, and together they
must cover everything the question asks.{gaps}

Matter: {matter}
Question: {question}"""
PLAN_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["subquestions"],
               "properties": {"subquestions": {"type": "array", "items": {"type": "string"}, "minItems": 1,
                                               "maxItems": 5}}}

GRADE = """Retrieval evaluator (CRAG). For each numbered passage, decide whether it contains information that helps
answer the question (directly, or as a necessary part of the answer). Be strict: same topic but different
market/party/period is NOT relevant.

Question: {question}

{passages}

Return the numbers of the relevant passages (empty list if none)."""
GRADE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["relevant"],
                "properties": {"relevant": {"type": "array", "items": {"type": "integer"}}}}

REWRITE = """A document search for this question returned weak results (CRAG corrective step).
Write:
1. keywords: a BM25 keyword query of 5-12 terms likely to appear verbatim in the document (names and terms from the
   question, synonyms, defined-term and section/heading vocabulary), different from the failed query. NEVER invent
   names or figures here: BM25 matches literally, so invented terms only add noise;
2. hyde: a hypothetical 60-100 word passage written as the matter's document that would contain the answer (HyDE).
   Invented details are fine: it is only a search probe.

Question: {question}
Failed query: {query}
Best results so far (judged insufficient):
{seen}"""
REWRITE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["keywords", "hyde"],
                  "properties": {"keywords": {"type": "string"}, "hyde": {"type": "string"}}}

SUFFICIENCY = """Given the question and the evidence gathered so far (grouped by sub-question), decide whether the
evidence is sufficient. It is sufficient unless some PART of the question is not addressed by any evidence at all
(more detail on a part that is already covered does not count). If not, list up to 2 NEW sub-questions for the
unaddressed parts only.
Only ask for information that THIS MATTER'S OWN DOCUMENTS could contain (deal facts, figures, parties' statements,
internal analyses). Never ask for current law, updated thresholds, market news or anything outside the data room;
the answer will flag those as needing outside verification. Do not repeat earlier sub-questions. Leave the list
empty if nothing more could plausibly be found in the documents.

Question: {question}

{evidence}"""
SUFFICIENCY_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["sufficient", "missing"],
                      "properties": {"sufficient": {"type": "boolean"},
                                     "missing": {"type": "array", "items": {"type": "string"}, "maxItems": 2}}}

GENERATE_SYSTEM = ("You are a senior associate at a law firm (antitrust, M&A and regulatory matters). You answer "
                   "strictly from the evidence provided from the client's data room. Precision matters more than length.")
# The rules after the first four target the unfaithful answers seen in the eval (docs/05-evaluation.md, "Unfaithful-answer
# patterns").
GENERATE = """Answer the question using ONLY the numbered evidence below.

Rules
- Cite every factual statement with its evidence number(s), e.g. [2] or [1][4].
- Quote figures, dates, names and section numbers exactly as written in the evidence.
- If the evidence does not establish something the question asks, say so explicitly ("The documents provided do not
  establish ...") instead of guessing. Never present outside knowledge as coming from the documents.
- Keep document facts and your own reasoning apart: facts carry citations; any legal inference, recommendation,
  characterization, comparison or conclusion goes in a section or sentence starting with "Analysis:" (no citation
  needed there).
- Headings, short answers and bottom lines only summarize cited facts or your Analysis; they never add a new claim.
- No exclusivity or completeness claims ("the only", "all", "none", "no other", "never") unless the evidence itself
  says so; write "the documents provided show ..." instead.
- Keep proposals, drafts, redlines, requests and negotiating positions apart from agreed or executed terms: write
  "the markup proposes ... [3]", not "the agreement provides ...".
- Describe signatures, dates, execution status and markings exactly as shown: a blank signature line means the
  document is unsigned; a "DRAFT" marking means it is a draft.
- Attribute each fact only to the document or party that the evidence shows stating it.
- Answer exactly what is asked, concisely and structured (short paragraphs or bullets).
- Before finishing, re-read the answer and delete any sentence that has no citation and is neither "Analysis:" nor
  an explicit "The documents provided do not establish ...".
{insufficient}
Matter: {matter}
Question: {question}

Evidence:
{evidence}"""

DIRECT = """Answer briefly. You have NOT consulted any of the matter's documents for this answer: if the question
concerns the matter's specific facts, say that document retrieval is needed.

Question: {question}"""
