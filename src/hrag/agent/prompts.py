"""Prompts and JSON schemas for the agentic retrieval controller."""

SYSTEM = ("You are a meticulous legal research assistant working inside one client matter's data room "
          "(deal documents, memos, e-mails, spreadsheets, board decks). Follow the output format exactly.")

ROUTER = """Classify how much retrieval this question needs (Adaptive-RAG routing).

- direct: answerable WITHOUT this matter's documents (greetings, questions about you, generic definitions or general
  law not tied to this matter's facts).
- simple: needs ONE specific fact or passage from the documents (a figure, date, party, clause, single statement).
- complex: needs SEVERAL facts from different places, comparison, aggregation, multi-step reasoning or legal analysis.

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
1. keywords: a BM25 keyword query of 5-12 terms (exact names, figures, defined terms, section/heading words likely to
   appear verbatim in the document), different from the failed query;
2. hyde: a hypothetical 60-100 word passage written as the matter's document that would contain the answer (HyDE).
   Invented details are fine: it is only a search probe.

Question: {question}
Failed query: {query}
Best results so far (judged insufficient):
{seen}"""
REWRITE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["keywords", "hyde"],
                  "properties": {"keywords": {"type": "string"}, "hyde": {"type": "string"}}}

SUFFICIENCY = """Given the question and the evidence gathered so far (grouped by sub-question), decide whether the
evidence is sufficient for a complete, accurate answer. If not, list up to 3 NEW sub-questions for what is missing
(do not repeat earlier sub-questions; leave the list empty if nothing more could plausibly be found).

Question: {question}

{evidence}"""
SUFFICIENCY_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["sufficient", "missing"],
                      "properties": {"sufficient": {"type": "boolean"},
                                     "missing": {"type": "array", "items": {"type": "string"}, "maxItems": 3}}}

GENERATE_SYSTEM = ("You are a senior antitrust associate. You answer strictly from the evidence provided from the "
                   "client's data room. Precision matters more than length.")
GENERATE = """Answer the question using ONLY the numbered evidence below.

Rules
- Cite every factual statement with its evidence number(s), e.g. [2] or [1][4].
- Quote figures, dates, names and section numbers exactly as written in the evidence.
- If the evidence does not establish something the question asks, say so explicitly ("The documents provided do not
  establish ...") instead of guessing. Never present outside knowledge as coming from the documents.
- Be concise and structured (short paragraphs or bullets).
{insufficient}
Matter: {matter}
Question: {question}

Evidence:
{evidence}"""

DIRECT = """Answer briefly. You have NOT consulted any of the matter's documents for this answer: if the question
concerns the matter's specific facts, say that document retrieval is needed.

Question: {question}"""
