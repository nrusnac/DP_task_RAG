# 01 · Dataset: Harvey LAB, `antitrust-competition` + 13 deal-regulatory matters

**Source:** [harveyai/harvey-labs](https://github.com/harveyai/harvey-labs) (MIT), a benchmark of ~2,000 legal agent
tasks across 24 practice areas. Each task is a folder with `task.json` and `documents/`, a mock **data room**.

## Why this subset
We index **one whole practice area plus its closest neighbours**, not a handful of tasks.
- **Big enough that RAG matters:** 44 matters, 318 documents, **~3.05M tokens**. That's far more than any context window, and even one matter is median **61k** tokens and up to **199k**.
- **Small enough to index on a free T4** in minutes (~8.6k chunks).
- **Coherent:** antitrust (merger control, cartels/leniency, protective orders, consent decrees) plus deal-regulatory work
  that overlaps with it, so cross-matter look-alikes (other matters' HHI tables, other deals' filing analyses) are
  realistic distractors.

### The 13 added matters (`config.extra_tasks`)
All 31 antitrust matters with documents were already indexed, so growing the corpus (+38% documents, +50% chunks)
meant adding matters from other practice areas. They were chosen by how much antitrust/regulatory content they share
(mentions of HSR, merger control, competition, CFIUS in `task.json`), and mostly from non-`draft-*` work types, because
only those produce eval questions (`draft-*` criteria grade the lawyer's own deliverable):

| Practice area | Matters | Docs | Why |
|---|---|---|---|
| `corporate-ma` | merger-filing requirements across jurisdictions, merger-control thresholds, seller's SPA markup, closing conditions, counterparty SPA markup, material-contract review, issues in a counterparty's draft SPA, *HSR notification draft* (documents only) | 53 | merger control / HSR filings and antitrust closing conditions are the M&A side of antitrust |
| `international-trade-sanctions` | national-security factors of a defense acquisition, deficiencies in a CFIUS voluntary notice, issues in a CFIUS mandatory declaration, counterparty markup of a CFIUS mitigation agreement, mitigation agreement vs. national-security terms | 35 | CFIUS review is the other regulatory clearance a deal needs; same document types (filings, markups, agreements) |

Retrieval is scoped to one matter's data room, so the added matters don't make an antitrust question easier or harder
in scoped search. What they add: **16 matters the eval set has never seen** (12 new + 4 unused antitrust ones, the
`test` split in [05-evaluation.md](05-evaluation.md)), more corpus-wide distractors, and harder same-area negatives.

## What a task contains
| Field | Example |
|---|---|
| `instructions` | "Review the attached deal documents ... prepare an antitrust issues memo covering risks, remedies, and filing strategy." |
| `criteria` | 27–161 PASS/FAIL rubric items, e.g. *"PASS if the memo references the post-merger HHI of 3,552 and/or the HHI delta of 1,242 for flame retardants in the South Central region."* |
| `documents/` | 5–19 files: memos, SPAs, board decks, market-share spreadsheets, e-mail chains |

Work types: 23 *analyze*, 13 *draft*, 5 *review*, 3 *research*.

## Measured corpus statistics (`artifacts/ingest_stats.json`)
| | |
|---|---|
| Documents | 318 (228 docx, 39 eml, 36 xlsx, 14 pptx, 1 txt): antitrust 230, corporate-ma 53, trade-sanctions 35 |
| Tokens (BGE-M3 tokenizer) | ~3.05M (antitrust 2.09M) |
| Doc length | median 7.9k tokens, p90 16.9k, max 80k |
| Chunks | 8,627 (median 339 tokens, p95 593). By kind: 6,658 text, 1,368 table, 378 slide, 223 e-mail. Antitrust 5,756, corporate-ma 2,032, trade-sanctions 839 |

The 5,756 antitrust chunks are byte-identical to the ones before the corpus grew (same ids, text and order), so the
existing silver labels and dev results stay valid. Source pinned at harvey-labs `e9b1f9f` (`ingest_stats.json`).

## Data-quality findings that shaped the parser
Found during exploration (`prep/colab_smoke_test.ipynb`) and handled in `src/hrag/ingest/parsers.py`:
1. **Tracked changes:** `doj-redline-markup.docx` has 78 insertions and 85 deletions. `python-docx` silently drops inserted text, so we parse the XML directly and render `[INSERTED: …]` / `[DELETED: …]`. That task is precisely about analysing the counterparty's markup.
2. **Malformed XML:** `jv-business-plan-final.docx` contains an unescaped `&`, so strict parsers crash. We use `lxml` with `recover=True`. The same bug appears in a spreadsheet of the added matters (`market-share-analysis.xlsx`); openpyxl has no recover mode, so the workbook's XML is re-zipped with bare `&` escaped and loaded again.
3. **Formula cells without cached values:** 22 cells in 4 spreadsheets read as empty with `data_only=True`. We keep the formula text instead of losing the cell.
4. **Unhelpful detected titles:** first headings are often confidentiality banners ("PRIVILEGED AND CONFIDENTIAL — ATTORNEY WORK PRODUCT") or letterheads. The filenames are descriptive, so the chunk header uses the humanised filename and appends the detected title only when it isn't boilerplate.
5. **No retrieval labels:** `criteria[].sources` is empty everywhere, so we derive silver labels (see [05-evaluation.md](05-evaluation.md)).
