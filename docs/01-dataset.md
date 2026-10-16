# 01 · Dataset: Harvey LAB, `antitrust-competition` subset

**Source:** [harveyai/harvey-labs](https://github.com/harveyai/harvey-labs) (MIT), a benchmark of ~2,000 legal agent
tasks across 24 practice areas. Each task is a folder with `task.json` and `documents/`, a mock **data room**.

## Why this subset
We index **one whole practice area**, not a handful of tasks.
- **Big enough that RAG matters:** 31 matters, 230 documents, **~2.1M tokens**. That's far more than any context window, and even one matter is median **58k** tokens and up to **198k**.
- **Small enough to index on a free T4** in minutes (~5.8k chunks).
- **Coherent:** one legal domain (merger control, cartels/leniency, protective orders, consent decrees), so cross-matter look-alikes (other matters' HHI tables) are realistic distractors.

## What a task contains
| Field | Example |
|---|---|
| `instructions` | "Review the attached deal documents ... prepare an antitrust issues memo covering risks, remedies, and filing strategy." |
| `criteria` | 27–161 PASS/FAIL rubric items, e.g. *"PASS if the memo references the post-merger HHI of 3,552 and/or the HHI delta of 1,242 for flame retardants in the South Central region."* |
| `documents/` | 5–19 files: memos, SPAs, board decks, market-share spreadsheets, e-mail chains |

Work types in the subset: 16 *analyze*, 12 *draft*, 2 *research*, 1 *review*.

## Measured corpus statistics (`artifacts/ingest_stats.json`)
| | |
|---|---|
| Documents | 230 (161 docx, 27 xlsx, 29 eml, 11 pptx, 2 txt) |
| Tokens (BGE-M3 tokenizer) | ~2.09M |
| Doc length | median 7.6k tokens, p90 14.6k, max 78k |
| Chunks | 5,756 (median 358 tokens, p95 594). By kind: 4,283 text, 1,012 table, 309 slide, 152 e-mail |

## Data-quality findings that shaped the parser
Found during exploration (`prep/colab_smoke_test.ipynb`) and handled in `src/hrag/ingest/parsers.py`:
1. **Tracked changes:** `doj-redline-markup.docx` has 78 insertions and 85 deletions. `python-docx` silently drops inserted text, so we parse the XML directly and render `[INSERTED: …]` / `[DELETED: …]`. That task is precisely about analysing the counterparty's markup.
2. **Malformed XML:** `jv-business-plan-final.docx` contains an unescaped `&`, so strict parsers crash. We use `lxml` with `recover=True`.
3. **Formula cells without cached values:** 22 cells in 4 spreadsheets read as empty with `data_only=True`. We keep the formula text instead of losing the cell.
4. **Unhelpful detected titles:** first headings are often confidentiality banners ("PRIVILEGED AND CONFIDENTIAL — ATTORNEY WORK PRODUCT") or letterheads. The filenames are descriptive, so the chunk header uses the humanised filename and appends the detected title only when it isn't boilerplate.
5. **No retrieval labels:** `criteria[].sources` is empty everywhere, so we derive silver labels (see [05-evaluation.md](05-evaluation.md)).
