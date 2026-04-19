# LMB History & Civics — RAG Storage Layer

A retrieval layer over the La Martiniere For Boys (Kolkata) History & Civics
past-paper question bank. Turns a single enriched JSON file into a queryable
**SQLite + FAISS** store that a downstream paper-generation pipeline can hit
without touching the raw JSON.

> **Scope is deliberately narrow.** This repo builds and queries the indexes.
> It does **not** generate papers, call any LLM, or expose a web API.

---

## Table of contents

- [What this project is](#what-this-project-is)
- [What it is *not*](#what-it-is-not)
- [The dataset](#the-dataset)
- [Architecture](#architecture)
- [Repo layout](#repo-layout)
- [Install & build](#install--build)
- [How the builder works](#how-the-builder-works)
- [Query API](#query-api)
- [Retrieval patterns](#retrieval-patterns)
- [Database schema](#database-schema)
- [Vector index](#vector-index)
- [Smoke test](#smoke-test)
- [Design decisions](#design-decisions)
- [When to outgrow this](#when-to-outgrow-this)
- [Troubleshooting](#troubleshooting)

---

## What this project is

A two-stage index built from one source file:

```
lmb_history_civics_question_bank_enriched.json    ← canonical source
                 │
                 ▼
           build_index.py
                 │
        ┌────────┴────────┐
        ▼                 ▼
   SQLite DB          FAISS index
   (metadata,         (semantic
    filters)           similarity)
        │                 │
        └────────┬────────┘
                 ▼
           retrieve.py
        (query helpers,
         no LLM calls)
```

Downstream code gets two primitives:

1. **Structured filter** — "give me all 1-mark MCQs for Class X, difficulty=easy."
2. **Semantic search** — "find the 5 questions closest in meaning to this one."

Everything else (paper blueprinting, weighted sampling, LLM drafting) lives
outside this repo.

## What it is *not*

- Not a paper generator. No blueprint logic, no sampling, no LLM.
- Not a web service. No FastAPI, no REST, no auth.
- Not a migration-managed DB. The builder wipes `data/` and rebuilds on
  every run. There is no schema evolution path — the source JSON *is* the
  migration.
- Not an ORM project. Raw SQL via `sqlite3`, parameterized with `?`.
- Not a multi-process store. SQLite file + FAISS file, both local.

---

## The dataset

**Source:** `lmb_history_civics_question_bank_enriched.json`
**Coverage:** 2022-23 → 2024-25 past papers, classes VI–XI.

### Top-level shape

```json
{
  "metadata": { ... },
  "standalone_questions": [ ...315 items... ],
  "context_groups":       [ ...58 items... ]
}
```

### A standalone question

Self-contained; has a `type` and `marks`.

```json
{
  "id": "MCQ-001",
  "type": "MCQ",
  "marks": 1,
  "question": "When a large number of voters choose their representatives...",
  "source": {
    "school": "La Martiniere For Boys, Kolkata",
    "class": "IX",
    "exam_type": "Annual",
    "year": "2024-25"
  },
  "topic": "Elections",
  "subtopic": "Types of Election",
  "difficulty": "easy",
  "bloom_level": "recall"
}
```

Optional extras on some rows: `options` (for MCQs), `note`.

### A context group

A shared stimulus (thematic prompt, image, map, or quote) with linked
sub-questions. Sub-questions inherit `topic` and source fields from the
group; they have no `type` of their own.

```json
{
  "id": "CG-001",
  "context_type": "thematic",
  "context_description": "Mandate of Lok Sabha Election 2024...",
  "topic": "Elections – Election Commission",
  "total_marks": 10,
  "source": { "class": "IX", "year": "2024-25", ... },
  "questions": [
    { "id": "CG-001_Q1", "sub_label": "i",  "marks": 3, "question": "...",
      "difficulty": "medium", "bloom_level": "recall" },
    { "id": "CG-001_Q2", "sub_label": "ii", "marks": 3, "question": "...", ... },
    { "id": "CG-001_Q3", "sub_label": "iii","marks": 4, "question": "...", ... }
  ]
}
```

### Value domains

| Field         | Values                                                               |
|---------------|----------------------------------------------------------------------|
| `type`        | MCQ · Short Answer · Long Answer · Fill in the Blank · Identify · True/False · Definition |
| `context_type`| thematic · image · map · quote                                       |
| `class`       | VI · VII-VIII · IX · X · XI                                          |
| `year`        | `YYYY-YY` academic format (e.g. `2024-25`), or `"Unknown"`           |
| `difficulty`  | easy · medium · hard                                                 |
| `bloom_level` | recall · understand · apply · analyze                                |
| `exam_type`   | Annual (predominant)                                                 |

### ID conventions

- Standalone: `{TYPE}-NNN` — e.g. `MCQ-001`, `DEF-007`, `ID-024`.
- Group: `CG-NNN` — e.g. `CG-042`.
- Sub-question: `{CG-id}_Q{n}` — e.g. `CG-001_Q3`.

### Quirks worth knowing

- **Duplicates exist in the source.** `find_duplicates(0.95)` surfaces 21
  pairs, several at cosine 1.0 (e.g. `DEF-002`/`DEF-007`,
  `MCQ-014`/`MCQ-090`). Preserved, not hidden.
- `subtopic` is only populated on standalone rows — always `None` on
  sub-questions.
- Some `year` values are the literal string `"Unknown"`.

---

## Architecture

### Two stores, one identity graph

```
           ┌──────────────────────────────────┐
           │  questions.db  (SQLite)          │
           │                                  │
           │  questions(id PK, kind, type,    │
           │    marks, question_text, topic,  │
           │    subtopic, difficulty,         │
           │    bloom_level, class, year,     │
           │    exam_type, school,            │
           │    context_group_id FK,          │
           │    sub_label, text_hash)         │
           │                                  │
           │  context_groups(id PK, ...)      │
           └──────────────┬───────────────────┘
                          │ qid
           ┌──────────────┴───────────────────┐
           │  faiss_ids.json                  │
           │  ["MCQ-001", "MCQ-002", ...]     │
           │   ↑ row index = list position    │
           └──────────────┬───────────────────┘
                          │ row_idx
           ┌──────────────┴───────────────────┐
           │  faiss.index  (IndexFlatIP)      │
           │  501 vectors × 384 dims (fp32)   │
           │  L2-normalized (cosine == IP)    │
           └──────────────────────────────────┘
```

FAISS stores **only vectors**. Every piece of human-readable metadata lives
in SQLite. The JSON sidecar is the bridge between a FAISS row number and a
question id.

### Full query round-trip

`similar_to("MCQ-001", k=3)`:

1. Look up `"MCQ-001"` in `faiss_ids.json` → row `0`.
2. `index.reconstruct(0)` → the 384-dim vector (no re-embedding).
3. `index.search(vec, k + overfetch)` → row indices + cosine scores.
4. Map row indices back to qids via the JSON list.
5. Drop excluded ids (incl. self), drop below `min_score`, truncate to `k`.
6. (Optional) For each qid, `SELECT * FROM questions WHERE id = ?` in SQLite.

---

## Repo layout

```
.
├── lmb_history_civics_question_bank_enriched.json   # canonical source (never modified)
├── build_index.py          # one-shot: JSON → SQLite + FAISS
├── retrieve.py             # query helpers, no LLM
├── schema.sql              # readable copy of the SQLite schema
├── requirements.txt        # pinned deps
├── README.md               # this file
└── data/                   # generated; wiped & rebuilt every run
    ├── questions.db
    ├── faiss.index
    └── faiss_ids.json
```

`data/` is disposable. Treat it as a build artifact — do not hand-edit,
do not commit to long-lived branches, rebuild whenever the JSON changes.

---

## Install & build

Requires **Python 3.11+**.

```bash
pip install -r requirements.txt
python build_index.py
```

Expected output:

```
Loading embedding model: sentence-transformers/all-MiniLM-L6-v2
Embedding 501 question texts...
---- build summary ----
context_groups rows : 58
questions rows      : 501 (standalone=315, context_sub=186)
faiss vectors       : 501 (dim=384)
db                  : .../data/questions.db
index               : .../data/faiss.index
build time          : ~11s
```

Pinned deps (see `requirements.txt`):

- `faiss-cpu==1.8.0` — vector index
- `sentence-transformers==3.0.1` — embedding model loader
- `numpy==1.26.4` — pinned for faiss-cpu wheel compatibility

---

## How the builder works

`build_index.py` is a single linear script. No CLI flags.

1. **Wipe.** Delete `data/` if present, recreate it empty.
2. **Load source JSON.** Raise immediately if the file is missing.
3. **Open SQLite** (`data/questions.db`), turn foreign keys on, execute
   `schema.sql`.
4. **Validate and transform each row.** Required-field sets are checked
   explicitly. If any row is missing an expected field, raise with the
   offending id — malformed rows are never silently skipped.
5. **Insert via `executemany`** — one batch for `context_groups`, one for
   `questions`. Sub-questions are promoted to first-class rows in
   `questions` with `kind='context_sub'` and inherit the group's topic and
   source fields.
6. **Load the embedding model** (`all-MiniLM-L6-v2`, downloads on first
   run, ~90 MB).
7. **Encode all 501 question strings in one batch** with
   `normalize_embeddings=True` → `(501, 384)` float32 array.
8. **Build the FAISS index**: `IndexFlatIP(384)`, `.add(vectors)`,
   `faiss.write_index(...)`.
9. **Write `faiss_ids.json`** — same order as vectors were added.
10. **Print summary.** Row counts per table, vector count, wall-clock time.

### Embedding input format

```
"{topic} — {subtopic}: {question_text}"
```

Missing parts drop cleanly — no literal `"None"` ever appears in the
embedded text.

Examples:

```
Elections — Types of Election: When a large number of voters choose...
Parliament: During the proclamation of Emergency, the ______ guaranteed...   # no subtopic
When the constitution was written, ...                                        # no topic
```

### `text_hash`

Stored per row. SHA-1 of the question text after lowercasing, collapsing
whitespace, and stripping trailing punctuation. Used for cheap exact-match
duplicate detection (a cheaper alternative to semantic dedupe).

---

## Query API

From `retrieve.py`. All functions accept optional `db_path` / `index_path`
overrides; default to `data/` in the repo root.

```python
from retrieve import (
    filter_questions, get_question,
    get_context_group, filter_context_groups,
    similar_to, find_duplicates,
)
```

### `filter_questions(...)`

Structured SQL filter. Every filter accepts either a scalar or a list.

```python
filter_questions(
    *, type=None, marks=None, class_=None, year=None,
    topic=None, difficulty=None, bloom_level=None,
    kind=None,                    # 'standalone' | 'context_sub' | None (both)
    exclude_ids=None, limit=None,
) -> list[dict]
```

### `get_question(qid)`

Single row as a dict, or `None`.

### `get_context_group(cg_id, include_questions=True)`

Returns the group row plus its sub-questions ordered by `sub_label`.
Pass `include_questions=False` for a light-weight lookup.

### `filter_context_groups(...)`

```python
filter_context_groups(
    *, class_=None, year=None, topic=None,
    total_marks=None,             # int (exact) or (min, max) tuple
    exclude_ids=None,
) -> list[dict]
```

Does **not** include sub-questions in the result (keep it cheap; call
`get_context_group` if you need them).

### `similar_to(qid_or_text, k=10, exclude_ids=None, min_score=None)`

Semantic search. Returns `[(question_id, cosine_score)]`.

- If the argument matches an existing id, reuses the stored embedding via
  `index.reconstruct` (no re-encoding).
- Otherwise, embeds the raw text on the fly with the same model.
- Input id auto-excluded from its own neighbours.
- Overfetches to keep `len(result) == k` even after exclusions / min_score.

### `find_duplicates(threshold=0.9)`

Whole-bank pairwise cosine scan. Returns `[(id_a, id_b, score)]` with
`id_a < id_b`, sorted by score descending. Single matmul on normalized
vectors — fast at this scale.

### `load_index(...)`

Memoized helper (`@lru_cache`). Loads and caches the FAISS index and id
list so repeated queries don't re-read the file.

---

## Retrieval patterns

Three realistic ways downstream code will consume this layer.

### Pattern 1 — Blueprint-driven paper build

Pure structured. No vectors.

```python
for slot in blueprint:
    pool = filter_questions(
        type=slot.type, marks=slot.marks, class_=slot.class_,
        difficulty=slot.difficulty, exclude_ids=already_used,
    )
    picks = random.sample(pool, slot.count)
    already_used.update(p["id"] for p in picks)
```

### Pattern 2 — Semantic dedupe during picking

Structured pool, then reject any candidate too close to something already
in the paper.

```python
candidate = pool[0]
neighbours = similar_to(candidate["id"], k=10, min_score=0.85)
if any(nid in already_used for nid, _ in neighbours):
    continue   # reject, try next
```

### Pattern 3 — Hybrid: "similar, but filtered"

FAISS can't filter by `class`/`marks`; SQL can't rank by similarity. Compose
them: semantic first (ranked), hard filter second (binary set intersection).

```python
candidates = similar_to("MCQ-001", k=50, exclude_ids=already_used)
allowed = {r["id"] for r in filter_questions(type="MCQ", marks=1, class_="X")}
picks = [(qid, s) for qid, s in candidates if qid in allowed][:5]
```

### Pattern 4 — Whole-bank dedupe report

```python
pairs = find_duplicates(threshold=0.95)
# [(id_a, id_b, score), ...] sorted by score desc
```

---

## Database schema

Full SQL in [`schema.sql`](schema.sql).

### `questions`

| Column             | Type    | Notes                                          |
|--------------------|---------|------------------------------------------------|
| `id`               | TEXT PK | e.g. `MCQ-001`, `CG-001_Q3`                    |
| `kind`             | TEXT    | `'standalone'` or `'context_sub'`              |
| `type`             | TEXT    | NULL for sub-questions                         |
| `marks`            | INTEGER | per sub-question for context subs              |
| `question_text`    | TEXT    |                                                |
| `topic`            | TEXT    | subs inherit from their group                  |
| `subtopic`         | TEXT    | NULL for subs                                  |
| `difficulty`       | TEXT    | easy · medium · hard                           |
| `bloom_level`      | TEXT    | recall · understand · apply · analyze          |
| `class`            | TEXT    | VI · VII-VIII · IX · X · XI                    |
| `year`             | TEXT    | `YYYY-YY` or `"Unknown"`                       |
| `exam_type`        | TEXT    |                                                |
| `school`           | TEXT    |                                                |
| `context_group_id` | TEXT FK | NULL for standalone                            |
| `sub_label`        | TEXT    | `'i'`, `'ii'`, …; NULL for standalone          |
| `text_hash`        | TEXT    | sha1 of normalized text                        |

Indexes: `(type, marks)`, `(class, year)`, `topic`, `difficulty`,
`bloom_level`, `text_hash`, `context_group_id`.

### `context_groups`

| Column                | Type    | Notes                              |
|-----------------------|---------|------------------------------------|
| `id`                  | TEXT PK | e.g. `CG-001`                      |
| `context_type`        | TEXT    | thematic · image · map · quote     |
| `context_description` | TEXT    | the stimulus itself                |
| `topic`               | TEXT    |                                    |
| `total_marks`         | INTEGER |                                    |
| `class`               | TEXT    |                                    |
| `year`                | TEXT    |                                    |
| `exam_type`           | TEXT    |                                    |
| `school`              | TEXT    |                                    |

---

## Vector index

| Property       | Value                                                 |
|----------------|-------------------------------------------------------|
| Library        | `faiss-cpu` 1.8.0                                     |
| Index type     | `IndexFlatIP` (exact brute-force, inner product)      |
| Vectors        | 501 (one per question; groups aren't embedded)        |
| Dim            | 384                                                   |
| Metric         | Cosine similarity (vectors are L2-normalized)         |
| Model          | `sentence-transformers/all-MiniLM-L6-v2`              |
| File           | `data/faiss.index` (~770 KB)                          |
| Id mapping     | `data/faiss_ids.json`                                 |

Why flat, not HNSW/IVF/PQ: at 501 vectors, exhaustive search is sub-
millisecond and deterministic. ANN structures add build params, recall
tuning, and non-determinism for no speed gain at this scale.

---

## Smoke test

Actual output from a clean build (2026-04-19):

```
### 1) filter_questions(type='MCQ', marks=1, class_='X', limit=5)
returned 5 rows
  MCQ-020  topic='Parliament'  text='Important subjects like Defence, Diplomatic Affairs...'
  MCQ-021  topic='Parliament'  text='During the proclamation of Emergency, the ______ guaranteed...'
  MCQ-022  topic='Parliament'  text='He is the defender of the government in Parliament.'
  MCQ-023  topic='President'   text='When a money bill reaches the President...'
  MCQ-024  topic='Judiciary'   text='Where can an appeal be made against a death punishment...'

### 2) get_context_group('CG-001')
{
  "id": "CG-001",
  "context_type": "thematic",
  "context_description": "Mandate of Lok Sabha Election 2024 / Role of Election Commission",
  "topic": "Elections – Election Commission",
  "total_marks": 10,
  "class": "IX",
  "year": "2024-25",
  "exam_type": "Annual",
  "school": "La Martiniere For Boys, Kolkata"
}
  [i]   CG-001_Q1  marks=3  'What constitutes the Election Commission?...'
  [ii]  CG-001_Q2  marks=3  'What is an election? Mention any two principles...'
  [iii] CG-001_Q3  marks=4  'Mention the kinds of election in India?...'

### 3) similar_to('MCQ-001', k=3)
  CG-051_Q1  score=0.7804  'Explain briefly what are types of elections that can take place?'
  MCQ-058    score=0.7587  'Mid term election takes place:'
  MCQ-077    score=0.7566  'The election to the State Legislative Assembly is an example of'

### 4) find_duplicates(threshold=0.95)
pairs found: 21
  DEF-002 <-> DEF-007  score=1.0000
  MCQ-014 <-> MCQ-090  score=1.0000
  ID-014  <-> ID-024   score=1.0000
  ID-015  <-> ID-025   score=1.0000
  DEF-003 <-> DEF-011  score=1.0000
```

Reproduce with:

```python
import retrieve as r
print(r.filter_questions(type="MCQ", marks=1, class_="X", limit=5))
print(r.get_context_group("CG-001"))
print(r.similar_to("MCQ-001", k=3))
print(len(r.find_duplicates(threshold=0.95)))
```

---

## Design decisions

**Why SQLite?** Zero setup, single file, bundled with Python. 559 rows
don't need Postgres. Trivial to swap later (`sqlite3.connect` →
`psycopg.connect`, `?` → `%s`) if the workload ever justifies it.

**Why FAISS flat?** Exact brute-force is faster than the tuning cost of
ANN at this N. Deterministic results matter for reproducible builds.

**Why MiniLM-L6-v2?** 384 dims, CPU-fine, well-studied for short-text
semantic search. Bigger encoders (bge-large, e5-large) add latency and
disk cost without changing retrieval quality meaningfully on 501 short
questions.

**Why promote sub-questions to first-class rows?** Paper generation
samples at the sub-question granularity (each has its own marks and
difficulty). A joined schema would force every query to `JOIN
context_groups`. Flat is simpler and faster; the `context_group_id` FK
preserves the relationship when you need it.

**Why wipe-and-rebuild?** The source JSON is the source of truth. There's
no long-lived state to migrate — every build is derivable from the JSON
in ~11s. Avoids a whole category of migration bugs.

**Why fail loud on malformed rows?** Silent skips corrupt the downstream
pipeline without a visible signal. `build_index.py` raises with the
offending id so data issues surface immediately.

**Why no LLM here?** Separation of concerns. This layer gives the
generation pipeline two primitives — "rows matching a spec" and "rows
near a meaning" — and stops. Every other concern (prompt design,
blueprint scoring, drafting, grading) is a separate problem.

---

## When to outgrow this

| Signal                                        | Next step                                           |
|-----------------------------------------------|-----------------------------------------------------|
| Question bank grows to 100K+                  | `IndexHNSWFlat` (still FAISS, still in-process)     |
| Need filter-during-search (not post-filter)   | Qdrant or Weaviate (payload-aware ANN)              |
| Multi-process / multi-tenant                  | pgvector on Postgres, or a dedicated vector DB      |
| Want hybrid sparse+dense                      | Add SQLite FTS5 for BM25, fuse with FAISS via RRF   |
| Schema needs to evolve in place               | Introduce proper migrations (Alembic, etc.)         |

None of these are justified for 501 past-paper questions.

---

## Troubleshooting

**`ImportError: numpy.core.multiarray failed to import` when importing faiss**
faiss-cpu 1.8.0 wheels are built against NumPy 1.x. If a NumPy 2.x is
active, pin it down:

```bash
pip install --user --force-reinstall "numpy==1.26.4"
```

**`FileNotFoundError: data/questions.db`**
You didn't run `python build_index.py`, or you ran it from a different
working directory. The builder writes `data/` next to itself.

**`ValueError: standalone id='...' missing fields: [...]`**
The source JSON has a row that doesn't match the expected shape. Fix the
JSON (the builder refuses to skip it).

**Slow first build**
First run downloads the ~90 MB MiniLM model to `~/.cache/huggingface`.
Subsequent builds take ~11s on CPU.

---

## License / attribution

Questions sourced from La Martiniere For Boys, Kolkata — History & Civics
past papers, 2022-23 through 2024-25. All rights to the question text
belong to the school. This repo builds an index over them; it does not
republish them.
