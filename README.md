# LMB History & Civics — RAG Retrieval Layer

A storage-and-retrieval layer over a bank of **501 La Martiniere History & Civics
exam questions**. It turns one source JSON file into two parallel indexes — a
**SQLite** database and a **FAISS** vector index — and provides query helpers
for both exact filtering and meaning-based similarity, fused via hybrid search.

> **Scope is deliberately narrow.** This repo builds and queries the indexes. It
> does **not** generate exam papers, call any LLM, or expose a web API.

This README explains the code and the retrieval process in full, defining every
technical term as it comes up. Run order: `pip install -r requirements.txt`,
then `python build_index.py`, then query via `retrieve.py` or run
`python eval_hybrid.py`.

---

## 1. The big picture

You have **501 exam questions** in a JSON file. The system turns them into two
parallel indexes so downstream code can ask two different kinds of question:

- **"Give me all 1-mark MCQs for Class X"** → a *structured filter* (a plain
  database query).
- **"Find questions that mean the same thing as this one"** → a *similarity
  search* (needs math on meaning).

The project builds those indexes and provides helper functions to query them. It
does **not** generate papers or call any LLM — it is purely the
storage-and-retrieval layer.

Two storage technologies are used together:

| Store | What it holds | Good at |
|-------|---------------|---------|
| **SQLite** (`questions.db`) | All readable metadata (text, marks, topic, class, year…) + the BM25 full-text index | Exact filtering: "where marks = 1 and class = X" |
| **FAISS** (`faiss.index`) | Only numbers — one *vector* per question | Meaning-based similarity |

A small JSON file (`faiss_ids.json`) is the bridge between them.

```
JSON ──build_index.py──┬─► SQLite: metadata + questions_fts (BM25 / sparse)
                       └─► FAISS:  384-dim vectors (dense) + faiss_ids.json bridge

query ──hybrid_search──┬─► dense:  embed → FAISS → top-50 by cosine ──┐
                       └─► sparse: tokenize → FTS5 → top-50 by BM25 ──┤
                                                                       ▼
                                                    RRF fuse on rank (kk=60)
                                                                       ▼
                                          exclude + deterministic sort → top-k
```

---

## 2. Core concepts (the vocabulary)

**Embedding / vector.** A neural network reads a question's text and outputs a
list of 384 numbers, e.g. `[0.02, -0.15, 0.33, …]`. That list is the
**embedding** (a.k.a. **vector**). The key property: questions with *similar
meaning* get *similar number-lists*, even if they share no words. The model used
is **MiniLM-L6-v2**, a small, fast sentence-embedding model — "384-dim" means
each vector has 384 components; "L6" means 6 transformer layers.

**Dimension (dim).** The length of the vector — 384 numbers. Picture each vector
as a point in 384-dimensional space (impossible to visualize, but the math works
the same as 2D or 3D).

**Cosine similarity.** The standard measure of how "close in meaning" two vectors
are. It is the cosine of the angle between them: **1.0** = same direction
(identical meaning), **0** = perpendicular (unrelated), **−1** = opposite. The
code makes this cheap via **L2-normalization**: every vector is scaled to length
1. Once all vectors have length 1, the **dot product** (multiply components
pairwise, then sum — also called **inner product**) equals the cosine. So
similarity becomes a single fast multiplication.

**FAISS** (Facebook AI Similarity Search). A C++ library for storing many
vectors and finding the nearest ones to a query, fast. The index type here is
**`IndexFlatIP`**:

- *Flat* = brute force: it compares the query against all 501 vectors, no
  approximation. At this scale it is sub-millisecond and exact every time
  (important for reproducibility).
- *IP* = Inner Product, which (because vectors are normalized) equals cosine.
- The alternative, **ANN** (Approximate Nearest Neighbor — index types like
  HNSW/IVF), trades exactness for speed and is only worth it at hundreds of
  thousands of vectors. Overkill here.

**BM25.** A classic **lexical** (word-matching) ranking formula from search
engines. It scores a document by how often the query's words appear, down-weights
common words ("the", "of") and up-weights rare ones. The intuition: rare exact
terms like *Lok Sabha*, *Article 356*, *Election Commissioner* are exactly what
embeddings tend to "smear" (blur with related concepts), whereas BM25 matches
them literally. **IDF** (Inverse Document Frequency) is BM25's "rarer word = more
important" component.

**FTS5** (Full-Text Search version 5). A search engine built *into* SQLite — no
extra library to install. It tokenizes text, builds an inverted index, and
provides the `bm25()` ranking function. This is why "no new dependency" holds:
FTS5 ships with SQLite, which ships with Python.

**Tokenize.** Split text into searchable units ("words"). The config
`unicode61 remove_diacritics 2` means: Unicode-aware word splitting, with accent
marks stripped (so "café" matches "cafe"). Deliberately **no porter stemmer** — a
stemmer would chop "elections" → "elect", blurring distinct entity terms. Keeping
tokens exact is the whole point.

**RRF** (Reciprocal Rank Fusion). The method for *combining* the two rankings
(dense + sparse) into one. See [section 5](#5-the-hybrid-retrieval-process--hybrid_search).

---

## 3. The build process — `build_index.py`

A one-shot script: run it and it wipes `data/` and rebuilds everything from the
JSON. "Wipe-and-rebuild" means there is no incremental update or migration logic
— the JSON is the single source of truth and the whole index regenerates in
~10s.

1. **Wipe and reload.** Delete `data/`, recreate it. Load the source JSON. If a
   row is missing an expected field, `raise` with that row's id — never silently
   skip bad data ("fail loud").

2. **Build the SQL rows.** The JSON has two shapes: *standalone questions*
   (self-contained, e.g. `MCQ-001`) and *context groups* (a shared stimulus — a
   map, quote, passage — with sub-questions, e.g. `CG-001_Q1`). The builder
   **flattens** sub-questions into first-class rows in the `questions` table
   (`kind='context_sub'`, with a foreign key back to the group), so every
   question is queryable uniformly without a JOIN.

3. **The composed embedding string.** The crucial bit. For each question the
   builder constructs one string — `embed_text()` in
   [build_index.py](build_index.py):

   ```
   "{topic} — {subtopic}: {question_text}"
   ```

   Missing parts drop cleanly (no literal `"None"` ever appears). Each pair
   `(qid, composed_string)` is collected into the list `embed_inputs`. **This
   list is reused for both dense and sparse**, guaranteeing both sides index
   *identical text*.

4. **Insert into SQLite.** `executemany` runs one parameterized `INSERT` per row
   in a batch. "Parameterized" = values passed via `?` placeholders, not
   string-concatenated — prevents SQL injection and quoting bugs.

5. **Build the FTS5 (BM25 / sparse) table:**

   ```sql
   CREATE VIRTUAL TABLE questions_fts USING fts5(
       qid UNINDEXED, search_text,
       tokenize='unicode61 remove_diacritics 2');
   ```

   - `qid UNINDEXED` — store the id but don't full-text-search it.
   - `search_text` — the searchable column, populated from the **same
     `embed_inputs` list** via `executemany`.
   - It is a **standalone** (non-external-content) FTS5 table: it keeps its own
     copy of the text. The "external-content" alternative keys off an integer
     rowid pointing into another table, but here the primary key is a *text* id
     (`MCQ-001`), so standalone is the natural fit.

6. **Build the dense (FAISS) index.** Load MiniLM, encode all 501 composed
   strings in one batch into a `(501, 384)` float32 array with
   `normalize_embeddings=True` (the L2-normalization trick). Then
   `IndexFlatIP(384)`, `.add(vectors)`, write to disk. Write `faiss_ids.json` —
   the list of qids **in the exact order vectors were added**, so FAISS row 0 ↔
   `ids[0]` ↔ `MCQ-001`. This list is the only bridge from a FAISS row number
   back to a question id.

7. **Print a summary**, including `fts5 rows` and `faiss vectors` — both must
   equal 501. That equality is the integrity check that dense and sparse cover
   the same corpus.

---

## 4. The query helpers — `retrieve.py`

**Caching helpers.** `load_index()` reads the FAISS file plus the id list and
builds `id_to_row` (a dict mapping `qid → row number`, the reverse of
`faiss_ids.json`). It is wrapped in `@lru_cache` — **memoization**, so the result
is computed once and reused across queries. `_model()` similarly loads MiniLM
once.

**`filter_questions(...)`** — pure SQL. Builds a `WHERE` clause from whatever
filters you pass (each may be a scalar or a list → `IN (...)`). No vectors. This
is the "structured filter" primitive.

**`similar_to(qid_or_text, k)`** — the **dense** search:

1. If the argument is a known qid, look up its row and call
   `index.reconstruct(row)` — fetch the *already-stored* vector, no re-encoding.
   If it is raw text, encode it on the fly with MiniLM.
2. `index.search(vec, want)` — FAISS returns the `want` nearest rows and their
   cosine scores.
3. **Overfetch:** ask for more than `k` (`k + exclusions + 1`) so that after
   dropping the query itself and any `exclude_ids`, a full `k` results remain.
   "Overfetch" = retrieve extra candidates to survive later filtering.
4. Returns `[(qid, cosine_score)]`.

**`bm25_search(text, k)`** — the **sparse** search:

```sql
SELECT qid FROM questions_fts WHERE questions_fts MATCH ?
ORDER BY bm25(questions_fts) ASC LIMIT ?;
```

- `MATCH` is FTS5's search operator.
- SQLite's `bm25()` returns **negative** numbers (more negative = more relevant),
  so `ORDER BY … ASC` puts the best matches first.
- Raw text must be **sanitized** into a safe MATCH expression first
  (`_sanitize_fts_query()`), because raw question text contains characters FTS5
  treats as operators (quotes, `*`, `:`, the words AND/OR/NEAR). The sanitizer:
  - `re.findall(r"\w+", text.lower())` — extract word-character tokens, lowercase.
  - Wrap each token in double quotes (so each is a literal, not an operator).
  - Join with `OR`, **not** `AND`. `AND` would require *every* word present,
    destroying recall on long questions. `OR` means "match any of these terms"
    and lets BM25 rank by how many / how rare the matches are.
  - Empty input → `'""'`, a query that matches nothing.
- Returns a plain list of qids, best first.

---

## 5. The hybrid retrieval process — `hybrid_search`

The heart of the system. It runs *both* searches and fuses them.

**Step 1 — resolve the query for both sides symmetrically.**

- If the argument is a stored qid: the dense side reconstructs its stored vector,
  and the sparse side reuses the **same composed string** that was embedded
  (rebuilt via `_embed_text_input`). Both sides see identical input.
- If it is raw text: both sides take the raw text (dense embeds it, sparse
  tokenizes it).

**Step 2 — get two ranked lists**, each at `overfetch=50` candidates:
`dense_ids` from `similar_to`, `sparse_ids` from `bm25_search`.

**Step 3 — Reciprocal Rank Fusion (RRF).** The critical design choice: **fuse on
rank only, ignore the actual scores.** Why? The two scores live on incompatible
scales — cosine is `0..1`, BM25 is some negative number — and normalizing across
them is fragile. Rank (1st, 2nd, 3rd…) is comparable across both. The formula:

```
score(q) = Σ over each list of   1 / (kk + rank)        # rank is 1-based
```

- `rank` is the 1-based position of question `q` in a list.
- `kk = 60` is a **damping constant** (the standard value from Cormack et al.,
  2009). It softens the gap between top ranks — without it, rank 1 would utterly
  dominate rank 2. With `kk=60`, rank 1 contributes `1/61`, rank 2 `1/62`, etc. —
  close together, so a question that ranks *decently in both* lists can beat one
  that ranks 1st in only one. That is the point of fusion: reward agreement
  across methods.
- A question appearing in both lists gets *both* contributions added → it floats
  to the top. This is how BM25's exact-term hit rescues something dense ranked
  low.

**Step 4 — exclude and sort.** Drop the input id and any `exclude_ids`, then:

```python
fused.sort(key=lambda p: (-p[1], p[0]))   # score desc, then qid asc
```

Breaking ties by qid makes the output **deterministic** — identical every run,
which matters for reproducible builds and tests (at N=501 the RRF scores cluster
tightly and ties are common). Returns the top `k` as `[(qid, rrf_score)]`.

**Concrete example** — the raw query `"Gram Sabha tenure five years"`:

- Dense ranks `MCQ-084` *below* position 5 (a short query gives MiniLM little to
  work with, and it blurs "Gram Sabha" with general local-government concepts).
- BM25 ranks `MCQ-084` high because it literally contains the rare tokens "gram"
  and "sabha".
- RRF adds both contributions → `MCQ-084` jumps into the top 5. Dense alone
  missed it; hybrid caught it. That is the +0.08 recall@5 win the eval measures.

---

## 6. The evaluation — `eval_hybrid.py`

**Recall@k** is the metric. For a "seed" question with a known set of
truly-relevant neighbours:

```
recall@k = (relevant items found in the top k) / (total relevant items)
```

Recall@5 = 1.0 means all the relevant ones made it into the top 5. (Recall
measures "did we find them"; it does not penalize extra junk — that would be
*precision*.)

The script compares **dense-only** (`similar_to`) vs **hybrid**
(`hybrid_search`) on two hand-labelled sets:

- **Stored-id seeds** — seed from an existing question (reuses stored vector +
  text).
- **Raw-text queries** — typed-out strings (exercises the on-the-fly
  embed/tokenize path, the realistic "user types a question" case).

**The honest finding:** on this 501-question bank, dense embeddings are already
so strong that hybrid is **break-even on id-seeded reuse** (occasionally slightly
worse at @5) and a **small win (+0.08 recall@5) on short raw-text entity
queries**. The "BM25 IDF is noisy at small N" caveat is expected, not a bug —
with only 501 documents, the rare-word statistics BM25 relies on are thin.

```
==== mean recall@k, stored-id seeds (21) ====
  @5   dense=0.976  hybrid=0.952  delta=-0.024
  @10  dense=1.000  hybrid=1.000  delta=+0.000

==== mean recall@k, raw-text queries (6) ====
  @5   dense=0.861  hybrid=0.944  delta=+0.083
  @10  dense=1.000  hybrid=1.000  delta=+0.000
```

---

## Summary

Dense answers "what *means* the same"; sparse answers "what *says* the same
word"; RRF lets a result win if *either* method is confident, and especially if
*both* are. The retrieval layer is intentionally complete at these two primitives
— exact filtering and similarity — and stops there; paper generation, sampling,
and drafting live outside this repo.

---

## Repo layout

```
.
├── lmb_history_civics_question_bank_enriched.json   # canonical source (never modified)
├── build_index.py     # one-shot: JSON → SQLite + FAISS + FTS5
├── retrieve.py        # query helpers (filter / similar_to / bm25_search / hybrid_search)
├── eval_hybrid.py     # recall@k: dense-only vs hybrid
├── schema.sql         # readable copy of the SQLite schema
├── requirements.txt   # pinned deps (faiss-cpu, sentence-transformers, numpy)
└── data/              # generated; wiped & rebuilt every run — NOT committed
    ├── questions.db   #   SQLite: questions, context_groups, questions_fts
    ├── faiss.index    #   501 vectors × 384 dims
    └── faiss_ids.json #   row-index → qid bridge
```

`data/` is a build artifact (gitignored). Rebuild it any time with
`python build_index.py`.

## License / attribution

Questions sourced from La Martiniere For Boys, Kolkata — History & Civics past
papers. All rights to the question text belong to the school. This repo builds an
index over them; it does not republish them.
</content>
</invoke>
