"""One-shot builder: JSON -> SQLite + FAISS.

Wipes `data/` and rebuilds. Fails loud on malformed rows.

    python build_index.py
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent
SOURCE_JSON = ROOT / "lmb_history_civics_question_bank_enriched.json"
SCHEMA_SQL = ROOT / "schema.sql"
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "questions.db"
INDEX_PATH = DATA_DIR / "faiss.index"
IDS_PATH = DATA_DIR / "faiss_ids.json"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

STANDALONE_REQUIRED = {
    "id", "type", "marks", "question", "source",
    "topic", "subtopic", "difficulty", "bloom_level",
}
CG_REQUIRED = {
    "id", "context_type", "context_description", "topic",
    "total_marks", "source", "questions",
}
SUB_REQUIRED = {"id", "sub_label", "marks", "question", "difficulty", "bloom_level"}
SOURCE_REQUIRED = {"school", "class", "year", "exam_type"}


def _norm(text: str) -> str:
    t = text.lower()
    t = re.sub(r"\s+", " ", t).strip()
    t = t.rstrip(" .?!:;,-_")
    return t


def text_hash(text: str) -> str:
    return hashlib.sha1(_norm(text).encode("utf-8")).hexdigest()


def _check(row: dict, required: set[str], label: str) -> None:
    missing = required - row.keys()
    if missing:
        raise ValueError(f"{label} id={row.get('id')!r} missing fields: {sorted(missing)}")


def embed_text(topic: str | None, subtopic: str | None, qtext: str) -> str:
    parts: list[str] = []
    if topic:
        parts.append(topic)
    tail = qtext
    if subtopic:
        tail = f"{subtopic}: {qtext}"
    parts.append(tail)
    return " \u2014 ".join(parts)


def main() -> None:
    t0 = time.time()
    if not SOURCE_JSON.exists():
        raise FileNotFoundError(SOURCE_JSON)
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir(parents=True)

    with SOURCE_JSON.open(encoding="utf-8") as f:
        data = json.load(f)

    standalone = data["standalone_questions"]
    groups = data["context_groups"]

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    with SCHEMA_SQL.open(encoding="utf-8") as f:
        conn.executescript(f.read())

    group_rows: list[tuple] = []
    q_rows: list[tuple] = []
    embed_inputs: list[tuple[str, str]] = []  # (qid, embed_text)

    for q in standalone:
        _check(q, STANDALONE_REQUIRED, "standalone")
        _check(q["source"], SOURCE_REQUIRED, f"standalone[{q['id']}].source")
        src = q["source"]
        qtext = q["question"]
        q_rows.append((
            q["id"], "standalone", q["type"], int(q["marks"]), qtext,
            q["topic"], q["subtopic"], q["difficulty"], q["bloom_level"],
            src["class"], src["year"], src["exam_type"], src["school"],
            None, None, text_hash(qtext),
        ))
        embed_inputs.append((q["id"], embed_text(q["topic"], q["subtopic"], qtext)))

    for g in groups:
        _check(g, CG_REQUIRED, "context_group")
        _check(g["source"], SOURCE_REQUIRED, f"context_group[{g['id']}].source")
        src = g["source"]
        group_rows.append((
            g["id"], g["context_type"], g["context_description"], g["topic"],
            int(g["total_marks"]), src["class"], src["year"], src["exam_type"], src["school"],
        ))
        for sub in g["questions"]:
            _check(sub, SUB_REQUIRED, f"context_group[{g['id']}].questions")
            qtext = sub["question"]
            q_rows.append((
                sub["id"], "context_sub", None, int(sub["marks"]), qtext,
                g["topic"], None, sub["difficulty"], sub["bloom_level"],
                src["class"], src["year"], src["exam_type"], src["school"],
                g["id"], sub["sub_label"], text_hash(qtext),
            ))
            embed_inputs.append((sub["id"], embed_text(g["topic"], None, qtext)))

    conn.executemany(
        "INSERT INTO context_groups VALUES (?,?,?,?,?,?,?,?,?)",
        group_rows,
    )
    conn.executemany(
        "INSERT INTO questions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        q_rows,
    )
    conn.commit()

    # --- BM25 sparse index (FTS5), built in the same wipe-and-rebuild pass ---
    # Standalone (non-external-content) FTS5: the PK is a TEXT id, so external-
    # content mode (which keys on an int rowid) does not fit. search_text is
    # populated with the SAME composed string that gets embedded (see
    # embed_inputs), so the dense and sparse sides index identical text and RRF
    # fusion stays symmetric. No porter stemmer: keep entity tokens exact.
    conn.execute(
        "CREATE VIRTUAL TABLE questions_fts USING fts5("
        "qid UNINDEXED, search_text, "
        "tokenize='unicode61 remove_diacritics 2')"
    )
    conn.executemany(
        "INSERT INTO questions_fts (qid, search_text) VALUES (?, ?)",
        embed_inputs,
    )
    conn.commit()

    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    texts = [t for _, t in embed_inputs]
    ids = [qid for qid, _ in embed_inputs]
    print(f"Embedding {len(texts)} question texts...")
    vectors = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    dim = int(vectors.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    faiss.write_index(index, str(INDEX_PATH))
    with IDS_PATH.open("w", encoding="utf-8") as f:
        json.dump(ids, f)

    # summary
    cg_count = conn.execute("SELECT COUNT(*) FROM context_groups").fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
    sa_count = conn.execute(
        "SELECT COUNT(*) FROM questions WHERE kind='standalone'"
    ).fetchone()[0]
    sub_count = conn.execute(
        "SELECT COUNT(*) FROM questions WHERE kind='context_sub'"
    ).fetchone()[0]
    fts_count = conn.execute("SELECT COUNT(*) FROM questions_fts").fetchone()[0]
    conn.close()

    elapsed = time.time() - t0
    print("---- build summary ----")
    print(f"context_groups rows : {cg_count}")
    print(f"questions rows      : {total} (standalone={sa_count}, context_sub={sub_count})")
    print(f"faiss vectors       : {index.ntotal} (dim={dim})")
    print(f"fts5 rows           : {fts_count}")
    print(f"db                  : {DB_PATH}")
    print(f"index               : {INDEX_PATH}")
    print(f"build time          : {elapsed:.2f}s")


if __name__ == "__main__":
    main()
