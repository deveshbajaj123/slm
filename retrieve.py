"""Query helpers over the SQLite + FAISS storage layer. No LLM calls."""
from __future__ import annotations

import json
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent
DEFAULT_DB = ROOT / "data" / "questions.db"
DEFAULT_INDEX = ROOT / "data" / "faiss.index"
DEFAULT_IDS = ROOT / "data" / "faiss_ids.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------- connections & caches ----------

def _connect(db_path: Path | str | None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DEFAULT_DB
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


@lru_cache(maxsize=4)
def load_index(
    index_path: str | None = None,
    ids_path: str | None = None,
) -> tuple[faiss.Index, list[str], dict[str, int]]:
    ip = Path(index_path) if index_path else DEFAULT_INDEX
    idp = Path(ids_path) if ids_path else DEFAULT_IDS
    index = faiss.read_index(str(ip))
    ids: list[str] = json.loads(idp.read_text(encoding="utf-8"))
    id_to_row = {qid: i for i, qid in enumerate(ids)}
    return index, ids, id_to_row


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


# ---------- helpers ----------

def _as_list(v: object) -> list | None:
    if v is None:
        return None
    if isinstance(v, (list, tuple, set)):
        return list(v)
    return [v]


def _in_clause(col: str, values: list, params: list) -> str:
    placeholders = ",".join("?" * len(values))
    params.extend(values)
    return f"{col} IN ({placeholders})"


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {k: row[k] for k in row.keys()}


# ---------- public API ----------

def filter_questions(
    *,
    type: str | list[str] | None = None,
    marks: int | list[int] | None = None,
    class_: str | list[str] | None = None,
    year: str | list[str] | None = None,
    topic: str | list[str] | None = None,
    difficulty: str | list[str] | None = None,
    bloom_level: str | list[str] | None = None,
    kind: str | None = None,
    exclude_ids: set[str] | None = None,
    limit: int | None = None,
    db_path: str | Path | None = None,
) -> list[dict]:
    where: list[str] = []
    params: list = []

    for col, val in [
        ("type", type), ("marks", marks), ("class", class_), ("year", year),
        ("topic", topic), ("difficulty", difficulty), ("bloom_level", bloom_level),
    ]:
        lv = _as_list(val)
        if lv is not None:
            where.append(_in_clause(col, lv, params))

    if kind is not None:
        where.append("kind = ?")
        params.append(kind)

    if exclude_ids:
        where.append(f"id NOT IN ({','.join('?' * len(exclude_ids))})")
        params.extend(exclude_ids)

    sql = "SELECT * FROM questions"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)

    with _connect(db_path) as conn:
        return [_row_to_dict(r) for r in conn.execute(sql, params)]


def get_question(qid: str, *, db_path: str | Path | None = None) -> dict | None:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM questions WHERE id = ?", (qid,)).fetchone()
    return _row_to_dict(row) if row else None


def get_context_group(
    cg_id: str,
    *,
    include_questions: bool = True,
    db_path: str | Path | None = None,
) -> dict | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM context_groups WHERE id = ?", (cg_id,)
        ).fetchone()
        if row is None:
            return None
        out = _row_to_dict(row)
        if include_questions:
            subs = conn.execute(
                "SELECT * FROM questions WHERE context_group_id = ? ORDER BY sub_label",
                (cg_id,),
            ).fetchall()
            out["questions"] = [_row_to_dict(s) for s in subs]
    return out


def filter_context_groups(
    *,
    class_: str | list[str] | None = None,
    year: str | list[str] | None = None,
    topic: str | list[str] | None = None,
    total_marks: int | tuple[int, int] | None = None,
    exclude_ids: set[str] | None = None,
    db_path: str | Path | None = None,
) -> list[dict]:
    where: list[str] = []
    params: list = []
    for col, val in [("class", class_), ("year", year), ("topic", topic)]:
        lv = _as_list(val)
        if lv is not None:
            where.append(_in_clause(col, lv, params))
    if total_marks is not None:
        if isinstance(total_marks, tuple):
            lo, hi = total_marks
            where.append("total_marks BETWEEN ? AND ?")
            params.extend([lo, hi])
        else:
            where.append("total_marks = ?")
            params.append(int(total_marks))
    if exclude_ids:
        where.append(f"id NOT IN ({','.join('?' * len(exclude_ids))})")
        params.extend(exclude_ids)

    sql = "SELECT * FROM context_groups"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id"
    with _connect(db_path) as conn:
        return [_row_to_dict(r) for r in conn.execute(sql, params)]


def _embed_text_input(topic: str | None, subtopic: str | None, qtext: str) -> str:
    parts: list[str] = []
    if topic:
        parts.append(topic)
    tail = qtext
    if subtopic:
        tail = f"{subtopic}: {qtext}"
    parts.append(tail)
    return " \u2014 ".join(parts)


def similar_to(
    qid_or_text: str,
    *,
    k: int = 10,
    exclude_ids: set[str] | None = None,
    min_score: float | None = None,
    db_path: str | Path | None = None,
    index_path: str | None = None,
    ids_path: str | None = None,
) -> list[tuple[str, float]]:
    index, ids, id_to_row = load_index(index_path, ids_path)
    exclude = set(exclude_ids) if exclude_ids else set()

    row_idx = id_to_row.get(qid_or_text)
    if row_idx is not None:
        vec = index.reconstruct(row_idx).reshape(1, -1)
        exclude.add(qid_or_text)
    else:
        # treat as raw text; try to enrich with topic if it is a known qid row but
        # not in the embed index (shouldn't happen, but cheap to keep uniform)
        row = get_question(qid_or_text, db_path=db_path)
        if row is not None:
            text = _embed_text_input(row["topic"], row["subtopic"], row["question_text"])
            exclude.add(qid_or_text)
        else:
            text = qid_or_text
        vec = _model().encode(
            [text], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

    n = index.ntotal
    # overfetch so exclusions/min_score don't starve k
    want = min(n, k + len(exclude) + 1)
    scores, idxs = index.search(vec, want)

    results: list[tuple[str, float]] = []
    for score, i in zip(scores[0].tolist(), idxs[0].tolist()):
        if i < 0:
            continue
        qid = ids[i]
        if qid in exclude:
            continue
        if min_score is not None and score < min_score:
            continue
        results.append((qid, float(score)))
        if len(results) >= k:
            break
    return results


def find_duplicates(
    threshold: float = 0.9,
    *,
    index_path: str | None = None,
    ids_path: str | None = None,
) -> list[tuple[str, str, float]]:
    index, ids, _ = load_index(index_path, ids_path)
    n = index.ntotal
    if n == 0:
        return []
    all_vecs = np.vstack([index.reconstruct(i) for i in range(n)]).astype(np.float32)
    sims = all_vecs @ all_vecs.T  # cosine (normalized)
    pairs: list[tuple[str, str, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sims[i, j])
            if s >= threshold:
                a, b = ids[i], ids[j]
                if a > b:
                    a, b = b, a
                pairs.append((a, b, s))
    pairs.sort(key=lambda p: -p[2])
    return pairs
