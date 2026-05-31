"""Recall@k: dense-only (similar_to) vs hybrid_search, over a hand-labelled
seed set. The point is to VERIFY hybrid actually helps on THIS bank, not to
assume it. Run after build_index.py.

    python eval_hybrid.py
    python eval_hybrid.py --k 5 10 20

Caveat (N=501): BM25 IDF stats are thin/noisy on a small corpus, so expect
hybrid to be erratic on borderline queries. That is a property of the data,
not a bug. Read per-seed recall, not just the mean.
"""
from __future__ import annotations

import argparse

import retrieve as r

# --- Hand-labelled seed set -------------------------------------------------
# {seed_qid: [relevant_qids]} — for each seed, the qids a human judges as
# genuinely relevant neighbours (same concept / near-duplicate / same entity).
# Recall@k = (relevant ids found in top-k) / (total relevant ids) for the seed.
#
# Labelled below by mining the bank for questions sharing rare entity tokens
# ("Panchayati Raj", "Presidential Election", "Election Commissioner", "Speaker"
# ...) plus the guaranteed near-duplicate pairs find_duplicates(0.97) surfaces.
# These are exactly the cases where exact-term BM25 should match — or hold its
# own against — dense embeddings. Keep relevant lists TIGHT (only truly-relevant
# ids), or recall numbers lose meaning. Every id is verified at runtime (a bad
# id prints a WARNING and silently caps recall < 1.0).
#
# TODO(krish): extend toward 30 seeds; History topics (Industrial Revolution,
# WWI, Nationalism) are under-represented — the bank skews Civics.
SEED_SET: dict[str, list[str]] = {
    # --- entity-heavy clusters (BM25's wheelhouse) ---------------------------
    # Elections / Election Commission (exact terms "election", "Commission").
    "MCQ-001": ["MCQ-058", "MCQ-077", "CG-051_Q1"],
    # "Election Commissioner" — appointment & term.
    "CG-051_Q3": ["MCQ-080"],
    # Presidential Election — "elected by an intermediary body".
    "MCQ-002": ["MCQ-078", "MCQ-041"],
    # Zila Parishad / "apex body under the Panchayati Raj Institution".
    "MCQ-006": ["MCQ-082", "CG-002_Q3", "CG-032_Q3"],
    # Panchayat Samiti — intermediate-level rural LSG.
    "MCQ-060": ["CG-052_Q3"],
    # Gram Sabha / Gram Panchayat — elects executive wing for 5-year tenure.
    "MCQ-081": ["MCQ-084"],
    # Speaker of the Lok Sabha — powers / presiding officer.
    "SA-095": ["SA-106", "CG-023_Q3"],

    # --- guaranteed near-duplicate pairs (find_duplicates >= 0.97) -----------
    # Both dense and sparse should recover these; they anchor the eval floor.
    "DEF-002": ["DEF-007"],
    "MCQ-014": ["MCQ-090"],
    "ID-014": ["ID-024"],
    "ID-015": ["ID-025"],
    "DEF-003": ["DEF-011"],
    "CG-021_Q1": ["CG-029_Q1"],
    "CG-021_Q2": ["CG-029_Q2"],
    "CG-041_Q2": ["CG-045_Q2"],
    "SA-078": ["SA-120"],
    "SA-051": ["SA-074"],
    "SA-090": ["SA-102"],
    "CG-040_Q3": ["CG-044_Q3"],
    "MCQ-025": ["MCQ-045"],
    "CG-005_Q2": ["CG-036_Q2"],
}


# --- Raw-text queries -------------------------------------------------------
# Keys here are RAW STRINGS, not stored qids: this exercises the on-the-fly
# embed/tokenize path (the realistic "user types a question" case, vs seeding
# from an existing row). Terse, entity-named queries are where BM25 should pull
# ahead of a short dense embedding. In practice on this bank MiniLM handles even
# single entity tokens well, so most tie — but a few (e.g. "Gram Sabha tenure
# five years", which recovers MCQ-084 via exact "gram"/"sabha" tokens dense
# dropped from the top-5) show hybrid's lexical rescue. Honest mix of wins+ties.
RAW_QUERIES: dict[str, list[str]] = {
    "Gram Sabha tenure five years": ["MCQ-081", "MCQ-084"],
    "Election Commissioner term of office": ["MCQ-080", "CG-051_Q3"],
    "apex body Panchayati Raj Zila Parishad": ["MCQ-006", "MCQ-082", "CG-002_Q3", "CG-032_Q3"],
    "Presidential Election intermediary body": ["MCQ-002", "MCQ-078", "MCQ-041"],
    "Speaker presiding officer Lok Sabha": ["SA-106", "SA-095", "CG-023_Q3"],
    "Municipal Commissioner appointment term": ["SA-035"],
}


def recall_at_k(found: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    topk = set(found[:k])
    return len(topk & relevant) / len(relevant)


def eval_seed_set(
    seed_set: dict[str, list[str]],
    ks: list[int],
    *,
    keys_are_ids: bool = True,
    label: str = "seed",
) -> None:
    maxk = max(ks)
    # Validate relevant ids up front so a typo doesn't silently depress recall.
    # When keys are raw queries (keys_are_ids=False) the key is text, not a qid,
    # so only the relevant lists are id-checked.
    bad = [qid for qid in seed_set if keys_are_ids and r.get_question(qid) is None]
    for seed, rels in seed_set.items():
        bad += [x for x in rels if r.get_question(x) is None]
    if bad:
        print(f"WARNING: {len(bad)} unknown qid(s) in {label} set: {sorted(set(bad))}\n")

    # accumulate recall per method per k
    dense_tot = {k: 0.0 for k in ks}
    hybrid_tot = {k: 0.0 for k in ks}
    n = 0

    for seed, rel_list in seed_set.items():
        relevant = set(rel_list)
        if not relevant:
            continue
        n += 1
        dense = [qid for qid, _ in r.similar_to(seed, k=maxk)]
        hybrid = [qid for qid, _ in r.hybrid_search(seed, k=maxk)]

        shown = seed if keys_are_ids else f'"{seed}"'
        print(f"{label} {shown}  (|relevant|={len(relevant)})")
        for k in ks:
            dr = recall_at_k(dense, relevant, k)
            hr = recall_at_k(hybrid, relevant, k)
            dense_tot[k] += dr
            hybrid_tot[k] += hr
            flag = "  <-- hybrid wins" if hr > dr else ("  <-- dense wins" if dr > hr else "")
            print(f"  @{k:<3} dense={dr:.3f}  hybrid={hr:.3f}{flag}")
        print()

    if n == 0:
        print(f"No labelled {label}s. Fill in the set.")
        return

    print(f"==== mean recall@k over {n} {label}(s) ====")
    for k in ks:
        d = dense_tot[k] / n
        h = hybrid_tot[k] / n
        delta = h - d
        print(f"  @{k:<3} dense={d:.3f}  hybrid={h:.3f}  delta={delta:+.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    args = ap.parse_args()
    ks = sorted(args.k)

    print("################ stored-id seeds (reuse stored vector + text) ################\n")
    eval_seed_set(SEED_SET, ks)

    print("\n################ raw-text queries (embed/tokenize on the fly) ################\n")
    eval_seed_set(RAW_QUERIES, ks, keys_are_ids=False, label="query")


if __name__ == "__main__":
    main()
