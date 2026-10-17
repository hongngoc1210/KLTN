from datasketch import MinHash, MinHashLSH
from tqdm import tqdm
import re

from common import (
    normalize,
    load_all_documents,
    save_documents,
    save_log,
    split_documents,
    print_summary
)


# ==========================================================
# CONFIG
# ==========================================================

NUM_PERM = 128
THRESHOLD = 0.90
NGRAM = 5


# ==========================================================
# SHINGLES
# ==========================================================

def shingles(text, k=NGRAM):

    text = normalize(text)

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    if len(text) <= k:
        return {text}

    return {
        text[i:i + k]
        for i in range(len(text) - k + 1)
    }


# ==========================================================
# MINHASH
# ==========================================================

def build_minhash(text):

    m = MinHash(
        num_perm=NUM_PERM
    )

    for s in shingles(text):

        m.update(
            s.encode("utf8")
        )

    return m


# ==========================================================
# NEAR DUPLICATE
# ==========================================================

def run():

    documents = load_all_documents()

    before = len(documents)

    if before == 0:

        print("No documents found.")

        return []

    lsh = MinHashLSH(
        threshold=THRESHOLD,
        num_perm=NUM_PERM
    )

    kept = []

    removed = []

    hashes = {}

    for doc in tqdm(
        documents,
        desc="Near Dedup"
    ):

        # ==================================================
        # NEW CHUNK FORMAT
        # ==================================================

        chunk_id = doc["chunk_id"]

        mh = build_minhash(
            doc["content"]
        )

        candidates = lsh.query(mh)

        duplicated = False

        for cid in candidates:

            jac = mh.jaccard(
                hashes[cid]
            )

            if jac >= THRESHOLD:

                duplicated = True

                removed.append({
                    "chunk_id": chunk_id,
                    "doc_id": doc.get("doc_id"),
                    "dataset": doc["_dataset"],
                    "duplicate_of": cid,
                    "similarity": round(
                        jac,
                        4
                    ),
                    "method": "MinHash-LSH"
                })

                break

        if duplicated:
            continue

        # ==================================================
        # KEEP
        # ==================================================

        kept.append(doc)

        hashes[chunk_id] = mh

        lsh.insert(
            chunk_id,
            mh
        )

    # ======================================================
    # SPLIT BY DATASET
    # ======================================================

    outputs = split_documents(
        kept
    )

    # ======================================================
    # SAVE
    # ======================================================

    save_documents(
        outputs,
        mode="near"
    )

    save_log(
        removed,
        "near_duplicate_removed.json",
        mode="near"
    )

    # ======================================================
    # SUMMARY
    # ======================================================

    print_summary(
        "Near Duplicate Removal",
        before,
        len(kept)
    )

    return kept


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    run()