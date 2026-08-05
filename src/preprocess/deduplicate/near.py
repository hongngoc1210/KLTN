from datasketch import MinHash, MinHashLSH
from tqdm import tqdm
import re

from .common import *


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

    text = re.sub(r"\s+", " ", text)

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

    m = MinHash(num_perm=NUM_PERM)

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

        doc_id = doc["id"]

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
                    "id": doc_id,
                    "dataset": doc["_dataset"],
                    "duplicate_of": cid,
                    "similarity": round(jac, 4),
                    "method": "MinHash-LSH"
                })

                break

        if duplicated:
            continue

        kept.append(doc)

        hashes[doc_id] = mh

        lsh.insert(
            doc_id,
            mh
        )

    outputs = split_documents(
        kept
    )

    save_documents(outputs, mode="near")

    save_log(
        removed,
        "near_duplicate_removed.json",
        mode="near"
    )

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