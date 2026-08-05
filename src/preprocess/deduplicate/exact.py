import hashlib
from tqdm import tqdm

from .common import *


# ==========================================================
# HASH
# ==========================================================

def compute_hash(text: str) -> str:
    """
    Compute SHA256 hash on normalized text.
    """

    return hashlib.sha256(
        normalize(text).encode("utf8")
    ).hexdigest()


# ==========================================================
# EXACT DEDUPLICATION
# ==========================================================

def run():

    documents = load_all_documents()

    before = len(documents)

    seen = {}

    kept = []

    removed = []

    for doc in tqdm(
        documents,
        desc="Exact Dedup"
    ):

        h = compute_hash(
            doc["content"]
        )

        if h in seen:

            removed.append({
                "id": doc["id"],
                "dataset": doc["_dataset"],
                "duplicate_of": seen[h]["id"],
                "duplicate_dataset": seen[h]["_dataset"],
                "method": "sha256"
            })

            continue

        seen[h] = doc

        kept.append(doc)

    outputs = split_documents(
        kept
    )

    save_documents(outputs, mode="exact")

    save_log(
        removed,
        "exact_duplicate_removed.json",
        mode="exact"
    )

    print_summary(
        "Exact Duplicate Removal",
        before,
        len(kept)
    )

    return kept


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    run()