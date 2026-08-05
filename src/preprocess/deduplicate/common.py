import json
import os
import re
from collections import defaultdict


# ==========================================================
# CONFIG
# ==========================================================

INPUT_DIR = "./data/process_data"
OUTPUT_DIR = "./data/deduplicate"

DATASETS = [
    "education",
    "finance",
    "legal",
    "medical",
    "news",
    "science",
]


# ==========================================================
# TEXT NORMALIZATION
# ==========================================================

def normalize(text: str):

    if text is None:
        return ""

    text = text.lower()

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==========================================================
# LOAD
# ==========================================================

def load_all_documents():

    all_docs = []

    for dataset in DATASETS:

        path = os.path.join(
            INPUT_DIR,
            f"{dataset}_clean.json"
        )

        if not os.path.exists(path):
            continue

        with open(
            path,
            "r",
            encoding="utf8"
        ) as f:

            docs = json.load(f)

        for doc in docs:

            doc = doc.copy()

            doc["_dataset"] = dataset

            all_docs.append(doc)

    return all_docs


# ==========================================================
# SPLIT
# ==========================================================

def split_documents(documents):

    outputs = defaultdict(list)

    for doc in documents:

        dataset = doc["_dataset"]

        new_doc = doc.copy()

        del new_doc["_dataset"]

        outputs[dataset].append(new_doc)

    return outputs


# ==========================================================
# SAVE
# ==========================================================

def save_documents(outputs, mode):

    save_dir = os.path.join(
        OUTPUT_DIR,
        mode
    )

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    for dataset in DATASETS:

        path = os.path.join(
            save_dir,
            f"{dataset}_clean.json"
        )

        with open(
            path,
            "w",
            encoding="utf8"
        ) as f:

            json.dump(
                outputs.get(dataset, []),
                f,
                ensure_ascii=False,
                indent=2
            )

        print(
            f"{dataset:<10}: {len(outputs.get(dataset, []))}"
        )


# ==========================================================
# SAVE LOG
# ==========================================================

def save_log(logs, filename, mode):

    save_dir = os.path.join(
        OUTPUT_DIR,
        mode
    )

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    path = os.path.join(
        save_dir,
        filename
    )

    with open(
        path,
        "w",
        encoding="utf8"
    ) as f:

        json.dump(
            logs,
            f,
            ensure_ascii=False,
            indent=2
        )


# ==========================================================
# SUMMARY
# ==========================================================

def print_summary(title, before, after):

    print("=" * 60)
    print(title)
    print("=" * 60)
    print(f"Original : {before}")
    print(f"Remain   : {after}")
    print(f"Removed  : {before-after}")
    print("=" * 60)