import json
import os
import re
from collections import defaultdict


# ==========================================================
# CONFIG
# ==========================================================

INPUT_DIR = "./data/chunking_rule_filtered"
OUTPUT_DIR = "./data/chunking_rule_filtered/dedup"

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

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ==========================================================
# LOAD
# ==========================================================

def load_all_documents():

    all_docs = []

    for dataset in DATASETS:

        # education -> education_chunks_filtered.json
        # educationgpt -> education_chunksgpt_filtered.json
        # finance -> finance_chunks_filtered.json

        if dataset == "educationgpt":
            filename = "education_chunksgpt_filtered.json"
        else:
            filename = f"{dataset}_chunks_filtered.json"

        path = os.path.join(
            INPUT_DIR,
            filename
        )

        if not os.path.exists(path):

            print(
                f"[WARNING] File not found: {path}"
            )

            continue

        with open(
            path,
            "r",
            encoding="utf8"
        ) as f:

            docs = json.load(f)

        for doc in docs:

            doc = doc.copy()

            # Dataset dùng cho việc split output
            doc["_dataset"] = dataset

            all_docs.append(doc)

        print(
            f"Loaded {dataset:<12}: {len(docs):,}"
        )

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

        outputs[dataset].append(
            new_doc
        )

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
            f"{dataset}_chunks.json"
        )

        docs = outputs.get(
            dataset,
            []
        )

        with open(
            path,
            "w",
            encoding="utf8"
        ) as f:

            json.dump(
                docs,
                f,
                ensure_ascii=False,
                indent=2
            )

        print(
            f"{dataset:<12}: {len(docs):,}"
        )


# ==========================================================
# SAVE LOG
# ==========================================================

def save_log(
    logs,
    filename,
    mode
):

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

def print_summary(
    title,
    before,
    after
):

    print("=" * 60)
    print(title)
    print("=" * 60)

    print(
        f"Original : {before:,}"
    )

    print(
        f"Remain   : {after:,}"
    )

    print(
        f"Removed  : {before - after:,}"
    )

    if before > 0:

        rate = (
            (before - after)
            / before
            * 100
        )

        print(
            f"Rate     : {rate:.2f}%"
        )

    print("=" * 60)