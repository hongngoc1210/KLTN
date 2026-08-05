from .clean_common import clean_news_finance
import json
import os
from tqdm import tqdm


DATASETS = [
    {
        "name": "finance",
        "clean_func": clean_news_finance,
    },
    {
        "name": "news",
        "clean_func": clean_news_finance,   # news dùng chung hàm
    },
]


for dataset in DATASETS:

    name = dataset["name"]
    clean_func = dataset["clean_func"]

    input_file = f"./data/raw_data/{name}.json"
    output_file = f"./data/process_data/{name}_clean.json"
    removed_file = f"./data/process_data/{name}_removed.json"

    os.makedirs(
        os.path.dirname(output_file),
        exist_ok=True
    )

    with open(
        input_file,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    cleaned_data = []
    removed_data = []

    for item in tqdm(data, desc=name):

        content = item.get("content", "")

        try:
            result = clean_func(content)

        except Exception as e:

            print(
                "\nERROR:",
                item.get("id"),
                e
            )

            result = None

        if result is None:

            removed_data.append({
                "id": item.get("id"),
                "reason": "clean_failed",
                "old_length": len(content)
            })

            continue

        word_count = len(result.split())

        if word_count < 30:

            removed_data.append({
                "id": item.get("id"),
                "reason": "too_short_after_clean",
                "word_count": word_count,
                "old_length": len(content)
            })

            continue

        new_item = item.copy()
        new_item["content"] = result.strip()

        cleaned_data.append(new_item)

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            cleaned_data,
            f,
            ensure_ascii=False,
            indent=2
        )

    with open(
        removed_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            removed_data,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("=" * 40)
    print(name.upper())
    print("Original :", len(data))
    print("Removed  :", len(removed_data))
    print("Remain   :", len(cleaned_data))
    print("Output   :", output_file)
    print("Removed log:", removed_file)
    print("=" * 40)