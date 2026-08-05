from .clean_common import clean_education
import json
import os
from tqdm import tqdm


INPUT_FILE = "./data/raw_data/education.json"
OUTPUT_FILE = "./data/process_data/education_clean.json"
REMOVED_FILE = "./data/process_data/education_removed.json"


os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)


with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)


cleaned_data = []
removed_data = []


for item in tqdm(data):

    content = item.get("content", "")

    try:
        result = clean_education(content)

    except Exception as e:

        print(
            "\nERROR:",
            item.get("id"),
            e
        )

        result = None


    # remove nếu clean lỗi
    if result is None:

        removed_data.append({
            "id": item.get("id"),
            "reason": "clean_failed",
            "old_length": len(content)
        })

        continue


    # đếm số từ
    word_count = len(result.split())

    # remove nếu quá ngắn
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
    OUTPUT_FILE,
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
    REMOVED_FILE,
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
print("DONE")
print("Original :", len(data))
print("Removed  :", len(removed_data))
print("Remain   :", len(cleaned_data))
print("Output   :", OUTPUT_FILE)
print("Removed log:", REMOVED_FILE)
print("=" * 40)