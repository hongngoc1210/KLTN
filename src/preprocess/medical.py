from .clean_common import clean_vjst, clean_text
import json
import os
from tqdm import tqdm


INPUT_FILE = "./data/raw_data/medical.json"
OUTPUT_FILE = "./data/process_data/medical_clean.json"
REMOVED_FILE = "./data/process_data/medical_removed.json"


# tạo folder nếu chưa có
os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)


# đọc json
with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)



cleaned_data = []
removed_data = []



for item in tqdm(data):

    item_id = item.get("id", "")
    content = item.get("content", "")


    try:
        if item_id.startswith("medical_vjst_"):
            result = clean_vjst(content)
        else:
            result = clean_text(content)

    except Exception as e:

        print(
            "\nERROR:",
            item.get("id"),
            e
        )

        result = None



    # ==========================
    # REMOVE EMPTY CONTENT
    # ==========================

    if (result is None or len(result.strip()) < 200):
        removed_data.append({
            "id": item.get("id"),
            "reason": "empty_after_clean",
            "old_length": len(content)
        })
        continue



    # tạo object mới
    new_item = item.copy()
    new_item["content"] = result.strip()

    cleaned_data.append(new_item)



# ==========================
# SAVE CLEAN DATA
# ==========================

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



# ==========================
# SAVE REMOVED LOG
# ==========================

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