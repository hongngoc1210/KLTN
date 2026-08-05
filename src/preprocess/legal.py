from .clean_common import clean_legal
import json
import os
import re
from tqdm import tqdm


INPUT_FILE = "./data/raw_data/legal.json"
OUTPUT_FILE = "./data/process_data/legal_clean.json"
REMOVED_FILE = "./data/process_data/legal_removed.json"


# ==========================================================
# Tạo folder
# ==========================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)


# ==========================================================
# Convert id
# ==========================================================

def convert_legal_id(id_str):
    if not isinstance(id_str, str):
        return id_str

    return re.sub(
        r"\blegal_(\d+)\b",
        r"legal_vbpl_\1",
        id_str
    )


# ==========================================================
# Load data
# ==========================================================

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)


cleaned_data = []
removed_data = []


# ==========================================================
# Clean
# ==========================================================

for item in tqdm(data):

    content = item.get("content", "")

    try:
        result = clean_legal(content)

    except Exception as e:

        print(
            "\nERROR:",
            item.get("id"),
            e
        )

        result = None


    # ==========================
    # Clean failed
    # ==========================

    if result is None:

        removed_data.append({
            "id": item.get("id"),
            "reason": "clean_failed",
            "old_length": len(content)
        })

        continue


    # ==========================
    # Too short
    # ==========================

    word_count = len(result.split())

    if word_count < 30:

        removed_data.append({
            "id": item.get("id"),
            "reason": "too_short_after_clean",
            "word_count": word_count,
            "old_length": len(content)
        })

        continue


    # ==========================
    # Save cleaned item
    # ==========================

    new_item = item.copy()

    new_item["id"] = convert_legal_id(
        item.get("id")
    )

    new_item["content"] = result.strip()

    cleaned_data.append(new_item)


# ==========================================================
# Save clean data
# ==========================================================

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


# ==========================================================
# Save removed log
# ==========================================================

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


# ==========================================================
# Summary
# ==========================================================

print("=" * 40)
print("DONE")
print("Original    :", len(data))
print("Removed     :", len(removed_data))
print("Remain      :", len(cleaned_data))
print("Output      :", OUTPUT_FILE)
print("Removed log :", REMOVED_FILE)
print("=" * 40)

# import pandas as pd
# from clean_common import clean_legal
# import re

# def convert_legal_id(text: str) -> str:
#     return re.sub(r"\blegal_(\d+)\b", r"legal_vbpl_\1", text)

# # ==========================================================
# # Demo
# # ==========================================================
# df = pd.read_json("./data/raw_data/legal.json")

# text = df.loc[
#     df["id"] == "legal_congbao_57090",
#     "content"
# ].iloc[0]

# # Lưu trước khi xử lý
# with open("before.txt", "w", encoding="utf8") as f:
#     f.write(text)

# # Tiền xử lý
# text_clean = clean_legal(text)

# # Lưu sau khi xử lý
# with open("after.txt", "w", encoding="utf8") as f:
#     f.write(text_clean)

# print("Done!")
