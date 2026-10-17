import json
import re
import csv
from pathlib import Path
from rapidfuzz.fuzz import ratio


# =========================
# Config
# =========================

DATA_DIR = Path("data")

OUTPUT_FILE = DATA_DIR / "all_generated_filtered.json"
REMOVED_FILE = DATA_DIR / "removed_questions.json"
REMOVED_CSV = DATA_DIR / "removed_questions.csv"

SIMILARITY_THRESHOLD = 90


# =========================
# Load all JSON files
# =========================

all_data = []

for file in sorted(DATA_DIR.glob("*_generated.json")):

    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"{file.name}: {len(data):,} samples")

    all_data.extend(data)


print(f"\nTổng tất cả: {len(all_data):,}")


# =========================
# Normalize question
# =========================

def normalize_question(question):
    question = question.strip().lower()
    question = re.sub(r"\s+", " ", question)
    return question


# =========================
# Removed questions
# =========================

removed_data = []


# =========================
# 1. Lọc câu không phải question
#    Chỉ giữ câu kết thúc bằng ?
# =========================

before = len(all_data)

valid_data = []

for item in all_data:

    question = item.get("question")

    if (
        isinstance(question, str)
        and question.strip().endswith("?")
    ):
        valid_data.append(item)

    else:
        removed_item = item.copy()
        removed_item["remove_reason"] = "not_question"

        removed_data.append(removed_item)


all_data = valid_data


print(f"\n[1] Lọc question không hợp lệ")
print(f"Trước: {before:,}")
print(f"Sau:   {len(all_data):,}")
print(f"Loại:  {before - len(all_data):,}")


# =========================
# 2. Exact duplicate
# =========================

before = len(all_data)

seen = set()
unique_data = []

for item in all_data:

    question = normalize_question(item["question"])

    if question in seen:

        removed_item = item.copy()
        removed_item["remove_reason"] = "exact_duplicate"

        removed_data.append(removed_item)

        continue

    seen.add(question)
    unique_data.append(item)


print(f"\n[2] Exact duplicate")
print(f"Trước: {before:,}")
print(f"Sau:   {len(unique_data):,}")
print(f"Loại:  {before - len(unique_data):,}")


# =========================
# 3. Near duplicate
# =========================

before = len(unique_data)

final_data = []

# Lưu cả question + item để biết
# câu nào là câu duplicate
kept_questions = []


for i, item in enumerate(unique_data):

    question = normalize_question(item["question"])

    is_duplicate = False
    duplicate_question = None
    duplicate_score = None
    duplicate_item = None


    for old_question, old_item in kept_questions:

        score = ratio(question, old_question)

        if score >= SIMILARITY_THRESHOLD:

            is_duplicate = True
            duplicate_question = old_item["question"]
            duplicate_score = score
            duplicate_item = old_item

            break


    if not is_duplicate:

        final_data.append(item)

        kept_questions.append(
            (question, item)
        )

    else:

        removed_item = item.copy()

        removed_item["remove_reason"] = "near_duplicate"
        removed_item["similarity_score"] = round(
            duplicate_score, 2
        )

        removed_item["duplicate_of_question"] = (
            duplicate_question
        )

        removed_item["duplicate_of_chunk_id"] = (
            duplicate_item.get("chunk_id")
        )

        removed_data.append(removed_item)


    if (i + 1) % 1000 == 0:

        print(
            f"Processed: "
            f"{i + 1:,}/{len(unique_data):,}"
        )


print(f"\n[3] Near duplicate")
print(f"Trước: {before:,}")
print(f"Sau:   {len(final_data):,}")
print(f"Loại:  {before - len(final_data):,}")


# =========================
# 4. Thống kê theo domain
# =========================

print("\n[4] Phân bố domain")

domain_count = {}

for item in final_data:

    domain = item.get("domain", "unknown")

    domain_count[domain] = (
        domain_count.get(domain, 0) + 1
    )


for domain, count in sorted(domain_count.items()):

    print(f"{domain:15s}: {count:,}")


# =========================
# 5. Thống kê question_type
#    Chỉ để kiểm tra
#    KHÔNG dùng để filter
# =========================

print("\n[5] Phân bố question_type")

question_type_count = {}

for item in final_data:

    question_type = item.get(
        "question_type",
        "unknown"
    )

    question_type_count[question_type] = (
        question_type_count.get(question_type, 0) + 1
    )


for qtype, count in sorted(
    question_type_count.items()
):

    print(f"{qtype:15s}: {count:,}")


# =========================
# 6. Thống kê removed
# =========================

print("\n[6] Phân bố lý do bị loại")

remove_reason_count = {}

for item in removed_data:

    reason = item["remove_reason"]

    remove_reason_count[reason] = (
        remove_reason_count.get(reason, 0) + 1
    )


for reason, count in sorted(
    remove_reason_count.items()
):

    print(f"{reason:20s}: {count:,}")


# =========================
# 7. Save filtered data
# =========================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        final_data,
        f,
        ensure_ascii=False,
        indent=2
    )


# =========================
# 8. Save removed JSON
# =========================

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


# =========================
# 9. Save removed CSV
# =========================

if removed_data:

    fieldnames = [
        "chunk_id",
        "doc_id",
        "domain",
        "question",
        "question_type",
        "remove_reason",
        "similarity_score",
        "duplicate_of_question",
        "duplicate_of_chunk_id",
    ]

    with open(
        REMOVED_CSV,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )

        writer.writeheader()
        writer.writerows(removed_data)


# =========================
# 10. Final summary
# =========================

print("\n" + "=" * 50)
print("FINAL SUMMARY")
print("=" * 50)

print(f"Tổng ban đầu : {len(all_data) + len(removed_data):,}")
print(f"Giữ lại      : {len(final_data):,}")
print(f"Bị loại      : {len(removed_data):,}")

print("\nFiles:")

print(f"  Kept    : {OUTPUT_FILE}")
print(f"  Removed : {REMOVED_FILE}")
print(f"  CSV     : {REMOVED_CSV}")