# -*- coding: utf-8 -*-

"""
Rule-based quality filtering for Vietnamese QA chunks.

Input:
    data/chunking_clean/*_clean.json

Output:
    data/chunking_rule_filtered/*_filtered.json
    data/chunking_rule_filtered/rule_filter_report.json

Purpose:
    Remove chunks that are unsuitable for Question Answering
    before sending them to an LLM quality filter.
"""

from pathlib import Path
import json
import re
from collections import Counter


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = Path("data/chunking_clean")
OUTPUT_DIR = Path("data/chunking_rule_filtered")

TEXT_FIELD = "content"

# Hard limits
MIN_WORDS = 20
MAX_WORDS = 200

# Minimum number of alphabetic characters
MIN_LETTERS = 30

# If more than this percentage is digits/symbols,
# consider the chunk suspicious.
MAX_NON_LETTER_RATIO = 0.70

# Too many repeated punctuation marks
MAX_PUNCT_RATIO = 0.35

# Duplicate handling
REMOVE_EXACT_DUPLICATES = True


# ============================================================
# REGEX
# ============================================================

# Vietnamese / Latin alphabet
LETTER_RE = re.compile(
    r"[A-Za-zÀ-ỹ]"
)

# Word
WORD_RE = re.compile(
    r"\S+"
)

# ============================================================
# STRICT SENTENCE END
# ============================================================
#
# Chunk MUST end with:
#     .
#     !
#     ?
#
# Optional closing quotation marks are allowed:
#     ."
#     .”
#     !”
#     ?'
#
# NOT accepted:
#     ...
#     …
#     ,
#     ;
#     :
#     -
#     "đang dở câu"
#
# ============================================================

SENTENCE_END_RE = re.compile(
    r'[.!?]["”’\']?$'
)

# Headings / numbering
HEADING_RE = re.compile(
    r"^\s*("
    r"(?:CHƯƠNG|Chương|PHẦN|Phần|MỤC|Mục|"
    r"SECTION|Section|BẢNG|Bảng|HÌNH|Hình)"
    r"\s+[\w\dIVXLCDM]+"
    r"|"
    r"\d+(?:\.\d+)*[\.\):\-]?\s*$"
    r")",
    re.UNICODE
)

# List-only lines
LIST_PREFIX_RE = re.compile(
    r"^\s*(?:[-•●○▪◦*]|\(?\d+[\.\)]|"
    r"\(?[a-zA-Z][\.\)])\s+"
)

# References / citations
REFERENCE_RE = re.compile(
    r"^\s*(?:tài liệu tham khảo|references?)\s*$",
    re.IGNORECASE
)

# Common discourse/reference expressions
ANAPHORA_PATTERNS = [
    r"\bđiều này\b",
    r"\bđiều đó\b",
    r"\bnhư vậy\b",
    r"\bnhư trên\b",
    r"\bnói trên\b",
    r"\bnêu trên\b",
    r"\btrên đây\b",
    r"\bdưới đây\b",
    r"\btheo đó\b",
    r"\btừ đó\b",
    r"\bqua đó\b",
    r"\bvấn đề này\b",
    r"\btrường hợp này\b",
    r"\bkết quả này\b",
    r"\bnghiên cứu này\b",
    r"\bmô hình này\b",
    r"\bphương pháp này\b",
    r"\bbiện pháp này\b",
]

ANAPHORA_RE = re.compile(
    "|".join(ANAPHORA_PATTERNS),
    re.IGNORECASE
)

# Starts with discourse connector -> often depends on previous chunk
WEAK_START_RE = re.compile(
    r"^\s*(?:"
    r"tuy nhiên|"
    r"do đó|"
    r"vì vậy|"
    r"bởi vậy|"
    r"ngoài ra|"
    r"hơn nữa|"
    r"đồng thời|"
    r"mặt khác|"
    r"trong khi đó|"
    r"theo đó|"
    r"từ đó|"
    r"qua đó|"
    r"như vậy|"
    r"vì thế|"
    r"do vậy"
    r")\b",
    re.IGNORECASE
)

# Markdown / HTML remnants
MARKUP_RE = re.compile(
    r"<[^>]+>|"
    r"\[[^\]]*\]\([^)]+\)|"
    r"[*_`#]{2,}"
)

# Table-like content
TABLE_RE = re.compile(
    r"\|.*\|"
)

# Multiple punctuation
REPEATED_PUNCT_RE = re.compile(
    r"[!?.,;:]{3,}"
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):
    if not isinstance(text, str):
        return ""

    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", "")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def word_count(text):
    return len(WORD_RE.findall(text))


def letter_count(text):
    return len(LETTER_RE.findall(text))


def punctuation_count(text):
    return sum(
        1
        for c in text
        if c in ".,!?;:，。！？；：…"
    )


# ============================================================
# INDIVIDUAL RULES
# ============================================================

def check_min_words(text):
    n = word_count(text)

    if n < MIN_WORDS:
        return False, f"too_short:{n}"

    return True, None


def check_max_words(text):
    n = word_count(text)

    if n > MAX_WORDS:
        return False, f"too_long:{n}"

    return True, None


def check_letters(text):
    n = letter_count(text)

    if n < MIN_LETTERS:
        return False, f"too_few_letters:{n}"

    return True, None


def check_non_letter_ratio(text):
    if not text:
        return False, "empty"

    letters = letter_count(text)

    ratio = 1 - (letters / len(text))

    if ratio > MAX_NON_LETTER_RATIO:
        return False, f"too_many_non_letters:{ratio:.2f}"

    return True, None


def check_punctuation_ratio(text):
    if not text:
        return False, "empty"

    punct = punctuation_count(text)

    ratio = punct / len(text)

    if ratio > MAX_PUNCT_RATIO:
        return False, f"too_many_punctuation:{ratio:.2f}"

    return True, None


def check_heading(text):
    if HEADING_RE.match(text):
        return False, "heading"

    return True, None


def check_reference(text):
    if REFERENCE_RE.match(text):
        return False, "reference_heading"

    return True, None


def check_table(text):
    if TABLE_RE.search(text):
        return False, "table_like"

    return True, None


def check_markup(text):
    if MARKUP_RE.search(text):
        return False, "markup"

    return True, None


def check_list_only(text):
    """
    Reject chunks that consist mostly of list items.
    """

    lines = [
        x.strip()
        for x in text.split("\n")
        if x.strip()
    ]

    if not lines:
        return False, "empty"

    list_lines = sum(
        1
        for line in lines
        if LIST_PREFIX_RE.match(line)
    )

    if len(lines) >= 2:
        ratio = list_lines / len(lines)

        if ratio >= 0.75:
            return False, "list_like"

    return True, None


def check_repeated_punctuation(text):
    if REPEATED_PUNCT_RE.search(text):
        return False, "repeated_punctuation"

    return True, None


# ============================================================
# STRICT SENTENCE STRUCTURE
# ============================================================

def check_sentence_structure(text):
    """
    Chunk MUST end with a complete sentence.

    Accepted:
        "Đây là một kết quả quan trọng."
        "Kết quả có ý nghĩa thống kê!"
        "Nghiên cứu cho thấy điều này?"

    Rejected:
        "Đây là một kết quả quan trọng"
        "Đây là một kết quả quan trọng,"
        "Đây là một kết quả quan trọng;"
        "Đây là một kết quả quan trọng:"
        "Đây là một kết quả quan trọng..."
        "Đây là một kết quả quan trọng…"
        "Kết quả cho thấy nguyên nhân của"
        "Nghiên cứu tập trung vào"
    """

    if not text:
        return False, "empty"

    # Bắt buộc kết thúc bằng:
    # . ! ?
    # Có thể có quotation mark phía sau.
    if not SENTENCE_END_RE.search(text):
        return False, "incomplete_sentence"

    return True, None


def check_anaphora(text):
    """
    Detect excessive context-dependent references.

    One occurrence is acceptable.
    Too many suggests the chunk depends heavily on previous text.
    """

    matches = ANAPHORA_RE.findall(text)

    if len(matches) >= 3:
        return False, f"too_many_anaphora:{len(matches)}"

    return True, None


def check_weak_start(text):
    """
    A chunk beginning with 'Tuy nhiên', 'Do đó', etc.
    isn't automatically bad.

    Only reject if it is very short because such a chunk
    is more likely to depend on the previous chunk.
    """

    if WEAK_START_RE.match(text):

        n = word_count(text)

        if n < 35:
            return False, "weak_context_dependent_start"

    return True, None


# ============================================================
# MAIN QUALITY CHECK
# ============================================================

def evaluate_chunk(text):
    """
    Return:

        keep: bool
        reasons: list[str]
    """

    reasons = []

    text = normalize_text(text)

    checks = [
        check_min_words,
        check_max_words,
        check_letters,
        check_non_letter_ratio,
        check_punctuation_ratio,
        check_heading,
        check_reference,
        check_table,
        check_markup,
        check_list_only,
        check_repeated_punctuation,

        # STRICT:
        # chunk bắt buộc phải kết thúc câu hoàn chỉnh
        check_sentence_structure,

        check_anaphora,
        check_weak_start,
    ]

    for check in checks:

        ok, reason = check(text)

        if not ok:
            reasons.append(reason)

    return len(reasons) == 0, reasons


# ============================================================
# PROCESS FILE
# ============================================================

def process_file(input_path, output_path):

    print("\n" + "=" * 70)
    print(f"Processing: {input_path.name}")
    print("=" * 70)

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"{input_path.name}: root must be a list"
        )

    kept = []
    removed = []

    reason_counter = Counter()

    seen = set()

    for item in data:

        text = normalize_text(
            item.get(TEXT_FIELD, "")
        )

        # ----------------------------------------------------
        # Exact duplicate
        # ----------------------------------------------------

        normalized_key = text.lower()

        if (
            REMOVE_EXACT_DUPLICATES
            and normalized_key in seen
        ):
            removed.append({
                **item,
                "_filter_reasons": [
                    "exact_duplicate"
                ]
            })

            reason_counter["exact_duplicate"] += 1

            continue

        seen.add(normalized_key)

        # ----------------------------------------------------
        # Quality checks
        # ----------------------------------------------------

        keep, reasons = evaluate_chunk(text)

        if keep:

            # Save normalized content
            item[TEXT_FIELD] = text

            kept.append(item)

        else:

            removed.append({
                **item,
                "_filter_reasons": reasons
            })

            for reason in reasons:
                reason_counter[reason] += 1

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            kept,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # Stats
    # --------------------------------------------------------

    original = len(data)
    kept_n = len(kept)
    removed_n = len(removed)

    stats = {
        "file": input_path.name,
        "original_chunks": original,
        "kept_chunks": kept_n,
        "removed_chunks": removed_n,
        "removed_ratio": (
            removed_n / original
            if original
            else 0
        ),
        "reasons": dict(
            reason_counter
        )
    }

    print(f"Original : {original:,}")
    print(f"Kept     : {kept_n:,}")
    print(f"Removed  : {removed_n:,}")

    print("\nReasons:")

    for reason, count in reason_counter.most_common():

        print(
            f"  {reason:<40} {count:,}"
        )

    print(
        f"\nSaved → {output_path}"
    )

    return stats


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    files = sorted(
        INPUT_DIR.glob("*_clean.json")
    )

    if not files:

        print(
            f"No *_clean.json files found in "
            f"{INPUT_DIR}"
        )

        return

    print(
        f"Found {len(files)} files."
    )

    all_stats = []

    for input_path in files:

        output_path = (
            OUTPUT_DIR
            / input_path.name.replace(
                "_clean.json",
                "_filtered.json"
            )
        )

        try:

            stats = process_file(
                input_path,
                output_path
            )

            all_stats.append(stats)

        except Exception as e:

            print(
                f"\nERROR: "
                f"{input_path.name}: {e}"
            )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report_path = (
        OUTPUT_DIR
        / "rule_filter_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_stats,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # Global summary
    # --------------------------------------------------------

    total_original = sum(
        x["original_chunks"]
        for x in all_stats
    )

    total_kept = sum(
        x["kept_chunks"]
        for x in all_stats
    )

    total_removed = sum(
        x["removed_chunks"]
        for x in all_stats
    )

    print("\n" + "=" * 70)
    print("GLOBAL SUMMARY")
    print("=" * 70)

    print(
        f"Original : {total_original:,}"
    )

    print(
        f"Kept     : {total_kept:,}"
    )

    print(
        f"Removed  : {total_removed:,}"
    )

    if total_original:

        print(
            f"Removed  : "
            f"{total_removed / total_original * 100:.2f}%"
        )

    print(
        f"\nReport → {report_path}"
    )


if __name__ == "__main__":
    main()