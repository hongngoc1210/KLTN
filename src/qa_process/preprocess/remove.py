# -*- coding: utf-8 -*-

"""
Remove length outliers from chunking JSON files.

Input:
    data/chunking/*.json

Output:
    data/chunking_clean/*_clean.json
    data/chunking_clean/outlier_report.json

Outlier detection:
    IQR method
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

The original files are NOT modified.
"""

from pathlib import Path
import json
import statistics


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = Path("data/chunking")
OUTPUT_DIR = Path("data/chunking_clean")

IQR_MULTIPLIER = 1.5

# Nếu muốn không loại chunk quá ngắn bằng IQR,
# đặt REMOVE_SHORT = False
REMOVE_SHORT = True

# Field chứa text
TEXT_FIELD = "content"


# ============================================================
# UTILS
# ============================================================

def word_count(text):
    """
    Count words using whitespace.

    Works reasonably well for Vietnamese text.
    """
    if not isinstance(text, str):
        return 0

    return len(text.split())


def percentile(sorted_values, p):
    """
    Linear interpolation percentile.
    p: 0.0 -> 1.0
    """
    n = len(sorted_values)

    if n == 0:
        return 0

    if n == 1:
        return sorted_values[0]

    index = (n - 1) * p
    lower = int(index)
    upper = min(lower + 1, n - 1)

    weight = index - lower

    return (
        sorted_values[lower] * (1 - weight)
        + sorted_values[upper] * weight
    )


def get_iqr_bounds(values, multiplier=1.5):
    """
    Return Q1, Q3, IQR, lower bound, upper bound.
    """

    values = sorted(values)

    q1 = percentile(values, 0.25)
    q3 = percentile(values, 0.75)

    iqr = q3 - q1

    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr

    return q1, q3, iqr, lower, upper


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(input_path, output_path):

    print(f"\n{'=' * 70}")
    print(f"Processing: {input_path.name}")
    print(f"{'=' * 70}")

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"{input_path.name}: JSON root must be a list."
        )

    # --------------------------------------------------------
    # Calculate word counts
    # --------------------------------------------------------

    valid_items = []
    invalid_items = []

    for item in data:

        text = item.get(TEXT_FIELD, "")

        wc = word_count(text)

        if wc == 0:
            invalid_items.append(item)
            continue

        item["_word_count"] = wc
        valid_items.append(item)

    if not valid_items:
        print("No valid chunks.")
        return None

    word_counts = [
        item["_word_count"]
        for item in valid_items
    ]

    # --------------------------------------------------------
    # IQR
    # --------------------------------------------------------

    q1, q3, iqr, lower, upper = get_iqr_bounds(
        word_counts,
        IQR_MULTIPLIER
    )

    # --------------------------------------------------------
    # Find outliers
    # --------------------------------------------------------

    kept = []
    outliers = []

    for item in valid_items:

        wc = item["_word_count"]

        is_short_outlier = (
            REMOVE_SHORT and wc < lower
        )

        is_long_outlier = (
            wc > upper
        )

        if is_short_outlier or is_long_outlier:

            item["_outlier_reason"] = (
                "too_short"
                if is_short_outlier
                else "too_long"
            )

            outliers.append(item)

        else:
            kept.append(item)

    # --------------------------------------------------------
    # Remove temporary fields
    # --------------------------------------------------------

    for item in kept:
        item.pop("_word_count", None)

    for item in outliers:
        item.pop("_word_count", None)

    # --------------------------------------------------------
    # Save cleaned data
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            kept,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    kept_counts = [
        word_count(item.get(TEXT_FIELD, ""))
        for item in kept
    ]

    outlier_counts = [
        word_count(item.get(TEXT_FIELD, ""))
        for item in outliers
    ]

    stats = {
        "file": input_path.name,

        "original_chunks": len(data),

        "valid_chunks": len(valid_items),

        "removed_chunks": len(outliers),

        "kept_chunks": len(kept),

        "removed_ratio": (
            len(outliers) / len(valid_items)
            if valid_items
            else 0
        ),

        "q1": round(q1, 2),
        "q3": round(q3, 2),
        "iqr": round(iqr, 2),

        "lower_bound": round(lower, 2),
        "upper_bound": round(upper, 2),

        "original_min_words": min(word_counts),
        "original_max_words": max(word_counts),
        "original_mean_words": round(
            statistics.mean(word_counts),
            2
        ),
        "original_median_words": round(
            statistics.median(word_counts),
            2
        ),

        "kept_min_words": (
            min(kept_counts)
            if kept_counts
            else None
        ),

        "kept_max_words": (
            max(kept_counts)
            if kept_counts
            else None
        ),

        "kept_mean_words": (
            round(statistics.mean(kept_counts), 2)
            if kept_counts
            else None
        ),

        "kept_median_words": (
            round(statistics.median(kept_counts), 2)
            if kept_counts
            else None
        ),

        "removed_min_words": (
            min(outlier_counts)
            if outlier_counts
            else None
        ),

        "removed_max_words": (
            max(outlier_counts)
            if outlier_counts
            else None
        ),
    }

    print(f"Original : {len(data):,}")
    print(f"Valid    : {len(valid_items):,}")
    print(f"Kept     : {len(kept):,}")
    print(f"Removed  : {len(outliers):,}")

    print()
    print(f"Q1       : {q1:.2f}")
    print(f"Q3       : {q3:.2f}")
    print(f"IQR      : {iqr:.2f}")
    print(f"Lower    : {lower:.2f}")
    print(f"Upper    : {upper:.2f}")

    print()
    print(
        f"Words before: "
        f"{min(word_counts)} - "
        f"{max(word_counts)}"
    )

    if kept_counts:
        print(
            f"Words after : "
            f"{min(kept_counts)} - "
            f"{max(kept_counts)}"
        )

    print(f"Saved      : {output_path}")

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
        INPUT_DIR.glob("*.json")
    )

    # Không xử lý report cũ nếu chạy lại
    files = [
        f for f in files
        if f.name != "outlier_report.json"
    ]

    if not files:
        print(
            f"No JSON files found in {INPUT_DIR}"
        )
        return

    print(
        f"Found {len(files)} JSON files."
    )

    all_stats = []

    for input_path in files:

        output_name = (
            input_path.stem + "_clean.json"
        )

        output_path = (
            OUTPUT_DIR / output_name
        )

        try:

            stats = process_file(
                input_path,
                output_path
            )

            if stats is not None:
                all_stats.append(stats)

        except Exception as e:

            print(
                f"\nERROR processing "
                f"{input_path.name}: {e}"
            )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    report_path = (
        OUTPUT_DIR / "outlier_report.json"
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
    # Summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

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

    print(
        f"Original : {total_original:,}"
    )

    print(
        f"Kept     : {total_kept:,}"
    )

    print(
        f"Removed  : {total_removed:,}"
    )

    print(
        f"Removed %: "
        f"{total_removed / total_original * 100:.2f}%"
        if total_original
        else "Removed %: 0%"
    )

    print()
    print(
        f"Report: {report_path}"
    )


if __name__ == "__main__":
    main()