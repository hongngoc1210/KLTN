import json
import random
import re
from collections import defaultdict
from pathlib import Path

# ==========================================================
# CONFIG
# ==========================================================

INPUT_DIR = Path(
    "data/chunking_rule_filtered/dedup/near"
)

OUTPUT_DIR = Path(
    "data/qa_context_candidates"
)

DATASETS = [
    "education",
    "finance",
    "legal",
    "medical",
    "news",
    "science",
]

TARGET = 5000

# Số chunk tối đa lấy từ cùng một document
MAX_PER_DOC = 5

# Random seed để lần chạy sau reproducible
SEED = 42

random.seed(SEED)


# ==========================================================
# BASIC FILTER CONFIG
# ==========================================================

MIN_WORDS = 40
MAX_WORDS = 200

MIN_SENTENCES = 2

# Loại chunk nếu tỷ lệ ký tự không phải chữ/số quá cao
MAX_NON_ALNUM_RATIO = 0.30

# URL
URL_PATTERN = re.compile(
    r"https?://|www\.",
    re.IGNORECASE
)

# Reference / citation
REFERENCE_PATTERN = re.compile(
    r"\[[0-9,\-\s]+\]"
)

# Các đoạn thường không hữu ích để sinh QA
BAD_PATTERNS = [
    r"^tài liệu tham khảo$",
    r"^references$",
    r"^reference$",
    r"^phụ lục$",
    r"^appendix$",
    r"^mục lục$",
    r"^table of contents$",
]


# ==========================================================
# LOAD
# ==========================================================

def load_dataset(dataset):

    path = (
        INPUT_DIR
        / f"{dataset}_chunks.json"
    )

    if not path.exists():

        print(
            f"[WARNING] Missing: {path}"
        )

        return []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        docs = json.load(f)

    print(
        f"Loaded {dataset:<12}: "
        f"{len(docs):,}"
    )

    return docs


# ==========================================================
# TEXT UTILITIES
# ==========================================================

def word_count(text):

    return len(
        text.split()
    )


def sentence_count(text):

    # Đếm tương đối số câu.
    # Không dùng để tokenize chính thức.
    sentences = re.split(
        r"[.!?。！？]+",
        text
    )

    return len([
        s for s in sentences
        if s.strip()
    ])


def non_alnum_ratio(text):

    if not text:
        return 1.0

    non_alnum = sum(
        1
        for c in text
        if not c.isalnum()
        and not c.isspace()
    )

    return (
        non_alnum
        / len(text)
    )


def is_bad_heading(text):

    normalized = (
        text
        .strip()
        .lower()
    )

    for pattern in BAD_PATTERNS:

        if re.fullmatch(
            pattern,
            normalized
        ):
            return True

    return False


# ==========================================================
# HARD FILTER
# ==========================================================

def basic_filter(doc):

    text = doc.get(
        "content",
        ""
    ).strip()

    if not text:
        return False

    wc = doc.get(
        "word_count",
        word_count(text)
    )

    # ----------------------------------------------
    # Length
    # ----------------------------------------------

    if wc < MIN_WORDS:
        return False

    if wc > MAX_WORDS:
        return False

    # ----------------------------------------------
    # Sentence
    # ----------------------------------------------

    if sentence_count(text) < MIN_SENTENCES:
        return False

    # ----------------------------------------------
    # Không phải heading/reference đơn thuần
    # ----------------------------------------------

    if is_bad_heading(text):
        return False

    # ----------------------------------------------
    # URL quá nhiều
    # ----------------------------------------------

    if URL_PATTERN.search(text):

        # Nếu có URL và chunk ngắn
        # thường không hữu ích
        if wc < 80:
            return False

    # ----------------------------------------------
    # Ký tự rác
    # ----------------------------------------------

    if non_alnum_ratio(text) > MAX_NON_ALNUM_RATIO:
        return False

    return True


# ==========================================================
# QUALITY SCORE
# ==========================================================

def quality_score(doc):

    text = doc.get(
        "content",
        ""
    ).strip()

    wc = doc.get(
        "word_count",
        word_count(text)
    )

    sentences = sentence_count(text)

    score = 0.0

    # ======================================================
    # 1. LENGTH
    # ======================================================
    #
    # 40 -> thấp
    # 70-100 -> cao
    # 128 -> vẫn cao
    #
    # QA thường cần context đủ thông tin.
    # ======================================================

    if 60 <= wc <= 110:

        score += 0.30

    elif 45 <= wc < 60:

        score += 0.22

    elif 110 < wc <= 128:

        score += 0.25

    else:

        score += 0.15

    # ======================================================
    # 2. SENTENCE COUNT
    # ======================================================

    if sentences >= 4:

        score += 0.25

    elif sentences == 3:

        score += 0.22

    elif sentences == 2:

        score += 0.18

    # ======================================================
    # 3. INFORMATION DENSITY
    # ======================================================

    # Ưu tiên chunk có số, năm, đơn vị,
    # tên riêng, factual information.

    information_signals = 0

    information_signals += len(
        re.findall(
            r"\b\d+(?:[.,]\d+)?\b",
            text
        )
    )

    information_signals += len(
        re.findall(
            r"\b(?:19|20)\d{2}\b",
            text
        )
    )

    information_signals += len(
        re.findall(
            r"%|kg|g|mg|km|cm|mm|°C|USD|VND",
            text,
            re.IGNORECASE
        )
    )

    if information_signals >= 4:

        score += 0.20

    elif information_signals >= 2:

        score += 0.15

    elif information_signals >= 1:

        score += 0.10

    # ======================================================
    # 4. STRUCTURE
    # ======================================================

    # Ưu tiên chunk có câu kết thúc rõ ràng.
    if re.search(
        r"[.!?。！？]$",
        text
    ):

        score += 0.15

    # ======================================================
    # 5. PENALTY
    # ======================================================

    # Quá nhiều citation
    citation_count = len(
        REFERENCE_PATTERN.findall(
            text
        )
    )

    if citation_count >= 5:

        score -= 0.10

    elif citation_count >= 3:

        score -= 0.05

    # URL
    if URL_PATTERN.search(text):

        score -= 0.10

    # ======================================================
    # CLAMP
    # ======================================================

    return max(
        0.0,
        min(
            1.0,
            score
        )
    )


# ==========================================================
# DOCUMENT DIVERSITY SAMPLING
# ==========================================================

def take_diverse(
    docs,
    target,
    max_per_doc
):

    if not docs:
        return []

    # ------------------------------------------------------
    # Group by document
    # ------------------------------------------------------

    by_doc = defaultdict(list)

    for doc in docs:

        doc_id = doc.get(
            "doc_id"
        )

        if not doc_id:

            doc_id = doc.get(
                "chunk_id"
            )

        by_doc[doc_id].append(
            doc
        )

    # ------------------------------------------------------
    # Shuffle documents
    # ------------------------------------------------------

    doc_ids = list(
        by_doc.keys()
    )

    random.shuffle(
        doc_ids
    )

    # ------------------------------------------------------
    # Shuffle chunks inside document
    # ------------------------------------------------------

    for doc_id in doc_ids:

        random.shuffle(
            by_doc[doc_id]
        )

    selected = []

    # ------------------------------------------------------
    # Round-robin
    # ------------------------------------------------------

    for round_idx in range(
        max_per_doc
    ):

        for doc_id in doc_ids:

            items = by_doc[doc_id]

            if round_idx >= len(items):
                continue

            selected.append(
                items[round_idx]
            )

            if len(selected) >= target:

                return selected

    return selected


# ==========================================================
# BUILD POOLS
# ==========================================================

def build_pools(docs):

    # Tính score
    scored = []

    for doc in docs:

        score = quality_score(
            doc
        )

        item = doc.copy()

        item["_quality_score"] = score

        scored.append(
            item
        )

    # Cao -> thấp
    scored.sort(
        key=lambda x: x["_quality_score"],
        reverse=True
    )

    n = len(scored)

    high_end = int(
        n * 0.60
    )

    medium_end = int(
        n * 0.90
    )

    high = scored[
        :high_end
    ]

    medium = scored[
        high_end:medium_end
    ]

    low = scored[
        medium_end:
    ]

    return (
        high,
        medium,
        low
    )


# ==========================================================
# SAMPLE 5000
# ==========================================================

def sample_candidates(docs):

    high, medium, low = build_pools(
        docs
    )

    # ======================================================
    # TARGET DISTRIBUTION
    # ======================================================
    #
    # 60% high quality
    # 30% medium
    # 10% random/low
    # ======================================================

    target_high = int(
        TARGET * 0.60
    )

    target_medium = int(
        TARGET * 0.30
    )

    target_low = (
        TARGET
        - target_high
        - target_medium
    )

    # ------------------------------------------------------
    # High
    # ------------------------------------------------------

    selected_high = take_diverse(
        high,
        target_high,
        MAX_PER_DOC
    )

    # ------------------------------------------------------
    # Medium
    # ------------------------------------------------------

    selected_medium = take_diverse(
        medium,
        target_medium,
        MAX_PER_DOC
    )

    # ------------------------------------------------------
    # Low/random
    # ------------------------------------------------------

    selected_low = take_diverse(
        low,
        target_low,
        MAX_PER_DOC
    )

    selected = (
        selected_high
        + selected_medium
        + selected_low
    )

    # ------------------------------------------------------
    # Nếu chưa đủ 5000
    # ------------------------------------------------------

    if len(selected) < TARGET:

        selected_ids = {
            doc["chunk_id"]
            for doc in selected
        }

        remaining = [
            doc
            for doc in docs
            if doc["chunk_id"]
            not in selected_ids
        ]

        random.shuffle(
            remaining
        )

        selected.extend(
            remaining[
                :TARGET - len(selected)
            ]
        )

    # ------------------------------------------------------
    # Shuffle cuối
    # ------------------------------------------------------

    random.shuffle(
        selected
    )

    # ------------------------------------------------------
    # Remove internal score
    # ------------------------------------------------------

    final = []

    for doc in selected:

        doc = doc.copy()

        doc.pop(
            "_quality_score",
            None
        )

        final.append(
            doc
        )

    return final


# ==========================================================
# MAIN
# ==========================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PRE-LLM CONTEXT SAMPLING"
    )

    print(
        "=" * 70
    )

    for dataset in DATASETS:

        print(
            f"\n{'=' * 60}"
        )

        print(
            f"DOMAIN: {dataset}"
        )

        print(
            f"{'=' * 60}"
        )

        # --------------------------------------------------
        # Load
        # --------------------------------------------------

        docs = load_dataset(
            dataset
        )

        if not docs:

            continue

        # --------------------------------------------------
        # Basic filter
        # --------------------------------------------------

        filtered = [
            doc
            for doc in docs
            if basic_filter(doc)
        ]

        print(
            f"After basic filter : "
            f"{len(filtered):,}"
        )

        if len(filtered) < TARGET:

            print(
                f"[WARNING] Only "
                f"{len(filtered):,} "
                f"valid contexts."
            )

        # --------------------------------------------------
        # Sampling
        # --------------------------------------------------

        selected = sample_candidates(
            filtered
        )

        print(
            f"Selected for LLM   : "
            f"{len(selected):,}"
        )

        # --------------------------------------------------
        # Save
        # --------------------------------------------------

        output_path = (
            OUTPUT_DIR
            / f"{dataset}_5000.json"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                selected,
                f,
                ensure_ascii=False,
                indent=2
            )

        print(
            f"Saved              : "
            f"{output_path}"
        )


# ==========================================================
# ENTRY
# ==========================================================

if __name__ == "__main__":

    main()