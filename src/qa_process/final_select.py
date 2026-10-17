import json
import random
from pathlib import Path
from collections import Counter


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = Path("data/qa_filltered")

OUTPUT_DIR = INPUT_DIR / "selected_2500"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_PER_DOMAIN = 2500
SEED = 42

random.seed(SEED)


# ============================================================
# BASE RATIOS
# ============================================================
# Tỷ lệ cơ sở:
#
# KEEP score 5 : 20%
# KEEP score 4 : 30%
# KEEP score 3 : 20%
# WEAK         : 30%
#
# Mỗi domain sẽ random lệch nhẹ quanh các tỷ lệ này.
# Tổng cuối cùng luôn = 100%.
# ============================================================

BASE_RATIOS = {
    ("KEEP", 5): 0.20,
    ("KEEP", 4): 0.30,
    ("KEEP", 3): 0.20,
    ("WEAK", None): 0.30,
}


# ============================================================
# RANDOMIZE RATIOS
# ============================================================

def randomize_ratios(base_ratios, max_noise=0.05):
    """
    Tạo tỷ lệ hơi khác nhau cho từng domain.

    max_noise=0.05
    nghĩa là mỗi nhóm dao động khoảng +/- 5 percentage points.

    Sau đó normalize để tổng = 1.
    """

    raw = {}

    for key, ratio in base_ratios.items():

        noise = random.uniform(
            -max_noise,
            max_noise
        )

        raw[key] = max(
            0.05,
            ratio + noise
        )

    total = sum(raw.values())

    ratios = {
        key: value / total
        for key, value in raw.items()
    }

    return ratios


# ============================================================
# LOAD FILES
# ============================================================

files = sorted(
    INPUT_DIR.glob("*_5000_filtered.json")
)

if not files:
    raise FileNotFoundError(
        f"Không tìm thấy *_5000_filtered.json "
        f"trong {INPUT_DIR}"
    )


print(f"Tìm thấy {len(files)} file:")
for f in files:
    print(f"  - {f.name}")


# ============================================================
# LOAD DATA + GROUP DOMAIN
# ============================================================

domains = {}

for path in files:

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    if not data:
        print(f"WARNING: {path.name} rỗng")
        continue

    domain = data[0].get("domain")

    if not domain:
        domain = path.stem.replace(
            "_5000_filtered",
            ""
        )

    domains[domain] = data


# ============================================================
# SELECT
# ============================================================

summary = []


for domain, data in sorted(domains.items()):

    print("\n" + "=" * 70)
    print(f"DOMAIN: {domain}")
    print(f"Input: {len(data):,}")


    # ========================================================
    # GROUP
    # ========================================================
    #
    # QUAN TRỌNG:
    #
    # KEEP score 3 != WEAK score 3
    #
    # label quyết định nhóm trước.
    # score chỉ được dùng bên trong KEEP.
    # ========================================================

    groups = {
        ("KEEP", 5): [],
        ("KEEP", 4): [],
        ("KEEP", 3): [],
        ("WEAK", None): [],
    }

    dropped = 0
    unknown = 0

    for item in data:

        llm = item.get(
            "_llm_filter",
            {}
        )

        label = llm.get("label")
        score = llm.get("score")

        # ------------------------------
        # KEEP
        # ------------------------------

        if label == "KEEP":

            if score in {3, 4, 5}:

                groups[
                    ("KEEP", score)
                ].append(item)

            else:
                unknown += 1

        # ------------------------------
        # WEAK
        # ------------------------------

        elif label == "WEAK":

            # Không quan tâm score của WEAK
            groups[
                ("WEAK", None)
            ].append(item)

        # ------------------------------
        # DROP
        # ------------------------------

        elif label == "DROP":

            dropped += 1

        else:

            unknown += 1


    # ========================================================
    # AVAILABLE
    # ========================================================

    print("\nAvailable:")

    for key, items in groups.items():

        print(
            f"  {str(key):<20} "
            f"{len(items):,}"
        )

    print(f"  {'DROP':<20} {dropped:,}")

    if unknown:
        print(
            f"  {'UNKNOWN/INVALID':<20} "
            f"{unknown:,}"
        )


    # ========================================================
    # RANDOM RATIOS
    # ========================================================

    ratios = randomize_ratios(
        BASE_RATIOS,
        max_noise=0.05
    )

    print("\nRandom ratios:")

    for key, ratio in ratios.items():

        print(
            f"  {str(key):<20} "
            f"{ratio * 100:.1f}%"
        )


    # ========================================================
    # CALCULATE QUOTA
    # ========================================================

    quotas = {}

    for key, ratio in ratios.items():

        quotas[key] = round(
            TARGET_PER_DOMAIN * ratio
        )


    # Fix rounding
    diff = (
        TARGET_PER_DOMAIN
        - sum(quotas.values())
    )

    if diff != 0:

        largest = max(
            quotas,
            key=quotas.get
        )

        quotas[largest] += diff


    print("\nTarget quotas:")

    for key, quota in quotas.items():

        print(
            f"  {str(key):<20} "
            f"{quota:,}"
        )


    # ========================================================
    # SAMPLE
    # ========================================================

    selected = []

    shortage = 0

    for key, quota in quotas.items():

        available = groups[key]

        take = min(
            quota,
            len(available)
        )

        if take < quota:

            shortage += (
                quota - take
            )

            print(
                f"WARNING: {key} thiếu "
                f"{quota - take} mẫu"
            )

        if take > 0:

            sampled = random.sample(
                available,
                take
            )

            selected.extend(
                sampled
            )


    # ========================================================
    # HANDLE SHORTAGE
    # ========================================================
    #
    # Nếu một nhóm không đủ quota,
    # bù từ những mẫu hợp lệ còn lại.
    #
    # KHÔNG lấy DROP.
    # ========================================================

    if shortage > 0:

        selected_ids = {
            item.get("chunk_id")
            for item in selected
        }

        remaining = []

        for key, items in groups.items():

            for item in items:

                chunk_id = item.get(
                    "chunk_id"
                )

                if chunk_id not in selected_ids:

                    remaining.append(
                        item
                    )

        random.shuffle(
            remaining
        )

        extra = remaining[
            :shortage
        ]

        selected.extend(
            extra
        )


    # ========================================================
    # REMOVE DUPLICATE chunk_id
    # ========================================================

    unique = {}

    for item in selected:

        chunk_id = item.get(
            "chunk_id"
        )

        if chunk_id is not None:

            unique[chunk_id] = item

    selected = list(
        unique.values()
    )


    # ========================================================
    # FINAL SIZE
    # ========================================================

    if len(selected) > TARGET_PER_DOMAIN:

        selected = random.sample(
            selected,
            TARGET_PER_DOMAIN
        )

    elif len(selected) < TARGET_PER_DOMAIN:

        print(
            f"\nWARNING:"
            f" chỉ lấy được "
            f"{len(selected):,}/"
            f"{TARGET_PER_DOMAIN:,}"
        )


    # Shuffle cuối
    random.shuffle(
        selected
    )


    # ========================================================
    # SAVE
    # ========================================================

    output_path = (
        OUTPUT_DIR
        / f"{domain}_2500.json"
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


    # ========================================================
    # STATISTICS
    # ========================================================

    counter = Counter()

    for item in selected:

        llm = item.get(
            "_llm_filter",
            {}
        )

        label = llm.get(
            "label"
        )

        score = llm.get(
            "score"
        )

        if label == "KEEP":

            counter[
                f"KEEP_{score}"
            ] += 1

        elif label == "WEAK":

            counter[
                "WEAK"
            ] += 1

        elif label == "DROP":

            counter[
                "DROP"
            ] += 1


    # ========================================================
    # PRINT RESULT
    # ========================================================

    print("\nSelected:")

    for key, count in sorted(
        counter.items()
    ):

        percentage = (
            count / len(selected) * 100
            if selected
            else 0
        )

        print(
            f"  {key:<12}: "
            f"{count:,} "
            f"({percentage:.1f}%)"
        )


    print(
        f"\nSaved: {output_path}"
    )


    summary.append({
        "domain": domain,
        "input": len(data),
        "selected": len(selected),
        **dict(counter),
    })


# ============================================================
# GLOBAL SUMMARY + SAVE STATISTICS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

total_selected = 0

for item in summary:

    print(
        f"{item['domain']:<12} "
        f"{item['selected']:,}"
    )

    total_selected += item["selected"]


print(
    f"\nTotal: "
    f"{total_selected:,}"
)


# ============================================================
# SAVE STATISTICS
# ============================================================

statistics = {
    "config": {
        "target_per_domain": TARGET_PER_DOMAIN,
        "seed": SEED,
        "base_ratios": {
            f"{label}_{score}" if score is not None else label: ratio
            for (label, score), ratio in BASE_RATIOS.items()
        },
        "random_noise": 0.05,
    },

    "domains": summary,

    "total": {
        "domains": len(summary),
        "selected": total_selected,
        "expected": len(summary) * TARGET_PER_DOMAIN,
    }
}


stats_path = (
    OUTPUT_DIR
    / "selection_statistics.json"
)


with open(
    stats_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        statistics,
        f,
        ensure_ascii=False,
        indent=2
    )


print(
    f"\nStatistics saved to:"
    f"\n{stats_path}"
)