import json
import random
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = Path("data/qa_filltered/selected_2500")
OUTPUT_DIR = INPUT_DIR / "manual_qa"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATIO = 0.10
SEED = 42

random.seed(SEED)

# ============================================================
# RANDOM 10% TỪNG DOMAIN
# ============================================================

files = sorted(INPUT_DIR.glob("*_2500.json"))

if not files:
    raise FileNotFoundError(
        f"Không tìm thấy *_2500.json trong {INPUT_DIR}"
    )

all_samples = []
statistics = []

for path in files:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        continue

    sample_size = round(len(data) * SAMPLE_RATIO)

    sampled = random.sample(data, sample_size)

    # Tạo template để tự điền question + answer
    template = []

    for item in sampled:
        template.append({
            "chunk_id": item.get("chunk_id"),
            "doc_id": item.get("doc_id"),
            "domain": item.get("domain"),
            "content": item.get("content"),
            "question": "",
            "answer": ""
        })

    domain = sampled[0].get("domain") if sampled else path.stem.replace("_2500", "")

    output_path = OUTPUT_DIR / f"{domain}_manual_qa.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            template,
            f,
            ensure_ascii=False,
            indent=2
        )

    all_samples.extend(template)

    statistics.append({
        "domain": domain,
        "input": len(data),
        "sample_ratio": SAMPLE_RATIO,
        "sampled": len(template),
        "output": str(output_path)
    })

# ============================================================
# SHUFFLE TOÀN BỘ 1.500 MẪU
# ============================================================

random.shuffle(all_samples)

all_output = OUTPUT_DIR / "manual_qa_1500_template.json"

with open(all_output, "w", encoding="utf-8") as f:
    json.dump(
        all_samples,
        f,
        ensure_ascii=False,
        indent=2
    )

# ============================================================
# STATISTICS
# ============================================================

stats = {
    "seed": SEED,
    "sample_ratio": SAMPLE_RATIO,
    "total_input": sum(x["input"] for x in statistics),
    "total_sampled": len(all_samples),
    "domains": statistics
}

stats_output = OUTPUT_DIR / "manual_qa_statistics.json"

with open(stats_output, "w", encoding="utf-8") as f:
    json.dump(
        stats,
        f,
        ensure_ascii=False,
        indent=2
    )

print("=" * 60)
print("ĐÃ TẠO MẪU ĐỂ ĐẶT CÂU HỎI")
print("=" * 60)

for x in statistics:
    print(
        f"{x['domain']:<12} "
        f"{x['sampled']:>4} / {x['input']}"
    )

print("-" * 60)
print(f"Tổng: {len(all_samples)} mẫu")
print(f"Template: {all_output}")
print(f"Statistics: {stats_output}")
print(f"Seed: {SEED}")
