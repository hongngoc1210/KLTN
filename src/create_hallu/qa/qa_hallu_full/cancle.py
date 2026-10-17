# -*- coding: utf-8 -*-

import os
from together import Together


BATCH_ID = "b3630835-001e-44fa-ac1f-18d314482089"


api_key = os.environ.get("TOGETHER_API_KEY")

if not api_key:
    raise RuntimeError(
        "Không tìm thấy TOGETHER_API_KEY"
    )

client = Together(api_key=api_key)


print("=" * 70)
print("TOGETHER BATCH CANCEL")
print("=" * 70)

# ============================================================
# 1. CHECK TRẠNG THÁI TRƯỚC KHI CANCEL
# ============================================================

batch = client.batches.retrieve(BATCH_ID)

print(f"Batch ID : {batch.id}")
print(f"Status   : {batch.status}")
print(f"Progress : {getattr(batch, 'progress', None)}")
print(f"Error    : {getattr(batch, 'error', None)}")


# ============================================================
# 2. NẾU ĐÃ XONG THÌ KHÔNG CANCEL
# ============================================================

if batch.status == "COMPLETED":

    print("\n⚠ Batch đã COMPLETED.")
    print("Không cancel.")

    print(
        "Output file:",
        getattr(batch, "output_file_id", None),
    )

    raise SystemExit(0)


if batch.status in {
    "FAILED",
    "EXPIRED",
    "CANCELLED",
}:

    print(
        f"\n⚠ Batch đã ở trạng thái {batch.status}."
    )

    raise SystemExit(0)


# ============================================================
# 3. CANCEL
# ============================================================

print("\nĐang cancel batch...")

try:

    response = client.batches.cancel(
        BATCH_ID
    )

    print("✓ Đã gửi yêu cầu cancel.")

except Exception as error:

    print(
        f"❌ Cancel API error: {error}"
    )

    raise


# ============================================================
# 4. CHECK LẠI
# ============================================================

print("\nKiểm tra trạng thái sau cancel...")

batch = client.batches.retrieve(
    BATCH_ID
)

print(f"Status   : {batch.status}")
print(f"Progress : {getattr(batch, 'progress', None)}")

print(
    "Output file:",
    getattr(batch, "output_file_id", None),
)

print(
    "Error file :",
    getattr(batch, "error_file_id", None),
)

print("=" * 70)