
# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path

from together import Together


# ============================================================
# CONFIG
# ============================================================

BATCH_ID = "b3630835-001e-44fa-ac1f-18d314482089"
API_KEY_NAME = "TOGETHER_API_KEY"

BATCH_INPUT_FILE = Path(
    "data/output_deepseek/batch_work/"
    "batch_input_round1.jsonl"
)

ERROR_OUTPUT_FILE = Path(
    "data/output_deepseek/batch_work/"
    "batch_error.jsonl"
)


# ============================================================
# CLIENT
# ============================================================

def get_client():

    api_key = os.environ.get(API_KEY_NAME)

    if not api_key:
        raise RuntimeError(
            f"Không tìm thấy env {API_KEY_NAME}"
        )

    return Together(
        api_key=api_key.strip()
    )


# ============================================================
# PRINT BATCH INFO
# ============================================================

def print_batch_info(client):

    print("=" * 70)
    print("CHECK TOGETHER BATCH")
    print("=" * 70)
    print(f"Batch ID : {BATCH_ID}")

    # SDK Together hiện tại dùng retrieve()
    batch = client.batches.retrieve(
        BATCH_ID
    )

    print("\n" + "=" * 70)
    print("FULL BATCH OBJECT")
    print("=" * 70)

    try:
        print(
            batch.model_dump_json(
                indent=2
            )
        )
    except Exception:
        print(batch)

    print("\n" + "=" * 70)
    print("IMPORTANT FIELDS")
    print("=" * 70)

    fields = [
        "id",
        "status",
        "input_file_id",
        "output_file_id",
        "error_file_id",
        "error",
        "errors",
    ]

    for field in fields:

        value = getattr(
            batch,
            field,
            None,
        )

        print(
            f"{field:<20}: {value}"
        )

    return batch


# ============================================================
# FIND ERROR FILE ID
# ============================================================

def get_error_file_id(batch):

    # Một số version SDK có thể đặt tên field khác nhau
    candidates = [
        "error_file_id",
        "errors_file_id",
    ]

    for field in candidates:

        value = getattr(
            batch,
            field,
            None,
        )

        if value:
            return value

    # Thử đọc từ model dump
    try:

        batch_dict = batch.model_dump()

        for field in candidates:

            value = batch_dict.get(field)

            if value:
                return value

    except Exception:
        pass

    return None


# ============================================================
# DOWNLOAD ERROR FILE
# ============================================================

def download_error_file(
    client,
    batch,
):

    print("\n" + "=" * 70)
    print("ERROR FILE")
    print("=" * 70)

    error_file_id = get_error_file_id(
        batch
    )

    if not error_file_id:

        print(
            "Batch không có error_file_id."
        )

        print(
            "Nếu FULL BATCH OBJECT có field lỗi khác, "
            "xem trực tiếp ở phía trên."
        )

        return

    print(
        f"Error file ID : {error_file_id}"
    )

    ERROR_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:

        client.files.retrieve_content(
            id=error_file_id,
            output=str(
                ERROR_OUTPUT_FILE
            ),
        )

        print(
            f"✓ Saved: {ERROR_OUTPUT_FILE}"
        )

    except Exception as error:

        print(
            "❌ Không download được error file:"
        )

        print(
            repr(error)
        )

        return

    # ========================================================
    # PRINT FIRST ERRORS
    # ========================================================

    print("\nFIRST 20 ERRORS:")
    print("-" * 70)

    try:

        with ERROR_OUTPUT_FILE.open(
            "r",
            encoding="utf-8",
        ) as f:

            for i, line in enumerate(f):

                if i >= 20:
                    break

                line = line.strip()

                if not line:
                    continue

                try:

                    row = json.loads(
                        line
                    )

                    print(
                        json.dumps(
                            row,
                            ensure_ascii=False,
                            indent=2,
                        )
                    )

                except json.JSONDecodeError:

                    print(line)

                print("-" * 70)

    except Exception as error:

        print(
            f"❌ Không đọc được error file: {error}"
        )


# ============================================================
# CHECK ONE REQUEST SCHEMA
# ============================================================

def inspect_request(
    row,
    line_number,
):

    problems = []

    # --------------------------------------------------------
    # custom_id
    # --------------------------------------------------------

    if "custom_id" not in row:
        problems.append(
            "missing custom_id"
        )

    # --------------------------------------------------------
    # body
    # --------------------------------------------------------

    body = row.get("body")

    if not isinstance(body, dict):

        problems.append(
            "body không phải dict"
        )

        return problems

    # --------------------------------------------------------
    # model
    # --------------------------------------------------------

    if not body.get("model"):

        problems.append(
            "missing body.model"
        )

    # --------------------------------------------------------
    # messages
    # --------------------------------------------------------

    messages = body.get(
        "messages"
    )

    if not isinstance(
        messages,
        list,
    ) or not messages:

        problems.append(
            "missing/invalid body.messages"
        )

    # --------------------------------------------------------
    # stream
    # --------------------------------------------------------

    if body.get("stream") is True:

        problems.append(
            "stream=True trong batch request"
        )

    # --------------------------------------------------------
    # reasoning
    # --------------------------------------------------------

    if "reasoning_effort" in body:

        problems.append(
            "request chứa reasoning_effort="
            f"{body.get('reasoning_effort')!r}"
        )

    if "reasoning" in body:

        problems.append(
            "request chứa reasoning="
            f"{body.get('reasoning')!r}"
        )

    return problems


# ============================================================
# CHECK INPUT JSONL
# ============================================================

def check_batch_input():

    print("\n" + "=" * 70)
    print("CHECK BATCH INPUT")
    print("=" * 70)

    if not BATCH_INPUT_FILE.exists():

        print(
            f"❌ Không tìm thấy: "
            f"{BATCH_INPUT_FILE}"
        )

        return

    total = 0
    invalid_json = 0
    schema_warnings = 0

    first_request = None
    first_warnings = []

    custom_ids = set()
    duplicate_ids = set()

    with BATCH_INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            total += 1

            # =================================================
            # JSON VALIDATION
            # =================================================

            try:

                row = json.loads(
                    line
                )

            except json.JSONDecodeError as error:

                invalid_json += 1

                if invalid_json <= 10:

                    print(
                        f"❌ Invalid JSON "
                        f"line {line_number}: "
                        f"{error}"
                    )

                continue

            # =================================================
            # FIRST REQUEST
            # =================================================

            if first_request is None:

                first_request = row

            # =================================================
            # CUSTOM ID DUPLICATES
            # =================================================

            custom_id = row.get(
                "custom_id"
            )

            if custom_id:

                if custom_id in custom_ids:

                    duplicate_ids.add(
                        custom_id
                    )

                custom_ids.add(
                    custom_id
                )

            # =================================================
            # BASIC SCHEMA CHECK
            # =================================================

            warnings = inspect_request(
                row,
                line_number,
            )

            if warnings:

                schema_warnings += 1

                if len(first_warnings) < 10:

                    first_warnings.append(
                        (
                            line_number,
                            custom_id,
                            warnings,
                        )
                    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        f"Total requests       : {total:,}"
    )

    print(
        f"Invalid JSON         : {invalid_json:,}"
    )

    print(
        f"Schema warning rows  : {schema_warnings:,}"
    )

    print(
        f"Duplicate custom_id  : {len(duplicate_ids):,}"
    )

    # ========================================================
    # WARNINGS
    # ========================================================

    if first_warnings:

        print("\n" + "=" * 70)
        print("FIRST SCHEMA WARNINGS")
        print("=" * 70)

        for (
            line_number,
            custom_id,
            warnings,
        ) in first_warnings:

            print(
                f"\nLine      : {line_number}"
            )

            print(
                f"custom_id : {custom_id}"
            )

            for warning in warnings:

                print(
                    f"⚠ {warning}"
                )

    # ========================================================
    # FIRST REQUEST
    # ========================================================

    if first_request is not None:

        print("\n" + "=" * 70)
        print("FIRST REQUEST")
        print("=" * 70)

        print(
            json.dumps(
                first_request,
                ensure_ascii=False,
                indent=2,
            )
        )


# ============================================================
# MAIN
# ============================================================

def main():

    client = get_client()

    # ========================================================
    # 1. BATCH INFORMATION
    # ========================================================

    try:

        batch = print_batch_info(
            client
        )

    except Exception as error:

        print("\n" + "=" * 70)
        print("❌ KHÔNG LẤY ĐƯỢC BATCH")
        print("=" * 70)

        print(
            repr(error)
        )

        return

    # ========================================================
    # 2. ERROR FILE
    # ========================================================

    download_error_file(
        client,
        batch,
    )

    # ========================================================
    # 3. INPUT JSONL
    # ========================================================

    check_batch_input()

    print("\n" + "=" * 70)
    print("CHECK FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    main()
