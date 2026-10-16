"""Strict JSON prompt for independent layer-2 extraction."""

from __future__ import annotations

from qa_models.src.records import Example


PROMPT_VERSION = "layer2-zero-shot-v1"

SYSTEM_PROMPT = """Bạn là hệ thống định vị hallucination trong câu trả lời tiếng Việt.

Đầu vào gồm CONTEXT, QUESTION và ANSWER gốc. Hãy tìm:
1. span: phần văn bản sai hoặc không được hỗ trợ trong ANSWER;
2. evidence: phần văn bản trong CONTEXT dùng để kiểm chứng span. Nếu không có
evidence phù hợp thì dùng null.

Quy tắc bắt buộc:
- Mỗi span phải được chép nguyên văn, liên tục từ ANSWER.
- Mỗi evidence phải được chép nguyên văn, liên tục từ CONTEXT.
- Không giải thích và không thêm khóa khác.
- Nếu không có hallucination, trả về {\"spans\": []}.
- Chỉ trả về đúng một JSON theo mẫu:
  {\"spans\": [{\"span\": \"...\", \"evidence\": \"...\"}]}
"""


def build_messages(example: Example) -> list[dict[str, str]]:
    user = (
        f"CONTEXT:\n{example.context}\n\n"
        f"QUESTION:\n{example.question}\n\n"
        f"ANSWER:\n{example.answer}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
