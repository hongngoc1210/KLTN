"""Diagnosis-only prompts; no response label, evidence or repair annotation."""

from __future__ import annotations

import json
from typing import Sequence

from qa_models.tasks.layer3.labels import DIAGNOSIS_LABELS
from qa_models.tasks.layer3.records import DiagnosisExample


PROMPT_VERSION = "layer3-gold-span-diagnosis-baseline-v4"
LABELS = DIAGNOSIS_LABELS

SYSTEM_PROMPT = """Bạn là chuyên gia phân loại lỗi hallucination trong câu trả lời tiếng Việt.

NHIỆM VỤ
Một đoạn trong ANSWER đã được xác định là hallucination và được đánh dấu [SPAN]...[/SPAN]. Đối chiếu đoạn này với CONTEXT và QUESTION, rồi chọn đúng MỘT nhãn mô tả loại lỗi. Không tìm span khác, không sửa câu trả lời.

CONTEXT là căn cứ xác minh; QUESTION chỉ giúp hiểu yêu cầu, không phải bằng chứng.
Chọn nhãn theo phần thực sự sai, không theo từ loại xuất hiện trong span. Thông tin không có căn cứ không tự động trở thành mâu thuẫn chỉ vì chứa số, tên thực thể hoặc thuật ngữ.

BƯỚC 1: Span có bị CONTEXT bác bỏ hay không?
- MÂU THUẪN: CONTEXT có thông tin tương ứng nhưng khác với span.
- KHÔNG CÓ CĂN CỨ: CONTEXT không đủ thông tin để xác minh span.

BƯỚC 2A: Nếu KHÔNG CÓ CĂN CỨ, chọn một nhãn
1. unsupported_reference: viện dẫn hoặc quy thông tin cho nguồn, văn bản, điều khoản mà CONTEXT không đủ căn cứ xác minh việc viện dẫn đó.
2. unsupported_addition: thông tin khác không có căn cứ, gồm thêm chi tiết, thu hẹp phạm vi, suy diễn.
3. other: không khớp hai nhãn trên.

BƯỚC 2B: Nếu MÂU THUẪN, xem phần khác biệt nhỏ nhất giữa span và CONTEXT, rồi chọn nhãn ĐẦU TIÊN phù hợp theo thứ tự:
1. incorrect_reference: sai số hiệu, điều/khoản, tên văn bản hoặc nguồn mà CONTEXT xác định được.
2. sino_vietnamese_term_confusion: nhầm lẫn giữa hai thuật ngữ Hán-Việt hoặc chuyên ngành dễ nhầm trong cùng hệ khái niệm nhưng khác nghĩa (ví dụ: tạm giữ / tạm giam). Không dùng chỉ vì span có từ chuyên ngành.
3. numerical_distortion: sai số lượng, giá trị, tỷ lệ, đơn vị, so sánh định lượng. Không gồm thời điểm, thời hạn hoặc thời lượng.
4. circumstance_error: sai thời điểm, thời hạn, thời lượng, địa điểm sự kiện, điều kiện, phạm vi áp dụng hoặc mức độ bắt buộc.
5. entity_substitution: thay tên hoặc danh tính người, tổ chức, cơ quan; gồm sai danh xưng, giới tính, vai vế hoặc đại từ của một biểu thức được viết rõ. Cũng gồm gán hành động cho sai chủ thể khi CONTEXT lược chủ ngữ nhưng ngữ cảnh xác định được chủ thể đúng; không đoán chủ thể khi quy chiếu còn mơ hồ.
6. predicate_error: sai hành động, trạng thái, quan hệ, xu hướng, phủ định, nhân quả.
7. other: có mâu thuẫn nhưng không khớp bước nào.

QUY TẮC XUẤT
Chỉ xuất một JSON duy nhất, không markdown, không giải thích:
{"diagnosis":"<một trong các nhãn trên>"}
"""


def user_prompt(example: DiagnosisExample) -> str:
    return (f"SPAN:\n{example.span}\n\nANSWER:\n{example.marked_answer()}\n\n"
            f"QUESTION:\n{example.question}\n\nCONTEXT:\n{example.context}")


def target_json(example: DiagnosisExample) -> str:
    return json.dumps({"diagnosis": example.label}, ensure_ascii=False, separators=(",", ":"))


def build_messages(
    example: DiagnosisExample, demonstrations: Sequence[DiagnosisExample] = (),
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for demonstration in demonstrations:
        if demonstration.source_example_id == example.source_example_id:
            raise ValueError("A few-shot demonstration cannot come from the query's source QA")
        messages.extend([
            {"role": "user", "content": user_prompt(demonstration)},
            {"role": "assistant", "content": target_json(demonstration)},
        ])
    messages.append({"role": "user", "content": user_prompt(example)})
    return messages
