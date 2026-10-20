"""Action-only prompting, using the annotation policy without gold attributes."""

from __future__ import annotations

import json

from qa_models.prompts.layer3_diagnosis import user_prompt


PROMPT_VERSION = "layer4-gold-span-repair-action-v1"
SYSTEM_PROMPT = """Bạn chọn thao tác sửa cho đúng một span hallucination tiếng Việt.
Span được đánh dấu [SPAN]...[/SPAN] trong ANSWER. Đối chiếu QUESTION và CONTEXT,
xét cả mệnh đề chứa span, rồi chọn đúng MỘT trong bốn repair_action:
- replace_span: thay cụm sai liên tục bằng thông tin đúng có trong CONTEXT.
  Với chi tiết quá cụ thể không được hỗ trợ, chỉ dùng khi CONTEXT có cách nói
  tổng quát tương ứng để thay thế; không tự tạo thông tin thay thế.
- delete_span: xóa chi tiết/cụm/mệnh đề phụ không có căn cứ, trong khi phần
  được hỗ trợ còn lại của claim vẫn giữ được.
- delete_claim: xóa toàn bộ claim không có căn cứ; không xóa những ý được
  CONTEXT hỗ trợ trong cùng câu. Chỉ cần xóa mệnh đề phụ thì dùng delete_span.
- rewrite_claim: viết lại cấu trúc claim có mâu thuẫn khi sửa một cụm liên tục
  không đủ, ví dụ nhiều vị trí không liên tục hoặc quan hệ/cấu trúc phải đổi.
Tự xác định span mâu thuẫn hay không có căn cứ từ CONTEXT; không được cung cấp
gold diagnosis/evidence/loại hallucination. Khi mâu thuẫn, ưu tiên replace_span
nếu sửa cục bộ an toàn, nếu không thì rewrite_claim; không xóa để che mâu thuẫn.
Khi không có căn cứ, chọn delete_span/delete_claim hoặc ngoại lệ replace_span
nêu trên. Không chọn delete_claim chỉ vì QUESTION thiếu thông tin.
Chỉ dự đoán thao tác: không xuất diagnosis, evidence, replacement, target_span
hay câu đã sửa. Không tìm span khác. Không giải thích hoặc markdown.
Chỉ xuất JSON {"repair_action":"<một trong bốn nhãn>"}.
"""


def target_json(example) -> str:
    return json.dumps({"repair_action": example.label}, ensure_ascii=False, separators=(",", ":"))


def build_messages(example, demonstrations=()) -> list[dict[str, str]]:
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
