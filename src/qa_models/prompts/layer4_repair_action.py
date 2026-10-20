"""Action-only prompting, using the annotation policy without gold attributes."""

from __future__ import annotations

import json

from qa_models.prompts.layer3_diagnosis import user_prompt


PROMPT_VERSION = "layer4-gold-span-repair-action-v1"

SYSTEM_PROMPT = """Bạn chọn thao tác sửa cho đúng một span hallucination tiếng Việt.
Span được đánh dấu [SPAN]...[/SPAN] trong ANSWER. Đối chiếu QUESTION và CONTEXT,
xét cả mệnh đề/claim chứa span, rồi chọn đúng MỘT trong bốn repair_action:
- replace_span: claim mâu thuẫn với CONTEXT và chỉ MỘT cụm liền mạch sai, sửa
  cục bộ được bằng thông tin đúng có trong CONTEXT. Cũng dùng khi claim không
  có căn cứ do quá cụ thể, mà CONTEXT có cách nói tổng quát tương ứng để thay;
  không tự tạo thông tin thay thế.
- rewrite_claim: claim mâu thuẫn và cần sửa từ hai chỗ không liền nhau trở lên,
  hoặc phải đổi quan hệ/cấu trúc câu; sửa một cụm liền mạch là không đủ.
- delete_span: claim không có căn cứ; bỏ phần thừa (chi tiết/cụm/mệnh đề phụ)
  thì phần còn lại vẫn có nghĩa và được CONTEXT hỗ trợ.
- delete_claim: claim không có căn cứ; bỏ phần thừa thì không còn gì được
  CONTEXT hỗ trợ. Không xóa những ý được hỗ trợ trong cùng câu.
Tự xác định claim mâu thuẫn hay không có căn cứ từ CONTEXT; không có gold
diagnosis/evidence/loại hallucination. Khi mâu thuẫn, ưu tiên replace_span nếu
sửa cục bộ an toàn, nếu không thì rewrite_claim; không xóa để che mâu thuẫn.

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
