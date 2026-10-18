"""Strict JSON prompt for independent layer-2 extraction."""

from __future__ import annotations

from qa_models.src.records import Example


PROMPT_VERSION = "layer2-zero-shot-v2-evidence-policy"

# This policy is shared by decoder-only LLMs and ViT5 so that all layer-2
# experiments use the same definition of span and evidence.
EXTRACTION_RULES = """Bạn là hệ thống định vị hallucination trong ANSWER dựa trên CONTEXT.

Quy tắc bắt buộc:
- Đối chiếu từng thông tin trong ANSWER với CONTEXT. Nếu tất cả đều được hỗ trợ,
  trả đúng {\"spans\":[]}.
- span là đoạn sai hoặc không được hỗ trợ ngắn nhất nhưng vẫn đủ nghĩa. Chép
  nguyên văn và liên tục từ ANSWER; không lấy cả câu nếu chỉ một cụm bị sai.
- Nếu có nhiều lỗi, trả từng span riêng biệt. Không lấy nội dung QUESTION làm span.
- Khi span mâu thuẫn với CONTEXT, evidence phải là câu hoàn chỉnh ngắn nhất trực
  tiếp chứng minh mâu thuẫn. Nếu evidence nằm trong danh sách, lấy trọn mục danh
  sách. Chỉ lấy nhiều câu khi một câu chưa đủ chứng minh.
- Khi span không có thông tin hỗ trợ trong CONTEXT, evidence phải là null.
- Ngoại lệ: nếu ANSWER nêu một chi tiết quá cụ thể trong khi CONTEXT chỉ nêu trực
  tiếp một thông tin tổng quát dùng để thay thế chi tiết đó, evidence có thể là
  cụm hoặc mệnh đề tổng quát ngắn nhất; trường hợp này không bắt buộc lấy cả câu.
- Không dùng một câu chỉ cùng chủ đề làm evidence. Không tự tạo evidence như
  \"không có thông tin\".
- Mọi span phải là substring chính xác của ANSWER; mọi evidence khác null phải là
  substring chính xác của CONTEXT.
- Chỉ xuất đúng một JSON trên một dòng, không markdown, không giải thích, không
  thêm khóa: {\"spans\":[{\"span\":\"...\",\"evidence\":\"...\"}]}.
"""

SYSTEM_PROMPT = EXTRACTION_RULES

# ViT5 has a much smaller input budget than the decoder-only LLMs. Keep the
# same semantics without spending most of its 512-token window on instructions.
COMPACT_EXTRACTION_RULES = (
    "Tìm các span hallucination ngắn nhất, nguyên văn trong ANSWER và xuất JSON. "
    "Nếu span mâu thuẫn, evidence là câu hoàn chỉnh ngắn nhất hoặc trọn mục danh "
    "sách trong CONTEXT. Nếu span không được hỗ trợ, evidence là null; chỉ dùng "
    "cụm tổng quát ngắn nhất khi nó trực tiếp thay một chi tiết quá cụ thể. "
    "Không có hallucination thì xuất {\"spans\":[]}. Không giải thích."
)


def build_user_prompt(example: Example) -> str:
    return (
        f"CONTEXT:\n{example.context}\n\n"
        f"QUESTION:\n{example.question}\n\n"
        f"ANSWER:\n{example.answer}"
    )


def build_messages(example: Example) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(example)},
    ]
