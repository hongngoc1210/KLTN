"""Zero-shot prompt for four-class layer-1 hallucination detection."""

from __future__ import annotations

from qa_models.src.records import Example


PROMPT_VERSION = "layer1_zero_shot_v1"

SYSTEM_PROMPT = """Bạn là bộ phân loại hallucination tiếng Việt.

Nhiệm vụ: đối chiếu CÂU TRẢ LỜI với NGỮ CẢNH và CÂU HỎI, sau đó chọn đúng
một trong bốn nhãn sau:

1. no_hallucination
   Mọi claim trong câu trả lời đều được ngữ cảnh hỗ trợ. Câu trả lời được phép
   diễn đạt lại hoặc tổng hợp thông tin nhưng không làm thay đổi ý nghĩa.

2. intrinsic_hallucination
   Có ít nhất một claim mâu thuẫn với thông tin được nêu trong ngữ cảnh, chẳng
   hạn sai người, số liệu, thời gian, thuộc tính, quan hệ hoặc phủ định.

3. extrinsic_hallucination
   Có ít nhất một claim không được ngữ cảnh hỗ trợ; ngữ cảnh không xác nhận và
   cũng không cung cấp đủ evidence cho claim đó.

4. mixed_hallucination
   Câu trả lời đồng thời chứa ít nhất một claim intrinsic và ít nhất một claim
   extrinsic.

Quy tắc đầu ra:
- Chỉ xuất đúng một JSON trên một dòng.
- Không giải thích, không markdown và không thêm trường khác.
- Thay LABEL bằng đúng một trong bốn nhãn trên.

{"label":"LABEL"}
"""


def build_user_prompt(example: Example) -> str:
    """Render one dataset example without exposing its gold label."""

    return (
        f"NGỮ CẢNH:\n{example.context}\n\n"
        f"CÂU HỎI:\n{example.question}\n\n"
        f"CÂU TRẢ LỜI CẦN KIỂM TRA:\n{example.answer}"
    )


def build_messages(example: Example) -> list[dict[str, str]]:
    """Return standard chat messages shared by Qwen and Gemma."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(example)},
    ]

