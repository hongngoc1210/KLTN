# Cấu trúc dự án

- `main.py`, `main_v4.py`, `main_judge2_v4_openai_batch.py`: các entrypoint chính.
- `src/`: logic Judge và validation dùng chung.
- `scripts/judge_pipeline/`: chuẩn bị, phân tuyến và retry Judge.
- `scripts/human_review/`: tạo batch, nhập và phân tích human review.
- `scripts/dataset/`: merge, lọc và chọn dataset cuối.
- `prompts/`: toàn bộ prompt và các version; được giữ nguyên.
- `configs/`: cấu hình chạy Judge.
- `tests/`: kiểm thử.
- `notebooks/`: notebook và dữ liệu demo.
- `data/output/detectors/`: kết quả Qwen/XLM-R theo model và số nhãn.
- `data/output/human_review/`: kết quả human review và dataset cuối.
- `archive/legacy/`: bản cũ được giữ để đối chiếu.

Xem lệnh chạy utility tại `scripts/README.md`.
