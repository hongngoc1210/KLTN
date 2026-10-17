# Utility scripts

Chạy các script từ root `llm_judge` bằng `python -m`.

## `judge_pipeline/`

- `filter_pending_v4`: tạo tập chờ Judge.
- `route_judge2_v4`: phân tuyến mẫu sang Judge 2.
- `retry_errors_v4`: chạy lại lỗi Judge.

## `human_review/`

- `filter_human_review_v4`: tạo auto pass và human review.
- `sample_human_review_v4`: lấy một batch review.
- `split_human_review_batches_v4`: chia nhiều batch review.
- `import_human_reviews_v4`: nhập nhãn người gán.
- `export_human_review_subsets_v4`: xuất các tập con để phân tích.

## `dataset/`

- `merge_judge_data`: merge dữ liệu ban đầu.
- `filter_low_quality_v4`: loại mẫu chất lượng thấp.
- `merge_human_review_v4`: merge các batch human review.
- `merge_review_and_pass_v4`: merge Judge 1, auto pass và human review.
- `select_hard_samples_v4`: chọn hard hallucination.
- `select_no_hallucination_v4`: lấy no-hallucination cân bằng.
