# V4 — evidence-first + validation + verbalized confidence

V4 giữ rubric phân loại và sáu quality scores của V3. Phần thay đổi chỉ gồm:

- không đưa original label vào prompt;
- Python kiểm tra schema, exact evidence spans và tính lại actual label từ hai evidence arrays;
- model trả `alternative_label`, uncertainty có kiểm soát và confidence 0.50–1.00;
- chỉ dùng một API call cho mỗi lần thử thành công;
- mỗi QA có ID riêng dù nhiều QA dùng chung một `chunk_id`.

V4 không chứa policy chuyển tầng. Ngưỡng và Judge 2 sẽ được thiết kế sau khi có
output thực nghiệm và tập gold.

## Chạy

Từ thư mục project:

```bash
python -m pip install requests tqdm
read -rsp 'OpenRouter API key: ' OPENROUTER_API_KEY
export OPENROUTER_API_KEY
python main_v4.py --limit 3 --no-resume
```

Lệnh trên đọc `data/prepared/demo/judge_demo_30.json`, chạy ba QA đầu và ghi vào
`data/output/judge1/gemma_judge_v4_prompt_confidence.json`. Bỏ `--limit 3` để chạy hết.

Muốn chỉ định file riêng:

```bash
python main_v4.py \
  --input data/prepared/demo/judge_demo_30.json \
  --output data/output/judge1/gemma_judge_v4_test.json \
  --limit 3 \
  --no-resume
```

- Không có `--no-resume`: bỏ qua QA đã có `status="success"` trong output.
- Có `--no-resume`: chạy lại từ đầu và ghi đè checkpoint bằng kết quả mới.
- `--limit N`: lấy N QA đầu dataset, không phải N `chunk_id`.

## Evidence validation

Python kiểm tra:

- `answer_span` là chuỗi con nguyên văn của ANSWER;
- `context_evidence` của contradicted claim là chuỗi con nguyên văn của CONTEXT;
- span không trùng và không chồng lấn giữa hai loại lỗi;
- `proposed_label` đúng với bảng quyết định từ hai evidence arrays;
- scores là số nguyên 0–5 và tự tính lại `total_score`.

Validation này kiểm tra tính toàn vẹn cấu trúc và nguồn trích dẫn. Nó không thể tự
chứng minh quan hệ ngữ nghĩa giữa claim và evidence là đúng; việc đó cần đánh giá
trên tập gold đã rà thủ công.

## Confidence

Model tự trả confidence số từ 0.50 đến 1.00; giá trị lớn hơn 0.95 được phép.
Python kiểm tra range, làm tròn hai chữ số và ghi
`confidence_source="model_verbalized_uncalibrated"`.

Nếu có `alternative_label`, model phải khai báo ít nhất một uncertainty factor
và confidence không vượt quá 0.84. Sample invalid phải có confidence 0.50.

Đây là verbalized confidence chưa calibration, không phải xác suất đã được xác
nhận rằng nhãn đúng. Không dùng trực tiếp làm ngưỡng Judge 2 trước khi đo
accuracy và calibration trên tập gold người.

## Nhiều QA trong một chunk

Mỗi record luôn có `qa_id`. Nếu chưa có `candidate_id`/`id`, code tạo ID dạng
`chunk_id__qa1`, `chunk_id__qa2`, ...; `chunk_id` gốc vẫn được giữ. Vì vậy nhiều
QA chung một context không bị skip nhầm khi resume. Nếu dataset có thể thay đổi
thứ tự, nên tạo `candidate_id` duy nhất ngay từ bước sinh QA.

Thứ tự field output là: `qa_id`, `status`, metadata, context/question/answer,
`original_label`, `evaluation`, rồi usage rút gọn. Các field có thể suy ra hoặc
bị lặp như `proposed_label`, `has_hallucination`, `contradicted_spans`,
`unsupported_spans` và `confidence_source` không được chép vào từng record.

## Kiểm thử offline

```bash
python -m unittest discover -s tests -p 'test_v4_openrouter.py' -v
```

Các file chính: `main_v4.py`, `src/judge_v4.py`, `src/v4_validation.py`,
`configs/config_v4.py` và các prompt `prompts/*_v4*.txt`.
