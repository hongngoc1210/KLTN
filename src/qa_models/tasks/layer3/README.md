# Layer 3: diagnosis cho từng gold span

Đây là **đánh giá độc lập (gold-span/oracle)**, không phải kết quả end-to-end.
Mỗi annotation trong `span_label` tạo một mẫu với ID
`<qa_id>::span:<index>`. Mẫu không có span không tạo mẫu diagnosis. Không đổi
train/dev/test, không chia lại dữ liệu ở cấp span.

Đầu vào: question, context, answer và vị trí gold span được đánh dấu
`[SPAN]...[/SPAN]`. Gold diagnosis chỉ là nhãn giám sát; không đưa gold evidence,
nhãn response, nhãn contradicted/unsupported hay repair_action vào input.
Chưa triển khai chạy diagnosis trên predicted spans hoặc metric E2E; không
diễn giải điểm ở đây như chất lượng của toàn pipeline.

## Model và config

Tất cả config ở `qa_models/configs/layer3/`, outputs ở
`qa_models/outputs/layer3/<model>/`, không dùng checkpoint Layer 1/2.

| Config stem | Phương pháp |
|---|---|
| `phobert_base_v2` | VnCoreNLP + marked-span sequence classification |
| `xlmr_base`, `xlmr_large` | Marked-span sequence classification với context |
| `mmbert_base` | Marked-span sequence classification với context |
| `qwen35_4b_zero_shot`, `gemma4_e4b_zero_shot` | JSON diagnosis zero-shot |
| `qwen35_4b_few_shot`, `gemma4_e4b_few_shot` | 1 demonstration train cho mỗi diagnosis (10 mẫu) |
| `qwen35_4b_qlora` | Completion-only SFT bằng QLoRA; dev sinh tự do |

Encoder dùng classification head chuẩn trên biểu diễn sequence/CLS đã thấy
span markers và context, **không phải một span-pooling head riêng**. Token
budget giữ toàn bộ target span và markers, lấy cửa sổ answer quanh span,
tối đa 32 token question, và phần đầu context trong budget còn lại. Target
span quá dài sẽ báo lỗi thay vì bị cắt mất. PhoBERT word-segment từng phần
văn bản trước khi chèn markers; cần VnCoreNLP như Layer 1.

Generative models dùng cùng prompt/version tại
`qa_models/prompts/layer3_diagnosis.py`, sinh greedy:

```json
{"diagnosis":"entity_substitution"}
```

Few-shot chỉ chọn từ train, ưu tiên mẫu ngắn và cố định seed. Nếu không đủ
mẫu cho một lớp hoặc query trùng source QA với demonstration sẽ báo lỗi.
Không âm thầm cắt demonstrations/gold span khi prompt vượt token budget;
hãy tăng `max_input_length` hoặc chỉnh `shots_per_label` trong config.
QLoRA mask toàn bộ prompt bằng -100; validation generate chỉ nhận prompt,
không nhận gold completion. Cả dev và test dùng cùng parser/metric.

## Nhãn và metric

Thứ tự 10 lớp cố định tại `labels.py`. Thiếu/khác taxonomy diagnosis trong gold
hoặc span không nằm trong answer sẽ báo lỗi, không bỏ qua mẫu. Khi không có
offset, chọn lần xuất hiện NFC-equivalent đầu tiên như Layer 2.

- `accuracy`: tỷ lệ span có diagnosis đúng.
- `macro_precision`, `macro_recall`, `macro_f1`: trung bình **cố định 10 lớp**;
  lớp có mẫu số 0 được tính 0, kể cả lớp không có gold trong split.
- `weighted_f1`: F1 trọng số theo số gold span mỗi lớp.
- `per_class`: precision, recall, F1 và support; `confusion_matrix` có thêm
  cột `__invalid_prediction__` cho đầu ra không hợp lệ.
- `macro_f1_present_classes`: metric phụ, chỉ trung bình lớp có gold;
  **không dùng để early stop/chọn checkpoint**.
- `parse_error_rate`: đầu ra JSON/schema/diagnosis không hợp lệ; giữ trong
  mẫu số, không fallback sang nhãn `other`, không loại khỏi đánh giá.

**Early stopping/chọn checkpoint:** `eval_macro_f1` diagnosis (10 lớp), càng
cao càng tốt; evaluate/save mỗi epoch, tối đa 10 epoch, patience 2,
threshold 0.0001 mặc định. Đây không phải extraction Macro F1 của Layer 2.
Zero/few-shot không train và không có early stopping.

Full split hiện có 7.793/973/973 span train/dev/test. Nhãn
`implicit_subject_drift` chỉ có 1/0/1 mẫu. Vì dev thiếu lớp này, perfect dev
trên 9 lớp có mặt có Macro-F1 cố định 10 lớp là 0.9, không phải 1.0. Luôn xem
support và metric phụ khi đọc điểm. Không tự gộp nhãn hiếm hoặc đưa test vào
train. Dữ liệu demo 20/4/4 có thể không đủ 10 diagnosis cho few-shot.

## Giảm lệch nhãn khi fine-tune encoder

PhoBERT, XLM-R base/large và mmBERT hỗ trợ cùng cơ chế giảm lệch nhãn cho
encoder Layer 3 và Layer 4. Các config encoder dùng trong protocol cuối bật
`inverse_sqrt`; implementation vẫn hỗ trợ `"none"` để chạy ablation. Cơ chế
này không áp dụng cho prompt zero/few-shot.

Thêm vào bảng `[training.params]` hiện có trong config để bật **weighted loss**:

```toml
class_weighting = "inverse_sqrt"
class_weight_max_ratio = 10.0
sampling_strategy = "none"
```

Hoặc thử **weighted sampling** thay cho weighted loss:

```toml
class_weighting = "none"
sampling_strategy = "inverse_sqrt"
sampling_max_ratio = 10.0
```

Mỗi lựa chọn nhận `"none"`, `"inverse_sqrt"` (trọng số tỷ lệ nghịch căn số mẫu),
hoặc `"balanced"` (nghịch số mẫu, mạnh hơn). `max_ratio` giới hạn tỷ lệ trọng
số lớp hiếm/lớp nhiều mẫu, mặc định 10; lớp vắng trong train có trọng số 0.
Không cho bật hai cách cùng lúc để tránh bù imbalance hai lần.

Weighted loss dùng `mean(weight[label] * CE)` với trọng số chuẩn hóa có kỳ vọng
trên train bằng 1, nên vẫn có tác dụng ở batch size 1. Gradient accumulation
chia theo tổng số mẫu trong cửa sổ cập nhật, kể cả cửa sổ cuối không đủ batch.
Weighted sampling lấy mẫu có hoàn lại, vẫn giữ số lượt lấy mẫu mỗi epoch bằng
số train span; không đảm bảo mọi mẫu được thấy mỗi epoch. Seed + epoch cố định
cho phép tái lập/thực hiện lại epoch khi resume cùng cấu hình. Accelerate chia
chuỗi lấy mẫu giữa các process; sampler không tự chia lần thứ hai.

Tần suất chỉ tính từ train; không lấy từ dev/test, không sửa dữ liệu nguồn.
Eval loss vẫn là CE thường, metric và early stopping vẫn là dev Macro-F1
cố định 10 lớp. Train counts và trọng số được log và lưu trong
`artifacts/adapter_metadata.json` dưới khóa `imbalance`. Chỉ dùng chúng để
theo dõi lần train, không cần class weight khi inference.

Khi so sánh baseline/reweighting/sampling, tự đặt `output_dir` khác nhau và
không resume checkpoint của chế độ cũ. Điều chỉnh trên dev, không chọn theo
test. Nhãn có 1 mẫu train vẫn dễ bị học thuộc dù đã giới hạn trọng số.

## Chạy

Từ `KLTN/src`, giữ nguyên CLI chung:

```bash
# Train + chấm test gold spans, không cần chạy Layer 1/2 trước.
python qa_models/main.py \
  --config-file qa_models/configs/layer3/mmbert_base.toml --stage all

# Chấm lại model đã fine-tune và lưu trong đúng output_dir/artifacts.
python qa_models/main.py \
  --config-file qa_models/configs/layer3/mmbert_base.toml --stage evaluate

# Zero/few-shot không cần artifacts đã train.
python qa_models/main.py \
  --config-file qa_models/configs/layer3/qwen35_4b_zero_shot.toml --stage evaluate
python qa_models/main.py \
  --config-file qa_models/configs/layer3/gemma4_e4b_few_shot.toml --stage evaluate

# SFT/QLoRA cần CUDA + PEFT/bitsandbytes, tương tự Layer 1/2.
python qa_models/main.py \
  --config-file qa_models/configs/layer3/qwen35_4b_qlora.toml --stage all

# Kiểm thử offline bằng model nhỏ; không tải checkpoint thật.
python -m unittest qa_models.tests.test_layer3 -v
python -m unittest qa_models.tests.test_layer3_imbalance -v
```

Tự chỉnh paths/hyperparameters trong config nếu dùng split khác. Khi đổi
taxonomy, input policy hay task, dùng output namespace mới; không resume
checkpoint Layer 1/2. Predictions lưu source QA ID, span index/offset, gold
diagnosis, predicted label, scores/raw output và parse-error flag.

Các bài kiểm thử nhỏ xác minh đường train/predict/save/load và chống lộ nhãn,
không chứng minh chất lượng checkpoint thật. Chưa chạy fine-tuning đầy đủ
các backbone/GPU trong bước triển khai code này.
