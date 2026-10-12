# Hệ thống đánh giá hallucination ba tầng

Khung code hiện đã triển khai phần hạ tầng dùng được cho tầng 1; kiến trúc đích
bao gồm ba tầng và hai chế độ đánh giá. Quyết định chia sẻ backbone, tách task
head và cách đánh giá end-to-end được mô tả trong [ARCHITECTURE.md](ARCHITECTURE.md).

Chưa có checkpoint cụ thể nào được triển khai. Khi thêm một experiment tầng 1,
chỉ cần thêm:

1. `configs/<model_name>.toml` — checkpoint và hyperparameter;
2. `models/<model_name>.py` — adapter kế thừa `BaseModelAdapter`.

Không cần sửa runner, metrics hay registry trung tâm.

## Cài đặt môi trường

Yêu cầu Python 3.13 và Java 8+ cho VnCoreNLP. Cài các package đã được pin:

```bash
python -m pip install -r qa_models/requirements.txt
```

PhoBERT còn cần tải resource VnCoreNLP một lần vào đường dẫn được khai báo trong
config (`qa_models/resources/vncorenlp`).

## Nhãn chuẩn

Thứ tự class ID là cố định trong `src/labels.py`:

| ID | Label |
|---:|---|
| 0 | `no_hallucination` |
| 1 | `intrinsic_hallucination` |
| 2 | `extrinsic_hallucination` |
| 3 | `mixed_hallucination` |

Adapter có thể dùng `LABEL_TO_ID` và `ID_TO_LABEL`; không tự định nghĩa lại thứ
tự nhãn trong từng model.

## Flow chung

```text
train/dev/test JSON
        │
        ▼
load + validate + normalize label
        │
        ▼
model adapter (encoder classifier hoặc generative LLM)
        │
        ▼
Prediction(example_id, label, scores/raw_output)
        │
        ▼
accuracy + macro/weighted P/R/F1 + per-class + confusion matrix
        │
        ▼
outputs/<model>/metrics.json + predictions.json + artifacts/
```

Mỗi input gồm `content`, `question`, `answer`. Model phải dự đoán mức độ
hallucination của **answer so với content**, trong ngữ cảnh của question.

## Các file có trách nhiệm gì?

- `configs/schema.py`: schema config ổn định cho mọi model.
- `configs/_template.toml`: mẫu để tạo config mới.
- `models/base.py`: interface bắt buộc của adapter.
- `models/_template.py`: mẫu file model mới.
- `src/data.py`: đọc và kiểm tra JSON, duplicate ID, field và label.
- `src/pipeline.py`: điều phối train/evaluate, không chứa code model.
- `src/metrics.py`: metrics chung để so sánh công bằng.
- `main.py`: CLI duy nhất cho mọi experiment.

## Thêm model từng bước

Ví dụ thêm PhoBERT:

1. Copy `configs/_template.toml` thành `configs/phobert_base_v2.toml`.
2. Copy `models/_template.py` thành `models/phobert_base_v2.py`.
3. Đổi class thành `PhoBERTBaseV2Adapter` và config `adapter` thành
   `qa_models.models.phobert_base_v2:PhoBERTBaseV2Adapter`.
4. Implement `fit`, `predict`, `save`, `load` trong adapter. Code tokenizer,
   Transformers/PEFT chỉ nằm trong file này.
5. Chạy ba lệnh kiểm tra ở phần dưới.

Quy ước file đề xuất cho danh sách model:

| Config/model stem | Checkpoint | Family/method |
|---|---|---|
| `phobert_base_v2` | `vinai/phobert-base-v2` | encoder / fine-tune |
| `xlmr_base` | `FacebookAI/xlm-roberta-base` | encoder / fine-tune |
| `xlmr_large` | `FacebookAI/xlm-roberta-large` | encoder / fine-tune |
| `mdeberta_v3_base` | `microsoft/mdeberta-v3-base` | encoder / fine-tune |
| `mmbert_base` | `jhu-clsp/mmBERT-base` | encoder / fine-tune |
| `qwen35_4b` | `Qwen/Qwen3.5-4B` | generative / zero/few-shot, SFT/QLoRA |
| `gemma_4_e4b_it` | `google/gemma-4-E4B-it` | generative / zero/few-shot |

Với Qwen, nên dùng các config riêng theo phương pháp, ví dụ
`qwen35_4b_zero_shot.toml` và `qwen35_4b_qlora.toml`, nhưng cả hai có thể trỏ
tới cùng `models/qwen35_4b.py`.

## Lệnh chạy

Từ thư mục gốc `src`:

```bash
# Cách chạy chính: config quyết định model/checkpoint/adapter được sử dụng.
python qa_models/main.py \
  --config-file qa_models/configs/layer1/phobert_base_v2.toml \
  --stage all

# Chỉ train hoặc chỉ evaluate artifact đã lưu.
python qa_models/main.py \
  --config-file qa_models/configs/layer1/phobert_base_v2.toml \
  --stage train
python qa_models/main.py \
  --config-file qa_models/configs/layer1/phobert_base_v2.toml \
  --stage evaluate

# Kiểm tra schema và thống kê nhãn; chưa cần adapter hoạt động.
python -m qa_models.main inspect-data --config qa_models/configs/<model>.toml

# Kiểm tra import path và interface của adapter.
python -m qa_models.main validate-config --config qa_models/configs/<model>.toml

# Huấn luyện, đánh giá, hoặc cả hai.
python -m qa_models.main run --config qa_models/configs/<model>.toml --stage train
python -m qa_models.main run --config qa_models/configs/<model>.toml --stage evaluate
python -m qa_models.main run --config qa_models/configs/<model>.toml --stage all
```

`inspect-data` chỉ đọc config và dữ liệu. `validate-config` sẽ thất bại với file
template cho tới khi adapter thật được tạo — đây là chủ ý để phát hiện sai tên
module/class sớm.

Thống kê đầy đủ label, diagnosis và repair action giữa ba split:

```bash
python -m qa_models.scripts.analyze_splits

# Nếu cần lưu kết quả JSON để dùng trong báo cáo:
python -m qa_models.scripts.analyze_splits \
  --output qa_models/reports/split_statistics.json
```

## Thứ tự triển khai đề xuất

1. PhoBERT-base-v2: tạo baseline encoder đầu tiên và chốt preprocessing.
2. XLM-R-base: tái sử dụng pattern encoder để kiểm tra tính tổng quát.
3. mDeBERTa-v3-base và mmBERT-base.
4. XLM-R-large sau khi flow/memory đã ổn định.
5. Qwen zero-shot/few-shot, rồi mới SFT/QLoRA.
6. Gemma zero-shot/few-shot; dùng processor riêng của checkpoint.

Không nên trộn code prompt-based LLM vào adapter encoder. Hai họ model dùng chung
record đầu vào, output và metrics, nhưng giữ preprocessing/inference riêng.

