# Hệ thống đánh giá hallucination ba tầng

Khung code hiện đã triển khai tầng 1 và chế độ đánh giá độc lập của tầng 2.
Kiến trúc đích gồm ba tầng và hai chế độ đánh giá. Quyết định chia sẻ backbone,
tách task head và cách đánh giá end-to-end được mô tả trong
[ARCHITECTURE.md](ARCHITECTURE.md).

Quy tắc EM/IoU, ý nghĩa từng metric Layer 2 và tiêu chí early stopping của từng
họ model: [Layer 2 metrics](tasks/layer2/METRICS.md).

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
- `prompts/layer1_zero_shot.py`: prompt dùng chung, có version, cho Qwen/Gemma.
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

Với Qwen, hai phương pháp được tách hoàn toàn: zero-shot dùng
`models/qwen35_4b.py` + `qwen35_4b_zero_shot.toml`, còn SFT bằng QLoRA dùng
`models/qwen35_4b_qlora.py` + `qwen35_4b_qlora.toml`. Hai cấu hình lưu artifact
vào hai thư mục khác nhau.

## Lệnh chạy

Từ thư mục gốc `src`:

```bash
# Cách chạy chính: config quyết định model/checkpoint/adapter được sử dụng.
python qa_models/main.py \
  --config-file qa_models/configs/layer1/phobert_base_v2.toml \
  --stage all

# XLM-R-base tầng 1 (không cần VnCoreNLP).
python qa_models/main.py \
  --config-file qa_models/configs/layer1/xlmr_base.toml \
  --stage all

# XLM-R-large tầng 1; config mặc định dùng fp16 và gradient checkpointing.
python qa_models/main.py \
  --config-file qa_models/configs/layer1/xlmr_large.toml \
  --stage all

# mDeBERTa-v3-base và mmBERT-base: fine-tune bốn lớp.
python qa_models/main.py \
  --config-file qa_models/configs/layer1/mdeberta_v3_base.toml \
  --stage all
python qa_models/main.py \
  --config-file qa_models/configs/layer1/mmbert_base.toml \
  --stage all

# Qwen3.5-4B và Gemma-4-E4B-it: zero-shot, không train.
python qa_models/main.py \
  --config-file qa_models/configs/layer1/qwen35_4b_zero_shot.toml \
  --stage evaluate
python qa_models/main.py \
  --config-file qa_models/configs/layer1/gemma4_e4b_zero_shot.toml \
  --stage evaluate

# Few-shot cân bằng: 1 demonstration từ train cho mỗi nhãn, không train model.
python qa_models/main.py \
  --config-file qa_models/configs/layer1/qwen35_4b_few_shot.toml \
  --stage evaluate
python qa_models/main.py \
  --config-file qa_models/configs/layer1/gemma4_e4b_few_shot.toml \
  --stage evaluate

# Qwen3.5-4B: supervised fine-tuning thật bằng QLoRA, sau đó đánh giá test.
python qa_models/main.py \
  --config-file qa_models/configs/layer1/qwen35_4b_qlora.toml \
  --stage all

# Chỉ đánh giá lại LoRA artifact đã train và lưu.
python qa_models/main.py \
  --config-file qa_models/configs/layer1/qwen35_4b_qlora.toml \
  --stage evaluate

# Tầng 2 độc lập: model nhận question + answer + context GỐC từ dataset.
# Các encoder dùng BIO 5 nhãn để trích xuất span và evidence.
python qa_models/main.py \
  --config-file qa_models/configs/layer2/xlmr_base.toml \
  --stage all

# Thay xlmr_base bằng xlmr_large hoặc mmbert_base.
# ViT5 fine-tune sinh JSON.
python qa_models/main.py \
  --config-file qa_models/configs/layer2/vit5_base.toml \
  --stage all

# Qwen/Gemma prompting tầng 2: chỉ evaluate, không train.
python qa_models/main.py \
  --config-file qa_models/configs/layer2/qwen35_4b_zero_shot.toml \
  --stage evaluate
python qa_models/main.py \
  --config-file qa_models/configs/layer2/gemma4_e4b_zero_shot.toml \
  --stage evaluate

# Few-shot tầng 2: tự chọn 5 kiểu demonstration từ train, không cập nhật trọng số.
python qa_models/main.py \
  --config-file qa_models/configs/layer2/qwen35_4b_few_shot.toml \
  --stage evaluate
python qa_models/main.py \
  --config-file qa_models/configs/layer2/gemma4_e4b_few_shot.toml \
  --stage evaluate

# Qwen tầng 2 SFT bằng QLoRA rồi đánh giá test.
python qa_models/main.py \
  --config-file qa_models/configs/layer2/qwen35_4b_qlora.toml \
  --stage all

# Chỉ train hoặc chỉ evaluate artifact đã lưu.
python qa_models/main.py \
  --config-file qa_models/configs/layer1/phobert_base_v2.toml \
  --stage train
python qa_models/main.py \
  --config-file qa_models/configs/layer1/phobert_base_v2.toml \
  --stage evaluate

# Ablation: giữ nguyên question/answer/label nhưng tráo context trong test.
# Kết quả được lưu riêng thành metrics_shuffled_context.json và
# predictions_shuffled_context.json.
python qa_models/main.py \
  --config-file qa_models/configs/layer1/phobert_base_v2.toml \
  --stage evaluate \
  --shuffle-context

# Chạy toàn bộ kiểm tra shortcut/context trong một lần và tạo
# ablation_summary.json để so sánh tự động.
python qa_models/main.py \
  --config-file qa_models/configs/layer1/phobert_base_v2.toml \
  --stage evaluate \
  --ablation-suite

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

