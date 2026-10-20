# Layer 4: Repair-action cho từng span

Tầng riêng sau Layer 3 Diagnosis, không gộp nhãn action vào diagnosis và
**chưa sinh câu trả lời đã sửa**. Bốn nhãn theo dataset:

| ID | Action | Ý nghĩa |
|---|---|---|
| 0 | `replace_span` | Thay đoạn sai bằng thông tin được context hỗ trợ |
| 1 | `delete_span` | Xóa chi tiết không có căn cứ, giữ phần còn lại có căn cứ |
| 2 | `delete_claim` | Xóa toàn bộ claim không có căn cứ |
| 3 | `rewrite_claim` | Viết lại cấu trúc/nhiều vị trí, không đủ bằng thay một cụm |

## Phạm vi

Đánh giá **độc lập trên gold span**, như Layer 3. Mỗi annotation trong
`span_label` tạo một mẫu `<qa_id>::span:<index>`, target lấy từ `repair_action`.
Giữ nguyên split nguồn và vị trí annotation. Mẫu không có span không tạo mẫu
action. Nhãn thiếu/sai hoặc span không grounded gây lỗi, không bị bỏ qua.

Input: question, context, answer và span được đánh dấu. Không đưa gold diagnosis,
gold evidence, gold span type/response label vào model. Không cần chạy Layer
1/2/3 trước. **Chưa nối prediction giữa các tầng/chưa đánh giá end-to-end**;
không diễn giải điểm gold-span thành điểm E2E.

Encoder tái sử dụng tokenization/train loop của Layer 3 nhưng có **head 4 lớp,
mapping và artifacts riêng**. Token budget giữ toàn bộ span và markers, cắt
answer window/question/context như Layer 3. PhoBERT cần VnCoreNLP.

LLM dùng prompt riêng `qa_models/prompts/layer4_repair_action.py`, sinh greedy:

```json
{"repair_action":"replace_span"}
```

Few-shot chọn 1 mẫu train cho mỗi action (4 demonstrations), không lấy dev/test
và từ chối query trùng source QA với demo. Thiếu lớp hoặc prompt quá budget sẽ
báo lỗi, không âm thầm bỏ lớp/cắt span. Qwen QLoRA là completion-only SFT;
validation generate chỉ nhận prompt, không nhận gold completion.

## Chạy

10 config trong `qa_models/configs/layer4/`: 5 encoders PhoBERT, XLM-R-base/large,
mDeBERTa, mmBERT; Qwen zero/few-shot/QLoRA; Gemma zero/few-shot.

Từ `KLTN/src`:

```bash
# Train + chấm test.
python qa_models/main.py \
  --config-file qa_models/configs/layer4/mmbert_base.toml --stage all

# Chấm model đã lưu ở đúng output_dir/artifacts, không train lại.
python qa_models/main.py \
  --config-file qa_models/configs/layer4/mmbert_base.toml --stage evaluate

# Prompting, không train.
python qa_models/main.py \
  --config-file qa_models/configs/layer4/qwen35_4b_zero_shot.toml --stage evaluate
python qa_models/main.py \
  --config-file qa_models/configs/layer4/gemma4_e4b_few_shot.toml --stage evaluate

# SFT/QLoRA cần CUDA + PEFT/bitsandbytes.
python qa_models/main.py \
  --config-file qa_models/configs/layer4/qwen35_4b_qlora.toml --stage all

python -m unittest qa_models.tests.test_layer4 -v
```

Tự chỉnh paths/hyperparameters trong config. Output riêng:
`qa_models/outputs/layer4/<model>/{artifacts,checkpoints,predictions.json,metrics.json}`.
Không resume hoặc tải weights/head fine-tuned Layer 1/2/3 cho task này.

Predictions lưu source QA, span index/offset, `repair_action`,
`gold_repair_action`, scores/raw output và parse-error flag. Trường `label`
được giữ để tương thích record classification.

## Metric và early stopping

- Accuracy; macro precision/recall/F1; weighted F1; per-class P/R/F1/support.
- Macro-F1 trung bình **cố định 4 lớp**, zero-division=0, kể cả lớp vắng ở split.
- Confusion matrix có cột invalid. JSON/schema/action không hợp lệ giữ trong
  mẫu số, tính sai, không fallback sang action hợp lệ.
- `macro_f1_present_classes` là metric phụ cho những lớp có gold.
- **Early stop/checkpoint:** `eval_macro_f1` action 4 lớp trên dev, maximize;
  evaluate/save mỗi epoch, patience 2, threshold 0.0001 mặc định. Đây không phải
  F1 diagnosis/extraction. Zero/few-shot không có early stopping.

Full split hiện có:

| Action | Train | Dev | Test |
|---|---:|---:|---:|
| replace_span | 4.046 | 503 | 508 |
| delete_span | 3.575 | 449 | 449 |
| delete_claim | 164 | 21 | 14 |
| rewrite_claim | 8 | 0 | 2 |

Do dev không có rewrite_claim, perfect trên ba lớp có mặt cho Macro-F1 cố định
4 lớp = **0.75**, không phải 1.0. Không đổi taxonomy hoặc đưa test vào train
để tăng điểm. Hai lớp phổ biến áp đảo nên phải xem per-class F1/support,
không kết luận chỉ từ accuracy.

Chưa sinh replacement/target_span/repaired_answer nên không tính Faithfulness
hay Post-repair Verification Rate. Kiểm thử model nhỏ chỉ xác minh code,
không thay thế fine-tuning đầy đủ trên GPU.
