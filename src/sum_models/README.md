# SUM models

Đã chuyển từ `hallucination_experiments` sang `KLTN/src/sum_models`.
Giao diện chạy thống nhất với `qa_models`: cấu hình TOML theo layer,
`main.py --config ... --stage train|evaluate|all`, và notebook Colab theo từng model.

## Bắt đầu train layer 2 trên Colab

Mở [KLTN_sum_models_layer2.ipynb](notebooks/KLTN_sum_models_layer2.ipynb).
Notebook theo trình tự của `KLTN_qa_models_layer3.ipynb` bạn cung cấp:
mount Drive → clone KLTN → cài thư viện → HF token nếu cần → cấu hình từng model → chạy.

1. Upload thư mục `src/sum_models/data/` lên
   `MyDrive/KLTN/sum_models/data/` (cần `train.json`, `dev.json`, `test.json`).
2. Khi code mới chưa có trên GitHub, upload
   `artifacts/bundles/sum_models_source.zip` lên
   `MyDrive/KLTN/sum_models/sum_models_source.zip`.
   Notebook sẽ nạp code này vào repo vừa clone. Nếu dùng code mới đã push lên GitHub,
   đặt `SOURCE_ZIP` trỏ tới đường dẫn không tồn tại để dùng code từ repo.
3. Bật GPU; kiểm tra `WORK_DIR`, `RUN_MODELS`, `ENVIRONMENT`, `RUN_TAG`.
   Mặc định `RUN_MODELS = ["xlmr_base"]`, `ENVIRONMENT = "encoders"`.
4. Chạy các cell từ trên xuống. Chỉ các model trong `RUN_MODELS` được chạy.
   Cell của từng model cho phép sửa hyperparameter trước khi ghi TOML.

Layer 2 độc lập với layer 1. `--stage all` tự kiểm tra model trên hai mẫu train,
train, predict và evaluate. Smoke check được tái sử dụng nếu đúng run identity.
Đối với encoder, lần lượt train **hai model**: BIO span và evidence có điều kiện
theo span; hai model không được giữ đồng thời trên GPU.

Mặc định đánh giá validation. Sau khi chốt hyperparameter, đổi `EVAL_SPLIT = "test"`
và `STAGE = "evaluate"` để đánh giá checkpoint đã train.

## Chạy local

Từ `KLTN/src/sum_models`:

```bash
python -m pip install -r requirements.txt
python main.py validate-config --config configs/layer2/xlmr_base.toml
python main.py inspect-data --config configs/layer2/xlmr_base.toml
python main.py --config configs/layer2/xlmr_base.toml --stage all
```

Chấp nhận cả `--config` và `--config-file`. Từ `KLTN/src` có thể chạy
`python -m sum_models --config sum_models/configs/layer2/xlmr_base.toml --stage all`.

## Cấu trúc

```text
sum_models/
  main.py
  configs/schema.py
  configs/layer1/*.toml
  configs/layer2/*.toml
  configs/layer3/*.toml
  models/                 # encoder và generative runtime
  prompts/                # prompt và lựa chọn few-shot
  src/                    # config, dữ liệu, pipeline, metric, checkpoint
  scripts/                # export dữ liệu, tạo config/notebook, đóng gói code
  notebooks/KLTN_sum_models_layer{1,2,3}.ipynb
  tests/
  data/                   # train.json, dev.json, test.json; không đưa lên Git
  outputs/                # kết quả mới; không đưa lên Git
  artifacts/              # giữ nguyên dữ liệu, bundle và run cũ; không đưa lên Git
  legacy/                 # config JSON, launcher/notebook/tài liệu cũ; lưu local
```

Có 30 TOML, 10 cấu hình mỗi layer. Layer 2 gồm XLM-R-base/large, mDeBERTa,
mmBERT, ViT5, Qwen zero/few-shot/QLoRA và Gemma zero/few-shot.
PhoBERT chỉ có ở layer 1/3.

## Format cấu hình

Các section giống QA: `[data]`, `[model]`, `[model.params]`, `[training]`,
`[training.params]`, `[evaluation]`. Xem
[configs/layer2/xlmr_base.toml](configs/layer2/xlmr_base.toml).
Đường dẫn tương đối tính từ `KLTN/src`, giống `qa_models`; trên Colab dùng đường dẫn tuyệt đối.
Các option không được hỗ trợ sẽ báo lỗi, không bị bỏ qua.

SUM giữ thuật toán xử lý văn bản dài qua nhiều cửa sổ. Vì vậy
`train_batch_size`, `eval_batch_size`, `evaluation.batch_size` phải là **1**;
`gradient_accumulation_steps = 16` tương ứng tích lũy 16 response mỗi optimizer step.
Không chép batch=8/16 từ notebook QA sang.

`training.params` dùng `warmup_ratio`, `checkpoint_steps`, `precision`
(`auto/fp32/fp16/bf16`), `early_stopping_patience`, `early_stopping_threshold`,
`resume_from_checkpoint = "auto"`. Checkpoint giữ latest và best.

XLM-R/mDeBERTa/mmBERT/ViT5 dùng `requirements-encoders.txt`.
Qwen/Gemma dùng **runtime riêng** với `requirements-llm.txt`;
đổi `ENVIRONMENT = "llm"`. Các version thư viện được giữ theo engine SUM cũ.

## Dữ liệu

Bản export giữ nguyên toàn bộ 14.187 response: train 11.357, dev 1.425, test 1.405;
10.988 vùng lỗi, 62 vùng đa nhãn, 10 diagnosis.
Các trường native là `id`, `document_id`, `context`, `question`, `response`,
`label`, `regions`, `source_atoms`, `split`. Trong `regions` giữ chính xác offset,
diagnosis đa nhãn, evidence và điều kiện chấm evidence.

Format chạy/cấu hình được thống nhất với QA; annotation SUM vẫn được giữ đầy đủ.
Bản `hallucination_dataset/*.json` rút gọn không có đủ thông tin này nên không
dùng thay cho `sum_models/data/*.json`. Không thay đổi nhãn hoặc ranh giới span.

Có thể tái tạo bản export từ dữ liệu cũ đã chuyển vào package:

```bash
python scripts/export_data.py
```

Trước train, kiểm tra trùng ID/document giữa các split và offset không khớp.
Dữ liệu dùng cho mỗi run được đóng băng tại `<output_dir>/dataset/`,
có hash kiểm tra khi chạy lại.

## Output và resume

`training.output_dir` chứa:

- `components/localization/`, `components/evidence/`: checkpoint riêng của layer 2 encoder.
- `run_manifest.json`: khóa config, code, dataset, model revision và môi trường.
- `predictions/<split>/predictions.jsonl`: prediction có thể tiếp tục khi gián đoạn.
- `evaluations/<split>/metrics.json`: metric theo split.
- `metrics.json`: metric của lần evaluate gần nhất, tiện dùng tương tự QA.

Chạy lại cùng config/output để resume; đổi hyperparameter, code hoặc dữ liệu thì
đổi `output_dir`/`RUN_TAG`. Các checkpoint/bundle cũ vẫn được giữ trong `artifacts/`,
nhưng không resume chúng bằng code đã đổi cấu trúc vì hash implementation đã đổi.

## Kiểm tra và tái tạo

Từ `KLTN/src`:

```bash
python -m unittest discover -s sum_models/tests -t . -v
```

Từ `KLTN/src/sum_models`:

```bash
python scripts/build_configs.py
python scripts/build_notebooks.py
python scripts/package_source.py
```

Hai lệnh build ghi lại template mặc định; giữ config tự chỉnh ở vị trí riêng
(notebook lưu trên Drive). ZIP source không chứa dataset, checkpoint, token hay môi trường Python.
Các test CPU kiểm tra contract, metric và checkpoint; không thay thế việc chạy train GPU thực tế.
