# Kiến trúc hệ thống bốn tầng

Hiện đã có independent/gold-span runners cho Layer 3 diagnosis và Layer 4
repair-action. Layer 4 là phân loại thao tác, chưa sinh câu trả lời đã sửa;
end-to-end vẫn là thiết kế đích, chưa phải runner được triển khai.

## Quyết định chính

`Model/checkpoint` và `task` là hai trục độc lập:

- backbone/checkpoint được khai báo một lần và có thể tái sử dụng;
- input builder, head, loss, output schema và metric phải tách theo từng tầng;
- mỗi lần fine-tune tạo artifact riêng, kể cả khi cùng dùng một checkpoint gốc.

Vì vậy không tạo một file khổng lồ `phobert.py` chứa toàn bộ pipeline, cũng không
copy toàn bộ code PhoBERT cho từng task. Cấu trúc đích:

```text
qa_models/
├── backbones/                 # cách load checkpoint/tokenizer/processor
│   ├── hf_encoder.py          # dùng chung PhoBERT, XLM-R, mDeBERTa, mmBERT
│   ├── vit5.py
│   ├── qwen35.py
│   └── gemma4.py
├── tasks/
│   ├── layer1_classification/
│   │   ├── adapter.py         # sequence classification / prompt classification
│   │   ├── records.py
│   │   └── metrics.py
│   ├── layer2_extraction/
│   │   ├── bio_adapter.py
│   │   ├── span_head_adapter.py
│   │   ├── generative_adapter.py
│   │   ├── records.py
│   │   └── metrics.py
│   ├── layer3_diagnosis/
│   │   ├── adapter.py         # span-aware classification / prompting
│   │   ├── records.py
│   │   └── metrics.py
│   └── layer4_repair_action/
│       ├── adapter.py         # span-aware classification / prompting
│       ├── records.py
│       └── metrics.py
├── pipelines/
│   ├── independent.py         # oracle input cho từng tầng
│   └── end_to_end.py          # output tầng trước → input tầng sau
├── evaluation/
│   ├── classification.py
│   ├── extraction.py
│   ├── faithfulness.py
│   └── repair.py
└── configs/
    ├── layer1/
    ├── layer2/
    ├── layer3/
    ├── layer4/
    └── pipelines/
```

Các file sẽ được tạo dần khi bắt đầu code task tương ứng; đây là layout đích,
không phải yêu cầu phải có file rỗng ngay từ đầu.

## Phần nào dùng chung?

### Encoder models

PhoBERT, XLM-R, mDeBERTa và mmBERT dùng chung loader/tokenization utilities khi
API checkpoint cho phép. Tuy nhiên artifact huấn luyện phải tách:

```text
artifacts/phobert_base_v2/layer1/  # sequence classification, 4 labels
artifacts/phobert_base_v2/layer3/  # span-aware diagnosis classification
artifacts/phobert_base_v2/layer4/  # four-action classification, separate head
```

Tầng 2 dùng token classification hoặc start/end head nên không thể dùng lại
classification head của tầng 1.

### Generative models

Qwen và Gemma dùng chung một runtime/processor cho các tầng, nhưng prompt,
JSON schema và parser tách theo task. Nếu SFT/QLoRA thì adapter weights cũng tách
theo tầng. Một base model có thể được load một lần khi chạy end-to-end để giảm
memory, sau đó lần lượt gắn prompt hoặc adapter tương ứng.

## Hai chế độ đánh giá

### 1. Independent/oracle

- Tầng 1 nhận `(content, question, answer)` và dự đoán label.
- Tầng 2 nhận trực tiếp `(content, question, answer)` gốc và `span_label` chỉ
  được dùng làm ground truth khi train/chấm điểm; tuyệt đối không dùng prediction
  tầng 1 làm input.
- Tầng 3 nhận gold span (và gold evidence nếu thiết kế model cần evidence) để
  chỉ đo năng lực diagnosis. Runner hiện tại không dùng gold evidence.
- Tầng 4 nhận gold span và context/question/answer để phân loại repair action;
  runner độc lập không dùng gold diagnosis/evidence hoặc prediction tầng trước.

Chế độ này trả lời: bản thân từng tầng tốt đến đâu khi input phía trước đúng?

### 2. End-to-end

```text
(content, question, answer)
  → layer 1 label
  → nếu hallucination: layer 2 spans + evidence
  → layer 3 diagnosis cho từng predicted span
  → layer 4 repair action cho từng predicted span (thiết kế E2E)
```

Nếu tầng 1 dự đoán `no_hallucination`, pipeline dừng và trả danh sách span rỗng.
Nếu tầng 1 bỏ sót hallucination, các span/diagnosis tương ứng được tính là bỏ
sót trong metric toàn pipeline. Diagnosis chỉ được so sánh sau khi ghép predicted
span với gold span.

Nên xuất cả hai nhóm kết quả:

- `oracle_layer1`, `oracle_layer2`, `oracle_layer3`;
- `end_to_end` và breakdown theo domain, label, số span.

## Metric contract

| Thành phần | Metric chính | Ghi chú |
|---|---|---|
| Tầng 1 label | Accuracy, per-class P/R/F1, Macro-F1, Weighted-F1 | Chạy trên toàn test set |
| Tầng 2 span | Exact Match, character IoU, span precision/recall/F1 | Báo cáo cả toàn test và hallucinated-only |
| Tầng 2 evidence | Evidence EM, character IoU, evidence availability F1 | `evidence = null` là kết quả hợp lệ cho unsupported span |
| Tầng 3 diagnosis | Accuracy, per-class P/R/F1, Macro-F1, Weighted-F1 | Oracle dùng gold span; E2E dùng matched predicted span |
| Tầng 4 repair action | Accuracy, per-class P/R/F1, Macro-F1, Weighted-F1 | Gold-span hiện tại; chưa sinh câu trả lời sau sửa |
| Toàn pipeline | joint exact success, error propagation | Một mẫu đúng hoàn toàn khi label, tập span/evidence và diagnosis đều đúng |
| Repair/verifier | Faithfulness, PVR | Chỉ dùng khi có response sau sửa và verifier được định nghĩa |

Với nhiều span, không nên tính IoU bằng cách nối chuỗi. Biểu diễn span bằng
character offsets `(start, end)`, ghép predicted/gold spans one-to-one theo IoU
lớn nhất, rồi mới tính extraction và diagnosis. Cần chốt một ngưỡng match (ví dụ
IoU >= 0.5) trước khi chạy thí nghiệm và giữ cố định cho mọi model.

## Faithfulness và PVR

Dataset hiện tại chỉ có `repair_action` ở cấp span; không có `repaired_answer`.
Vì vậy hiện tại có thể đánh giá classification của `repair_action`, nhưng chưa
thể tính Faithfulness hoặc Post-repair Verification Rate một cách hợp lệ.

Để tính hai metric này cần bổ sung:

1. đầu ra `repaired_answer` từ một repair stage;
2. claim splitter;
3. verifier cố định (NLI, LLM judge hoặc human annotation);
4. rule pass/fail và version của prompt/checkpoint verifier.

Nếu sửa chữa là mục tiêu chính thức, nên xem sinh câu sửa/verifier là `repair stage` sau tầng 4,
không gộp ngầm vào diagnosis.

