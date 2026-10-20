# Kế hoạch thực nghiệm cho khóa luận

## 1. Câu hỏi nghiên cứu

1. Encoder fine-tune và LLM prompting/SFT khác nhau thế nào khi phân loại bốn loại response?
2. Khả năng định vị span và tìm evidence khác nhau thế nào theo kiến trúc và kích thước model?
3. Diagnosis đa nhãn trên vùng lỗi tiếng Việt có cải thiện khi fine-tune không?
4. Chênh lệch giữa đầu vào tham chiếu và pipeline thực phản ánh lỗi lan truyền ở đâu?

Không dự đoán repair action, không sinh câu trả lời sửa và không tái gán nhãn production trong đợt này. mDeBERTa-NLI để sau ma trận chính.

## 2. Dataset và hợp đồng tác vụ

Snapshot cố định `fda18db1da1651143c26db9fa82438018798fa1398b4be01e6283a1b917d67a9`; 14.187 response đủ điều kiện, train 11.357 / validation 1.425 / test 1.405. Chỉ dùng ID đã có trong release rút gọn, nhưng đọc annotation chi tiết từ snapshot `source/dataset.jsonl` để lấy hallucinated_spans và evidence đúng mục đích. Đây là snapshot các mẫu đã được chấp nhận tại thời điểm xuất, chưa phải toàn bộ đợt mở rộng hoàn tất. So với bộ trước: giữ nguyên 13.896 mẫu, thêm 291 mẫu; xem DATA_UPDATE.md. Khóa snapshot này cho toàn bộ ma trận 30 cấu hình và ba seed; không cập nhật dữ liệu giữa các run được so sánh.

Không chia lại split, không thêm gold answer làm mẫu no-hallucination, không cân bằng validation/test. Kiểm tra giao document và context NFC/gộp khoảng trắng trên toàn snapshot, kể cả mẫu không đủ điều kiện. Việc này không loại trừ near-duplicate ngữ nghĩa.

| Nhãn response | Train | Validation | Test |
|---|---:|---:|---:|
| no_hallucination | 3063 | 405 | 414 |
| intrinsic | 5897 | 730 | 695 |
| extrinsic | 2245 | 271 | 271 |
| mixed | 152 | 19 | 25 |

`mixed` là cùng có contradicted và unsupported. Không dùng nghĩa supported + hallucinated.

Offset là chỉ số code point Unicode `[start,end)` trên `context`/`response` nguyên gốc. Không strip, chuẩn hóa Unicode hoặc bỏ Markdown trước khi xác minh offset. Chuẩn hóa chỉ dùng kiểm tra trùng context. Tokenizer xử lý trên bản đầu vào riêng và có ánh xạ về offset gốc.

Gộp span thực sự chồng lấn, giữ riêng span chỉ tiếp giáp. Diagnosis và contradiction evidence của vùng gộp là hợp annotation nguồn. Kết quả 10.988 vùng, 62 vùng có hai diagnosis. `source_atom_ids` và annotation gốc được giữ để truy vết. Không dùng `candidate_spans` của claim thay cho `annotation.hallucinated_spans`.

10 nhãn tầng 3: incorrect_reference, implicit_subject_drift, sino_vietnamese_term_confusion, numerical_distortion, circumstance_error, entity_substitution, predicate_error, other, unsupported_reference, unsupported_addition. Các lỗi nhầm đối tượng do danh xưng được gộp vào `entity_substitution`.

Evidence xác minh chỉ lấy từ annotation contradicted, `evidence_purpose=contradiction`, `evidence_scoring_eligible=True`. Unsupported thuần túy có target evidence rỗng. `repair_generalization` giữ riêng trong audit. Một vùng gộp có cả hai quan hệ vẫn nhận evidence của phần contradicted.

Nguồn nhãn vẫn là `existing_annotations_not_human_verified`, `quality_certified=False`. Chuỗi trích dẫn khớp offset chưa đảm bảo đúng ngữ nghĩa. Báo cáo có danh sách cần rà soát nhưng không tự chỉnh annotation theo dự đoán model.

## 3. Ma trận 30 cấu hình

| Tầng | Fine-tune encoder | Seq2seq | Qwen3.5-4B | Gemma-4-E4B-it | Tổng |
|---|---|---|---|---|---:|
| 1 | PhoBERT, XLM-R-base/large, mDeBERTa, mmBERT | — | zero-shot, 4-shot, QLoRA | zero-shot, 4-shot | 10 |
| 2 | XLM-R-base/large, mDeBERTa, mmBERT | ViT5-base | zero-shot, 4-shot, QLoRA | zero-shot, 4-shot | 10 |
| 3 | PhoBERT, XLM-R-base/large, mDeBERTa, mmBERT | — | zero-shot, 4-shot, QLoRA | zero-shot, 4-shot | 10 |

Head encoder tầng 1 là linear softmax 4 lớp. Tầng 3 nối biểu diễn toàn input với trung bình hidden states trong span đã đánh dấu, linear sigmoid 10 nhãn. Ngưỡng 0,5; nếu rỗng, lấy argmax. Loss tương ứng cross-entropy/BCE, không dùng class weighting hoặc oversampling trong baseline đầu tiên. Nhãn hiếm được báo support riêng.

Tầng 2 encoder có hai checkpoint độc lập: span BIO trên response; evidence BIO trên context, có điều kiện bằng response đánh dấu span. Head evidence học cả vùng có evidence và unsupported với nhãn toàn O. Cách này hỗ trợ nhiều evidence rời nhau, không ép về một start/end duy nhất.

### Input dài và đối chiếu công bằng

PhoBERT tối đa 256 token; XLM-R/mDeBERTa 512; mmBERT profile 2048. Cấu trúc cửa sổ: `[CLS] question [SEP] response-window [SEP] context-window [SEP]`. Cửa sổ response dài 64 token với PhoBERT, 128 với encoder khác; context dùng phần ngân sách còn lại; overlap 25%. Giữ mọi tổ hợp cửa sổ response/context.

Tầng 1 gộp trung bình biểu diễn cửa sổ rồi mới áp dụng head và loss response. Tagging gộp trung bình biểu diễn các bản sao của token gốc trước head/loss; mỗi token chỉ góp một nhãn. Tầng 3 giữ biểu diễn vùng đánh dấu qua mọi cửa sổ. Không nhân nhãn `mixed` thành nhãn của từng đoạn context riêng lẻ. Question vượt 64 token ghi overlength; không âm thầm cắt question.

Tag token dương nếu giao vùng annotation; phần word/subword bị cắt giữa chừng có thể không khôi phục đúng boundary ký tự, nên báo cả exact F1 lẫn character-F1. Các model dùng tokenizer/ngân sách khác nhau: báo coverage, số cửa sổ, thời gian và VRAM; không coi hiệu năng thuần túy là chỉ do kiến trúc.

ViT5 input tối đa 2048 token. Qwen train tối đa 4096 token gồm input + output reserve; Qwen/Gemma inference tối đa 8192. Output reserve tầng 1/2/3: 32/1024/192 token. Không cắt target để ép vừa. Train quá dài bị loại có ghi ID; validation/test quá dài vẫn trong mẫu số và có trạng thái lỗi.

### Tham số và lựa chọn checkpoint

- Encoder: AdamW lr 2e-5, weight decay 0,01, tối đa 5 epoch, patience 2.
- ViT5: lr 5e-5, tối đa 5 epoch, patience 2.
- Qwen: QLoRA NF4 double quantization, r16/alpha32/dropout0,05, lr2e-4, tối đa 3 epoch.
- Microbatch một response/region hoàn chỉnh; accumulation 16; warmup 6%; gradient clipping 1,0; gradient checkpointing. FP16 trên T4, BF16 khi hỗ trợ; precision thực được khóa trong run.
- Span head chọn validation exact span F1; evidence head chọn validation evidence character-F1. Tầng 1 chọn macro-F1 bốn lớp, tầng 3 chọn macro-F1 trên các nhãn có reference support. Generative tầng 2 chọn trung bình exact span F1 và evidence character-F1. Cùng chỉ số trung bình này dùng xếp hạng cấu hình tầng 2 hoàn chỉnh.
- Checkpoint đánh giá mỗi epoch. Lưu latest mỗi 100 optimizer step và cuối epoch; lưu best theo validation, giữ latest và best. Không chọn bằng test hoặc loss test.

### Giao diện output

```json
{"label":"mixed"}
```

```json
{"spans":[{"start":0,"end":2,"quote":"20","evidence":[{"start":8,"end":10,"quote":"10"}]}]}
```

```json
{"diagnosis_labels":["numerical_distortion","circumstance_error"]}
```

Ví dụ offset chỉ minh họa schema, phải được kiểm tra với văn bản của từng mẫu. JSON sai, duplicate key, trích dẫn không khớp, OOM hoặc output hết ngân sách được ghi trạng thái; không strip code fence, tự dò lại vị trí hoặc retry bằng prompt khác. Few-shot chứa bốn ví dụ cố định train ở document khác nhau; chọn ví dụ ngắn theo độ dài ký tự và tie-break seed, không truy xuất theo nhãn test.

## 4. Metric và trình tự chạy

Trên Colab, dùng trực tiếp notebook `02_tier1_classification`, `03_tier2_span_evidence` hoặc `04_tier3_diagnosis`. Mỗi notebook đã gộp setup/mount Drive/biến môi trường và kiểm tra dataset trong ZIP. Ba tầng độc lập không cần chạy tuần tự; chỉ pipeline dùng output dự đoán từ tầng trước. Các notebook trong cùng ma trận dùng chung ZIP và WORKSPACE_NAME. Hướng dẫn thao tác ở [COLAB_GUIDE.md](COLAB_GUIDE.md).

1. Prepare, kiểm thử CPU và kiểm tra annotation/coverage; đóng gói dataset cùng mã.
2. Smoke GPU từng họ; đo thời gian từ smoke và số token/cửa sổ để lên ngân sách. Không ước lượng số giờ cố định khi chưa đo GPU thực.
3. Chạy đủ seed 42: XLM-R-base → mDeBERTa → PhoBERT/mmBERT → XLM-R-large → ViT5 → Qwen/Gemma. Chạy validation và lưu prediction đầy đủ.
4. Khóa hai cấu hình có train tốt nhất mỗi tầng qua `select`, chạy thêm seed 17 và 2026. Không lặp greedy zero/few-shot thành ba "training seed" giả.
5. Khóa prompt/checkpoint; chạy test cho từng cấu hình và pipeline seed 42 được chọn từ validation. Báo riêng nhóm chỉ một seed và nhóm ba seed, không chọn lại best bằng test.

Tầng 1: macro-F1 bốn lớp, accuracy, confusion matrix và support. Tầng 2: exact span F1, F1 ghép một-một tại IoU≥0,5, micro character-F1/IoU, độ đúng trên response không có lỗi. Ghép tối đa số cặp hợp lệ trước, rồi tối đa tổng IoU; một prediction không khớp hai reference.

Evidence: so sánh vị trí ký tự trong context, exact match theo tập đoạn chuẩn hóa, accuracy của evidence rỗng. End-to-end evidence chấm theo span lỗi đã ghép IoU; vùng lỗi bị bỏ sót cũng làm mất evidence. Evidence gắn vào span sai không được tính đúng chỉ vì có xuất hiện đâu đó trong output. Oracle evidence được báo riêng với span tham chiếu.

Tầng 3: macro-F1 trên nhãn có support, macro đầy đủ 10 nhãn, micro-F1, exact-set accuracy, per-label support. Pipeline: exact và relaxed F1 của cặp span–diagnosis, evidence liên kết và số lỗi qua từng tầng. Tầng 1 dự đoán no_hallucination sẽ dừng; false negative ở tầng này vẫn làm mất toàn bộ span tương ứng trong metric cuối.

Mọi mẫu validation/test giữ trong mẫu số. Missing/invalid không được tự chuyển thành no_hallucination hoặc evidence rỗng đúng. Negative response không có span không tăng span-F1 một cách giả tạo; được đo ở clean-response accuracy và coverage. Báo theo domain, nguồn annotation, nhóm tổng độ dài ký tự (<2000, 2000–3999, ≥4000). CI95 bootstrap 1.000 lần theo document, seed42. CI phản ánh biến thiên mẫu, không xác minh độ đúng annotation.

Baseline: majority response label và tập diagnosis phổ biến nhất từ toàn train, giữ nguyên phân bố validation/test. Không lấy tập giao các prediction thành công giữa model làm cohort chính.

## 5. Nguồn kỹ thuật

- [PhoBERT-base-v2](https://huggingface.co/vinai/phobert-base-v2): giới hạn 256 và tách từ VnCoreNLP.
- [mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base): ModernBERT multilingual; profile thực nghiệm giới hạn 2048 dù checkpoint hỗ trợ dài hơn.
- [PsiloQA](https://aclanthology.org/2025.findings-emnlp.626/): tham khảo hướng token-level hallucination detection; evidence head là phần mở rộng của bộ mã này.
- [Qwen3.5 Transformers](https://huggingface.co/docs/transformers/model_doc/qwen3_5), [Gemma4 Transformers](https://huggingface.co/docs/transformers/model_doc/gemma4), [PEFT quantization](https://huggingface.co/docs/peft/main/en/developer_guides/quantization): loader đúng kiến trúc, template, QLoRA.

Các phiên bản môi trường được ghi trong manifest sau lần prepare model đầu tiên. Không sao chép environment Transformers 4.57.1 của encoder để load Gemma 4; dùng profile LLM và pipeline từng stage nếu cần.
