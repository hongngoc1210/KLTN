# Đánh giá Layer 2

Tất cả model dùng `layer2_metrics()` khi đánh giá test độc lập. Input là
question, answer và context gốc; không sử dụng prediction Layer 1.

## Quy tắc vị trí dùng chung cho EM và IoU

- Span được định vị trong answer, evidence trong context, bằng offset ký tự
  `[start, end)` của văn bản gốc. Whitespace và dấu câu được giữ nguyên.
- Encoder dùng offset BIO. Cùng chuỗi ở hai vị trí khác nhau là hai prediction.
  Một prediction lặp đúng cả chuỗi và offset chỉ được tính một lần.
- Chuỗi NFC/NFD tương đương được chấp nhận. Offset luôn trỏ vào source gốc;
  không gộp whitespace, không bỏ dấu, không fuzzy-match.
- Gold hỗ trợ `span_offset: [start, end]`, `evidence_offset: [start, end]` trong
  từng phần tử `span_label`. Nếu không có offset, cả gold lẫn model sinh dùng
  lần xuất hiện NFC-equivalent đầu tiên. Không chọn vị trí dựa trên prediction
  hay gold của phía còn lại. `ambiguous_span_offset_count` và
  `ambiguous_evidence_offset_count` đếm annotation gold có nhiều vị trí khả dĩ
  nhưng chưa có offset; cần kiểm tra các annotation này nếu muốn loại bỏ quy
  ước lần xuất hiện đầu. Hai chỉ số này đếm annotation, không đếm sample.
- Prediction không nằm trong source hoặc có offset sai vẫn là false positive.
  Gold thiếu `span_label`, sai chuỗi hoặc sai offset sẽ báo lỗi trước inference.
- `NONE`, `null`, `không có`... chỉ được chuẩn hóa thành rỗng khi không tìm thấy
  chuỗi đó trong source. Chuỗi thật trong source được ưu tiên giữ lại. JSON sai
  cấu trúc làm cả sample thành parse error; parser hiện không phục hồi một phần.

## Các metric

`span` và `evidence` có cùng các trường bên dưới:

| Trường | Định nghĩa |
|---|---|
| `exact_match` | Tỷ lệ sample có tập offset prediction bằng toàn bộ tập offset gold, không có prediction ungrounded và không parse error. Hai tập rỗng hợp lệ được điểm 1. |
| `positive_exact_match` | Cùng định nghĩa EM, nhưng chỉ lấy sample có gold không rỗng của trường đang xét: `span` có gold span, `evidence` có gold evidence. Phải đúng toàn bộ tập offset, không chỉ đúng một phần. Prediction rỗng/sai hoặc parse error trên các sample này vẫn nằm trong mẫu số và được 0. Không có sample positive thì trả 0. |
| `positive_exact_match_count`, `positive_sample_count` | Tử số và mẫu số của positive EM: số sample positive đúng hoàn toàn và tổng số sample có gold của trường đó. Sample có span nhưng evidence=null được tính cho span positive EM, không được tính cho evidence positive EM. |
| `precision`, `recall`, `f1` | P/R/F1 micro trên tất cả đơn vị trích xuất: P=TP/predicted_count, R=TP/gold_count. Ghép một-một tối đa số cặp với IoU ≥ ngưỡng (mặc định 0,5). |
| `mean_matched_iou` | IoU trung bình chỉ trên các cặp đã match. Chỉ số có điều kiện, không phải IoU của toàn test. Khi có nhiều cách ghép tối đa, ưu tiên cạnh IoU cao theo thứ tự xác định, không tối ưu tổng IoU. |
| `mean_sample_char_iou` | Trung bình sample IoU giữa hợp các vị trí ký tự prediction và gold. Rỗng–rỗng hợp lệ = 1; parse error = 0. |
| `mean_positive_sample_char_iou` | Sample character IoU trung bình chỉ trên sample có gold của trường đó; tránh điểm cao do nhiều mẫu rỗng đúng. |
| `char_precision`, `char_recall`, `char_f1` | P/R/F1 micro trên hợp vị trí ký tự. Chuỗi ungrounded chịu phạt bằng độ dài của nó ở mẫu số prediction/union. Đây là quy ước phạt bổ sung, được ghi ở `character_policy`. |
| `true_positive`, `predicted_count`, `gold_count` | Số cặp match, số prediction và số gold sau dedup. Prediction ungrounded vẫn được tính. |
| `ungrounded_count`, `ungrounded_rate` | Số prediction không resolve được và tỷ lệ trên predicted_count. `ungrounded_char_count` là tổng số ký tự chịu phạt. |
| `empty_gold_*` | Số sample gold rỗng theo từng trường và tỷ lệ sample đó có ít nhất một prediction. `clean_*` là alias cũ của các trường này. Evidence rỗng không đồng nghĩa với answer không hallucination. |

Các trường cấp trên:

| Trường | Định nghĩa |
|---|---|
| `macro_f1` | `(span.f1 + evidence.f1) / 2`; trung bình hai nhiệm vụ trích xuất, không phải Macro F1 bốn lớp của Layer 1. |
| `macro_char_f1` | `(span.char_f1 + evidence.char_f1) / 2`; chỉ số bổ sung. |
| `joint_exact_match` | Tỷ lệ sample đúng cả span EM và evidence EM theo cùng offset, chưa xét liên kết từng cặp. |
| `span_evidence_pair_exact_match` | Tỷ lệ sample đúng toàn bộ tập cặp `(span_offset, evidence_offset hoặc null)`. Hoán đổi evidence giữa hai span sẽ bị tính sai. |
| `pair_evaluable_count` | Số sample adapter có xuất liên kết `items`. Encoder BIO hiện không dự đoán liên kết, nên count=0 và pair EM=null. Không suy ra liên kết bằng thứ tự hoặc gold. |
| `evidence_at_iou_0_3` | Bộ metric evidence với ngưỡng phụ 0,3. Có thể đặt `model.params.relaxed_iou_threshold`; tên key thay đổi theo ngưỡng, ví dụ `evidence_at_iou_0_25`. |
| `no_hallucination_*` | Số sample không có gold span và tỷ lệ có ít nhất một prediction span/evidence. Parse error được báo riêng, không bị coi là false positive nếu không có prediction. |
| `parse_error_count`, `parse_error_rate` | Số/tỷ lệ output sai JSON/schema. Không được điểm EM, joint/pair EM hay sample IoU; không được TP nếu record lỗi vẫn có prediction. Gold của sample vẫn nằm trong mẫu số recall. |
| `iou_threshold`, `relaxed_iou_threshold` | Ngưỡng chính và phụ; cả hai phải thuộc (0,1]. |

P/R/F1 bằng 0 khi mẫu số tương ứng bằng 0. Đánh giá luôn giữ toàn bộ test,
bao gồm no-hallucination và parse error. IoU của một cặp đoạn là
`intersection_length / (len(pred) + len(gold) - intersection_length)`.

## Loss, chọn checkpoint và early stopping

Gradient training vẫn tối ưu cross-entropy loss. Metric validation chỉ dùng để
chọn best checkpoint và early stopping; không trực tiếp thay thế loss.

| Model | Metric monitor hiện tại trên dev |
|---|---|
| XLM-R-base/large, mmBERT BIO | Decode BIO bằng offset gốc, rồi `eval_macro_f1` = trung bình F1 extraction của span và evidence @IoU 0,5. Token F1 được log riêng bằng `token_span_f1`, `token_evidence_f1`, `token_macro_f1`; `accuracy` vẫn là token accuracy phụ của 5 BIO nhãn. |
| ViT5 | `eval_macro_f1`: F1 extraction @IoU 0,5 sau sinh JSON tự do và parse. |
| Qwen QLoRA | Generate JSON tự do chỉ từ prompt, rồi tính `eval_macro_f1` extraction @IoU 0,5. Validation loss vẫn dùng completion gold nhưng token gold không được truyền vào `generate()`. |
| Qwen/Gemma zero-shot, few-shot | Không train, không early stopping. |

Tất cả model trainable dùng chung `extraction_validation_metrics()` trên toàn
bộ dev, không lấy subset: `macro_f1 = (span_f1 + evidence_f1) / 2` với cùng
matching/IoU 0,5. ViT5 và Qwen đều sinh tự do; encoder decode prediction BIO.
Không dùng token F1 hay teacher-forced token argmax để chọn checkpoint.
Config mặc định evaluate mỗi epoch, patience=2,
early_stopping_threshold=0,0001, greater_is_better=true. `macro_char_f1`, loss,
EM và IoU trung bình không phải monitor. Qwen generation validation sẽ chậm
hơn teacher forcing. Tqdm evaluation đếm batch đã hoàn tất generate.
Positive EM được log thêm dưới tên `eval_span_positive_exact_match` và
`eval_evidence_positive_exact_match`; không thay đổi monitor `eval_macro_f1`.
Chỉ bổ sung chỉ số báo cáo này không cần train lại; có thể chấm lại prediction
còn đủ offset/source hoặc chạy evaluation từ checkpoint đã lưu.

Các vị trí code:

- `validation.py`: flatten cùng bộ metric extraction cho cả ba họ model.
- `encoder_adapter.py`: lưu offset dev và dùng cùng decoder như test.
- `causal_generation_trainer.py`: tách SFT loss input và prompt-only generation
  input; right-padding cho loss, left-padding cho generate. Generated completion
  được gather bởi Trainer trước `on_evaluate`, để early stopping/chọn checkpoint
  nhận được đúng `eval_macro_f1`.
- `qwen35_4b_qlora_layer2.py`: chuẩn bị prompt dev độc lập với gold completion,
  parse generated IDs; Layer-1 QLoRA giữ luồng validation cũ.

## Đánh giá lại và huấn luyện lại

- Model đã train có thể load checkpoint và chạy `--stage evaluate` để lấy metric
  mới; không cần train lại chỉ để tính test scores.
- Encoder đã train theo token F1 vẫn dùng được để inference. Muốn chọn lại best
  checkpoint theo quy trình chung thì re-score các checkpoint trên **dev** bằng
  metric extraction. Tương tự với checkpoint Qwen từng chọn bằng teacher forcing.
  Nếu checkpoint cần thiết đã bị xóa hoặc muốn tái lập early stopping mới thì
  chạy training mới. Không chọn checkpoint dựa trên test.
- Các config trainable hiện lưu output vào `*_extraction_f1`, tách khỏi run cũ
  `xlmr_base`, `xlmr_large`, `mmbert_base`, `vit5_base_macro_f1` và
  `qwen35_4b_qlora_macro_f1`. `resume_from_checkpoint="auto"` chỉ resume trong
  thư mục mới. Khi evaluate artifact cũ, dùng config trỏ output_dir về run cũ.
  Không chuyển resume checkpoint cũ sang run mới: best_metric/patience cũ không
  so sánh được dù cùng tên key `macro_f1`.
- JSON generative cũ có `raw_output` có thể parse và chấm lại mà không generate
  lại nếu còn đúng source data. Encoder output cũ chỉ có chuỗi đã mất offset
  thì cần chạy inference lại từ checkpoint.

## ViT5: tokenizer JSON và bộ demo

Tokenizer gốc của `VietAI/vit5-base` không biểu diễn đúng dấu `{`, `}` và một
số ký tự của source. Adapter dùng codec khả nghịch cho ba ký tự JSON:
`{` ↔ `<extra_id_0>`, `}` ↔ `<extra_id_1>`, `\` ↔ `<extra_id_2>`.
Đây là các ID có sẵn đã được T5 pretrain, không tạo hàng embedding cú pháp
ngẫu nhiên và không ghi đè embedding cũ. Codec dùng cho cả input lẫn target;
validation/test vẫn generate tự do, không nhận gold hoặc prefix JSON ép sẵn.

Khi decode, giữ ba sentinel trên để khôi phục ký tự JSON, bỏ pad/EOS và các
special token còn lại. Literal như `<extra_id_0>` hoặc `<pad>` bên trong chuỗi
được escape ở bước encode để không lẫn với cú pháp. Metaspace chỉ thêm khoảng
trắng đầu toàn input, không thêm ở từng đoạn quanh sentinel/added token. Codec
không thêm ngoặc đóng bị thiếu, không sửa schema hoặc nội dung model đã sinh.
Ví dụ model sinh thiếu dấu `}` thì output sau decode vẫn sai và nhận parse error.

Ký tự ngoài cú pháp JSON chỉ lấy từ question/answer/context của **train** để
bổ sung vocabulary khi cần. Không thêm cả dải Unicode
General Punctuation: các space/control token không có trong train có thể
chiếm ưu thế trong generation khi chưa được học. Không tạo vocabulary từ
dev/test hoặc gold của chúng. Embedding được mở rộng khi cần, giữ nguyên ID
token cũ. Tokenizer đã mở rộng được lưu cùng checkpoint/artifact.

Embedding của các ký tự source bổ sung được resize bằng `mean_resizing=False`,
rồi khởi tạo từ embedding ký tự gần nghĩa, NFKC có sẵn hoặc UNK pretrained,
không dùng một
vector ngẫu nhiên có thể lấn át output. Nhiễu nhỏ tách các alias; không sửa
embedding token cũ. Đây chỉ là khởi tạo trọng số, không ép prefix khi generate.

T5 tính cross-entropy đã lấy trung bình và không dùng `num_items_in_batch`.
Adapter đặt `trainer.model_accepts_loss_kwargs=False` để Trainer chia loss
đúng theo gradient accumulation và không nhân thêm số GPU do nhận nhầm
`**kwargs` của T5. Không thay loss thành metric hoặc thay đổi early stopping.

Khi validation có parse error, log in tối đa ba `raw_output` lỗi (giới hạn
500 ký tự/mẫu) để phân biệt output lặp token với lỗi JSON khác. Parser vẫn
giữ nguyên, không tự sửa JSON hoặc biến output lỗi thành dự đoán đúng.

Trước khi train, target của train/dev phải encode rồi decode về cùng JSON
payload (chấp nhận NFC/NFD tương đương). Target vượt `max_target_length` hoặc
bị đổi/mất ký tự sẽ báo lỗi, không bị cắt cụt âm thầm. `max_target_length` bao
gồm EOS; generation chừa thêm một vị trí decoder-start của T5.

Khi chạy lại ViT5, tự đặt `training.output_dir` mới trong config để tách khỏi
run cũ và không resume checkpoint lỗi. Checkpoint thiếu policy tokenizer mới hoặc có
vocabulary khác bị từ chối khi load/resume. **ViT5 đã học target mất ngoặc cần
train lại**, không thể chỉ sửa metric rồi chấm lại. Qwen/encoder không cần
train lại vì lỗi tokenizer riêng của ViT5. Metric early stopping vẫn giữ nguyên.
Checkpoint dùng JSON literal cũ không tương thích với codec sentinel và sẽ
bị từ chối thay vì decode hoặc resume nhầm. Giữ checkpoint cũ nếu cần phân
tích bằng phiên bản code đã tạo ra nó. Run mới lưu `layer2_training_policy`
trong config model và policy codec trong tokenizer để kiểm tra.
Sửa vocabulary/khởi tạo/loss không bảo đảm F1 > 0 ngay epoch đầu; cần xem generated output
và đường cong dev của run mới. Test CPU mô hình nhỏ chỉ kiểm chứng đường code,
không thay thế training/evaluation ViT5 đầy đủ trên GPU.

Bộ `qa_models/data/demo_20_4_4` giữ nguyên 20/4/4 record từ train/dev/test gốc,
đủ bốn nhãn và có manifest truy xuất nguồn. Tự chỉnh đường dẫn data và output
trong config model khi chạy demo; không dùng điểm trên 4 test samples để kết luận chất lượng.
Với demo nhỏ, đặt `warmup_steps=0`, `early_stopping_patience=0` để không dừng
khi model còn học cú pháp. Có thể đặt `training.params.load_best_model_at_end=false`
để kiểm tra model cuối run: nếu mọi epoch có F1=0, best-checkpoint mặc định vẫn
là epoch đầu, dù model cuối đã tiến bộ về cú pháp. Mặc định tham số này vẫn là
true cho thí nghiệm đầy đủ; không thay metric `macro_f1` hoặc cách tính điểm.

Kiểm tra CPU bằng ViT5-base thật trên 20 mẫu, 100 update, LR tuyến tính từ
5e-5 về 0 (không early stop, không train trên dev): 2/4 output dev có JSON hợp
lệ, span/evidence F1 vẫn bằng 0. Kết quả này chỉ chứng minh pipeline đã có thể
học/sinh định dạng; không chứng minh model đã trích xuất đúng hoặc lỗi sinh
đã được loại bỏ hoàn toàn. Không lấy unit test hoặc loss giảm để cam kết F1
trên bộ lớn. Cần huấn luyện và đánh giá đầy đủ cho kết luận chất lượng.

Faithfulness và Post-repair Verification Rate thuộc bước kiểm chứng/sửa chữa,
không được tính bằng các metric extraction trong file này.
