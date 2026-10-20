# Cập nhật dữ liệu ngày 18/10/2026

Dataset mặc định tại `artifacts/dataset/` hiện có **14.187 response**. Cả 30 cấu hình và pipeline dùng đường dẫn này; cách chạy notebook giữ nguyên.

Nguồn: `guided_hallucination_sum/outputs/dataset_expansion/training_snapshots/fda18db1da1651143c26db9fa82438018798fa1398b4be01e6283a1b917d67a9/manifest.json` ở thư mục cha của package. Đây là snapshot các mẫu đã được chấp nhận, không phải toàn bộ đợt mở rộng đã hoàn tất. Không lấy mẫu đang chờ hoặc bị loại vào thực nghiệm.

| Split | Bộ cũ | Thêm | Bộ mới |
|---|---:|---:|---:|
| Train | 11.124 | 233 | 11.357 |
| Validation | 1.388 | 37 | 1.425 |
| Test | 1.384 | 21 | 1.405 |
| Tổng | 13.896 | 291 | 14.187 |

- Xác minh hash toàn bộ manifest nguồn và manifest chuẩn hóa.
- Đối chiếu toàn bộ bản ghi chuẩn hóa theo ID: 13.896 mẫu cũ giữ nguyên, không có ID bị bỏ hoặc thay đổi.
- Không chia lại dữ liệu. Kiểm tra document/context chuẩn hóa trên 16.973 ID nguồn, kể cả pending: không giao split.
- Xác minh offset/quote span và evidence bằng bộ chuyển đổi hiện có; 10.988 vùng lỗi, 62 vùng đa diagnosis.
- Bốn nhãn response và 11 nhãn diagnosis giữ nguyên. Evidence sửa lỗi của unsupported không được dùng làm contradiction evidence.
- `quality_certified=False`; kết quả kiểm tra cấu trúc không thay thế duyệt ngữ nghĩa bởi người. Chưa loại trừ near-duplicate ngữ nghĩa.

Manifest chuẩn hóa mới SHA-256: `e3f7c0787ab1bd759463253f396ddd6cb3f69c5ea8aa7f900f3c0db682cf4c9e`.

Manifest chuẩn hóa cũ SHA-256: `08129fccdde21c8881a136775da68c65bc603992597cb9a7e4ab6230ccda6990`.

Bản cũ được lưu tại `artifacts/previous_13896/`: dataset, cấu hình, tài liệu, baseline, preflight và ZIP trước cập nhật. Báo cáo máy đọc được cùng 291 ID bổ sung ở `artifacts/data_updates/2026-10-18.json` trên máy local. ZIP mới chỉ chứa dataset hiện tại và mã/tài liệu cần cho thực nghiệm.

Validation và test đã mở rộng: so sánh các mô hình trên cùng snapshot mới; chạy lại đánh giá và không gộp điểm cũ vào bảng mới. Không resume checkpoint đã khóa hash dataset cũ với dataset mới. Bộ kiểm tra CPU ba tầng và hai baseline validation đã chạy lại; xem `VALIDATION.md`. Chưa chạy huấn luyện mô hình trên GPU.
