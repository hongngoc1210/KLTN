# Checkpoint thấp hơn tiến độ trong log

Log loss ghi nhận bước đã thực thi, không chứng minh file checkpoint còn tồn tại trên Drive. Trường hợp đang điều tra có log tới step 1448 và 1365 trong thư mục log XLM-R-base, nhưng latest.json hiện trỏ step 300, không có best.json. Không kết luận nguyên nhân cụ thể từ các thông tin này.

Ngoài ra, bản logger cũ lấy thư mục log từ cell setup và không ghi command. Khi gọi `run(..., --config, ...)` thủ công với config khác, tên thư mục log không tự thay đổi theo config. Vì vậy log cũ thiếu bằng chứng đầy đủ về đường dẫn checkpoint thực sự mà tiến trình sử dụng.

## Phục hồi run hiện có

1. Dừng các phiên ghi cùng run; giữ nguyên workspace và checkpoint còn lại. Không sửa số step trong state.json/latest.json và không ghép weights/optimizer của các bước khác nhau.
2. Kiểm tra cả thư mục `step-*` và `.staging-*` dưới `components/<task>/checkpoints/`. Các bản staging hoàn chỉnh có thể đáng kiểm tra nhưng không được tự coi là checkpoint hợp lệ.
3. Kiểm tra thư mục trên giao diện Google Drive, các bản sao/output và Thùng rác. Nếu tìm được bản cũ, cần toàn bộ model/, trainer.pt, state.json, manifest.json của cùng checkpoint. Khả năng phục hồi phụ thuộc dữ liệu còn tồn tại; không bảo đảm file bị xóa sẽ nằm trong Thùng rác.
4. Xác minh toàn bộ hash trong manifest, state identity/run_id và cấu hình trước khi chọn bản phục hồi. Một pointer JSON cũ không thay thế được file weights và optimizer đã mất.
5. Nếu cuối cùng chỉ còn checkpoint step 300 hợp lệ và không có bản sao cao hơn, chỉ có thể resume chính xác từ 300. Không dựng lại checkpoint 1365/1448 từ log loss.

[Google Drive: khôi phục file đã xóa](https://support.google.com/drive/answer/1716222).

## Thay đổi bảo vệ trong mã local

- Đọc và xác minh checkpoint sau khi lưu; kiểm tra pointer đọc lại khớp giá trị vừa ghi.
- Không cho latest lùi step hoặc đổi run identity.
- Cleanup không xóa gì khi latest thiếu; nếu có checkpoint không được giữ nhưng có step cao hơn hoặc bằng latest, hay thuộc run khác, dừng toàn bộ cleanup.
- Ghi lịch sử pointer bất biến theo tên checkpoint. Resume dừng khi latest thấp hơn lịch sử đã ghi, hoặc khi thiếu latest nhưng còn artifacts checkpoint.
- Log mới có INVOCATION (command/cwd), RESUME_CHECKPOINT/FRESH_TRAINING và CHECKPOINT_SAVED kèm đường dẫn. Kiểm tra đọc lại file không chứng minh dữ liệu đã bền vững trên cloud sau khi runtime bị thu hồi.

Các thay đổi này khắc phục một lỗ hổng có thể xóa checkpoint mới hơn khi latest bị cũ/lùi. Kiểm thử mô phỏng đã tái hiện điều kiện nguy hiểm; chưa xác nhận nó là nguyên nhân của sự cố trên Drive đang được điều tra.

**Không chép đè mã mới vào workspace của run cũ để ép resume.** Run khóa hash mã nguồn; thay mã sẽ làm identity khác. Hai ZIP Colab/Kaggle đã phát hành trước bản sửa này được giữ nguyên, chưa tự cập nhật. Bản sửa local dành cho đợt chạy mới hoặc một quy trình chuyển đổi được ghi nhận riêng; nó không phục hồi được file đã mất.
