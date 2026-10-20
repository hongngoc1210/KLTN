# Gộp nhãn diagnosis

Dataset hiện dùng và các bản train/dev/test trong `../hallucination_dataset`
đã gộp `honorific_*` vào `entity_substitution`. Tầng 3 còn 10 nhãn.
Các danh sách diagnosis được loại trùng sau khi gộp; văn bản, ID, span,
evidence, split và nhãn tầng 1 giữ nguyên.

Quy tắc được áp dụng ngay khi đọc snapshot nguồn, nên prepare lại vẫn cho
taxonomy mới. Prompt, số đầu ra classifier và metric dùng cùng 10 nhãn;
metric macro trên toàn bộ lớp có tên `macro_f1_all_10`.
Các thống kê diagnosis và hash trong manifest dataset được tính lại.

Snapshot nguồn, bundle ZIP đã phát hành, kết quả baseline và checkpoint cũ
là bản lịch sử của taxonomy trước khi gộp; không sửa lại nội dung của chúng.
Chi tiết các file dataset đã đổi và bản sao lưu trước chuyển đổi nằm ở
`../hallucination_dataset/diagnosis_merge_report.json`.

Dùng run mới cho dataset này: hash dataset/code đã đổi và head tầng 3 có
10 lớp. Không thay dataset/code mới vào workspace đang resume run cũ.
Các bundle Colab/Kaggle cũ và bản đã upload lên Drive chưa được cập nhật
bởi thay đổi local này.
