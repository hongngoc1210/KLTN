# Kiểm tra sau khi chuyển sang sum_models

Đã chạy local ngày 20/10/2026:

- Bộ unittest: **35 test, 33 pass, 2 skip**.
- Hai test skip cần Transformers (encoder architecture và ViT5); môi trường kiểm thử
  có PyTorch CPU nhưng chưa cài Transformers.
- Test thực tế trên model PyTorch nhỏ: gradient truyền tới encoder, save/load,
  ngắt rồi resume cho kết quả giống chạy liên tục, chống pointer checkpoint lùi step.
- Test giao diện mới: 30 TOML, đường dẫn độc lập CWD, hai cách gọi CLI,
  điều phối `all`/`evaluate`, đóng băng dataset và từ chối dữ liệu bị thay đổi.
  Các bước tải/train model pretrained trong test giao diện được mock.
- Kiểm tra schema notebook bằng nbformat và parse tất cả code cell bằng AST;
  không có output hay execution count cũ.
- `inspect-data` đã đọc đủ **14.187** mẫu thật; kiểm tra ID, document/context
  giữa các split, taxonomy và offset.
- Đối chiếu bản export với `artifacts/dataset/responses.jsonl`: mọi record
  bằng nhau, gồm cả `regions` và `source_atoms`. Không đổi split hay annotation.

Chưa chạy full train, tải checkpoint pretrained hoặc thực thi notebook trên GPU Colab.
Lần chạy `--stage all` trên GPU sẽ thực hiện smoke check model trước khi train.
