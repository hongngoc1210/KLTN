# Multi-University Education Crawler

Crawler chung cho nhiều website tuyển sinh đại học. Output chính chỉ gồm:

```json
{"id": "education_uit_...", "domain": "education", "text": "..."}
```

## Cài đặt

```powershell
python -m pip install -r requirements.txt
python -m playwright install --no-shell chromium
```

Playwright chỉ được sử dụng khi một site trong YAML có:

```yaml
renderer: playwright
```

Các cấu hình pilot hiện dùng `requests`.

## Chạy thử từng trường

```powershell
python crawl_universities.py --config universities.yaml --sites uit --max-pages 1 --max-docs 5
python crawl_universities.py --config universities.yaml --sites ptit --max-pages 1 --max-docs 5
```

## Chạy một nhóm trường

```powershell
python crawl_universities.py --config universities.yaml --sites uit,ptit,vnu,hust --max-pages 3 --max-docs 50
```

`--max-docs 50` áp dụng riêng cho từng trường.

## Chạy tất cả cấu hình

```powershell
python crawl_universities.py --config universities.yaml --sites all --max-pages 0 --max-docs 0
```

Không nên chạy toàn bộ ngay từ đầu. Hãy kiểm thử 5–10 tài liệu trên từng website và xem error log trước.

## Output

```text
data/raw/education_universities.jsonl
data/raw/education_universities_errors.jsonl
```

Crawler tự đọc các `id` đã lưu và bỏ qua khi chạy lại.

## Thêm trường mới

Thêm một block vào `universities.yaml`:

```yaml
  - key: abc
    name: Trường Đại học ABC
    base_url: https://tuyensinh.abc.edu.vn
    start_urls:
      - https://tuyensinh.abc.edu.vn/thong-bao
    id_prefix: education_abc
    renderer: requests
    detail_patterns:
      - '^https://tuyensinh\.abc\.edu\.vn/thong-bao/[^/?#]+'
    pagination_patterns:
      - '^https://tuyensinh\.abc\.edu\.vn/thong-bao/page/\d+'
    list_link_selectors:
      - 'main'
      - '.posts'
    content_selectors:
      - '.entry-content'
      - 'article'
    attachment_patterns:
      - '\.(pdf|docx?)(?:$|[?&#])'
    max_pages: 20
    max_docs: 500
    min_chars: 300
```

### `detail_patterns`

Phải chỉ khớp URL bài viết, không khớp menu, trang chủ, trang tag hoặc file.

### `pagination_patterns`

Chỉ khớp các trang danh sách tiếp theo. Crawler dùng BFS nên không phụ thuộc trang dùng `?page=2`, `/page/2/` hay kiểu URL khác.

### `content_selectors`

Crawler thử các selector và lấy vùng có text dài nhất. Khi output bị dính menu/footer, hãy thêm selector cụ thể hơn thay vì bật crawl toàn body.

### `renderer`

- `requests`: HTML server-rendered.
- `playwright`: trang render JavaScript hoặc cần giữ cookie.

Chạy Playwright có giao diện:

```powershell
python crawl_universities.py --sites abc --headed
```

Profile được lưu tại:

```text
data/browser_profiles/universities/<site-key>/
```

## Lưu ý dữ liệu

- PDF scan không được OCR tự động.
- `.doc` cũ không được đọc bằng `python-docx`.
- Error log có URL để debug, nhưng output chính chỉ có `id`, `domain`, `text`.
- Nên loại dữ liệu cá nhân, danh sách sinh viên và số điện thoại trước khi phát hành dataset.
- Luôn kiểm tra robots.txt và điều khoản của từng website.
