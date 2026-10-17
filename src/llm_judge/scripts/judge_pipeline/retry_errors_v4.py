# -*- coding: utf-8 -*-

"""Retry đúng một lần các QA lỗi đã lưu bởi main_v4.py.

Chạy với cùng --input và --output như lượt chính. Các QA thành công và QA lỗi
đã retry trước đó sẽ không gọi API lần nữa.
"""

from main_v4 import main


if __name__ == "__main__":
    main(retry_errors_only=True)
