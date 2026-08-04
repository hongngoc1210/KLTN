from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LIST_URL = "https://sbv.gov.vn/vi/tin-tuc-su-kien"
BASE_URL = "https://sbv.gov.vn"
DOMAIN = "finance"
ID_PREFIX = "finance_sbv_"
PAGE_SIZE = 12
PORTLET_ID = "com_liferay_asset_publisher_web_portlet_AssetPublisherPortlet_INSTANCE_jaxi"
PORTLET_NS = "_com_liferay_asset_publisher_web_portlet_AssetPublisherPortlet_INSTANCE_jaxi_"

DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b")
SPACE_RE = re.compile(r"[ \t\r\f\v]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

STOP_MARKERS = (
    "các tin khác",
    "trang bình luận",
    "thêm các bình luận",
    "các chuyên mục:",
)

SKIP_EXACT = {
    "aA",
    "Giọng Nam",
    "Giọng Nữ",
    "Trả lời",
    "0:00",
    "/",
}

_thread_local = threading.local()


def clean_inline(text: str) -> str:
    text = text.replace("\xa0", " ").replace("\u200b", " ")
    return SPACE_RE.sub(" ", text).strip()


def clean_multiline(text: str) -> str:
    lines = [clean_inline(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    return MULTI_NEWLINE_RE.sub("\n\n", "\n".join(lines)).strip()


def canonicalize_url(url: str) -> str:
    """Bỏ query/fragment để cùng một bài luôn có cùng ID."""
    absolute = urljoin(BASE_URL, url)
    parts = urlsplit(absolute)
    path = re.sub(r"/{2,}", "/", parts.path).rstrip("/")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, "", ""))


def make_document_id(url: str) -> str:
    canonical = canonicalize_url(url)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
    return f"{ID_PREFIX}{digest}"


def create_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)

    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/150.0 Safari/537.36"
            ),
            "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.7",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    )
    return session


def get_thread_session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = create_session()
        _thread_local.session = session
    return session


def fetch_html(url: str, *, timeout: int, params: dict[str, object] | None = None) -> str:
    session = get_thread_session()
    response = session.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"
    return response.text


def build_list_params(page: int) -> dict[str, object]:
    return {
        "p_p_id": PORTLET_ID,
        "p_p_lifecycle": 0,
        "p_p_state": "normal",
        "p_p_mode": "view",
        f"{PORTLET_NS}delta": PAGE_SIZE,
        f"{PORTLET_NS}cur": page,
        f"{PORTLET_NS}redirect": "/vi/tin-tuc-su-kien",
        "p_r_p_resetCur": "false",
    }


def nearby_text_has_date(anchor: Tag, max_levels: int = 5) -> bool:
    node: Tag | None = anchor
    for _ in range(max_levels):
        parent = node.parent if node is not None else None
        if not isinstance(parent, Tag):
            break
        text = clean_inline(parent.get_text(" ", strip=True))
        if DATE_RE.search(text):
            return True
        if len(text) > 2500:
            break
        node = parent
    return False


def extract_article_urls(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")

    # Giới hạn vào vùng nội dung chính để tránh menu/header/footer.
    roots: list[Tag] = []
    for selector in (
        "main",
        "#content",
        ".portlet-content-container",
        ".portlet-body",
        ".asset-publisher-web",
    ):
        roots.extend(soup.select(selector))

    if not roots:
        roots = [soup]

    urls: list[str] = []
    seen: set[str] = set()

    for root in roots:
        for anchor in root.select('a[href*="/vi/w/"]'):
            href = anchor.get("href")
            title = clean_inline(anchor.get_text(" ", strip=True))
            if not href or len(title) < 12:
                continue

            canonical = canonicalize_url(href)
            parts = urlsplit(canonical)
            if parts.netloc != "sbv.gov.vn" or not parts.path.startswith("/vi/w/"):
                continue

            # Trên trang danh sách, ngày đăng nằm gần liên kết tiêu đề.
            # Điều kiện này loại phần lớn liên kết menu và chuyên mục tĩnh.
            if not nearby_text_has_date(anchor):
                continue

            if canonical not in seen:
                seen.add(canonical)
                urls.append(canonical)

    return urls


def remove_noise(soup: BeautifulSoup) -> None:
    for selector in (
        "script",
        "style",
        "noscript",
        "svg",
        "nav",
        "header",
        "footer",
        "aside",
        "form",
        ".social-share",
        ".taglib-social-bookmarks",
        ".ratings",
        ".comments",
        ".lfr-discussion",
        ".related-news",
        ".other-news",
    ):
        for node in soup.select(selector):
            node.decompose()


def find_title(soup: BeautifulSoup) -> tuple[str, Tag | None]:
    selectors = (
        "main h1",
        ".portlet-content h1",
        ".asset-full-content h1",
        "article h1",
        "h1",
    )
    for selector in selectors:
        node = soup.select_one(selector)
        if node:
            title = clean_inline(node.get_text(" ", strip=True))
            if len(title) >= 8:
                return title, node
    return "", None


def select_article_root(soup: BeautifulSoup, title_node: Tag | None) -> Tag | None:
    # Ưu tiên ancestor nhỏ nhất của H1 chứa đủ phần thân bài.
    if title_node is not None:
        current: Tag | None = title_node.parent if isinstance(title_node.parent, Tag) else None
        for _ in range(9):
            if current is None or current.name in {"body", "html"}:
                break
            paragraphs = [
                clean_inline(p.get_text(" ", strip=True))
                for p in current.find_all("p")
            ]
            useful = [p for p in paragraphs if len(p) >= 40]
            total_length = sum(len(p) for p in useful)
            if len(useful) >= 2 and total_length >= 500:
                return current
            current = current.parent if isinstance(current.parent, Tag) else None

    # Fallback cho các template khác nhau của Liferay.
    candidates: list[Tag] = []
    for selector in (
        "article",
        ".asset-full-content",
        ".journal-content-article",
        ".article-content",
        ".news-detail",
        ".detail-content",
        ".entry-content",
        "main",
    ):
        candidates.extend(soup.select(selector))

    best: Tag | None = None
    best_score = -1
    for node in candidates:
        paragraphs = [
            clean_inline(p.get_text(" ", strip=True))
            for p in node.find_all("p")
        ]
        useful = [p for p in paragraphs if len(p) >= 40]
        score = sum(len(p) for p in useful) + 150 * len(useful)
        node_text = clean_inline(node.get_text(" ", strip=True)).lower()
        if "các tin khác" in node_text:
            score -= 2500
        if score > best_score:
            best_score = score
            best = node
    return best


def is_stop_text(text: str) -> bool:
    lowered = text.casefold()
    return any(marker in lowered for marker in STOP_MARKERS)


def extract_context(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    remove_noise(soup)

    title, title_node = find_title(soup)
    root = select_article_root(soup, title_node)
    if root is None:
        return ""

    pieces: list[str] = []
    seen: set[str] = set()

    # Chỉ lấy các đơn vị văn bản có ý nghĩa; không lấy text toàn trang.
    for node in root.find_all(["h2", "h3", "h4", "h5", "p", "li"]):
        text = clean_inline(node.get_text(" ", strip=True))
        if not text:
            continue
        if is_stop_text(text):
            break
        if text in SKIP_EXACT or text == title:
            continue
        if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}(?:\s+\d{1,2}:\d{2}:\d{2})?", text):
            continue
        if len(text) < 2:
            continue
        if text not in seen:
            seen.add(text)
            pieces.append(text)

    body = clean_multiline("\n\n".join(pieces))

    # Một số bài cũ không dùng thẻ <p>; fallback lấy các dòng text trong root.
    if len(body) < 250:
        fallback_lines: list[str] = []
        for raw_line in root.get_text("\n", strip=True).splitlines():
            line = clean_inline(raw_line)
            if not line or line == title or line in SKIP_EXACT:
                continue
            if is_stop_text(line):
                break
            if DATE_RE.fullmatch(line):
                continue
            if line not in fallback_lines:
                fallback_lines.append(line)
        body = clean_multiline("\n".join(fallback_lines))

    if title and body:
        return f"{title}\n\n{body}".strip()
    return (title or body).strip()


def read_existing_ids(output_path: Path) -> set[str]:
    ids: set[str] = set()
    if not output_path.exists():
        return ids

    with output_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                item_id = item.get("id")
                if isinstance(item_id, str):
                    ids.add(item_id)
            except json.JSONDecodeError:
                print(f"Cảnh báo: bỏ qua JSON lỗi tại dòng {line_number}.")
    return ids


def crawl_detail(
    url: str,
    *,
    timeout: int,
    min_chars: int,
    delay: float,
) -> tuple[str, dict[str, str] | None, str | None]:
    document_id = make_document_id(url)
    try:
        if delay > 0:
            time.sleep(delay + random.uniform(0, min(0.4, delay / 2)))
        html = fetch_html(url, timeout=timeout)
        context = extract_context(html)
        if len(context) < min_chars:
            return url, None, f"context quá ngắn ({len(context)} ký tự)"
        return (
            url,
            {
                "id": document_id,
                "domain": DOMAIN,
                "context": context,
            },
            None,
        )
    except Exception as exc:  # Tiếp tục crawl các bài khác nếu một bài lỗi.
        return url, None, f"{type(exc).__name__}: {exc}"


def chunked(items: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def crawl(args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_ids = read_existing_ids(output_path)
    print(f"Đã có {len(existing_ids)} bản ghi trong file output.")

    seen_urls: set[str] = set()
    saved = 0
    page = args.start_page
    pages_processed = 0

    with output_path.open("a", encoding="utf-8") as output_file:
        while True:
            if args.max_pages > 0 and pages_processed >= args.max_pages:
                break

            try:
                params = build_list_params(page)
                html = fetch_html(LIST_URL, params=params, timeout=args.timeout)
                urls = extract_article_urls(html)
            except Exception as exc:
                print(f"Trang {page}: lỗi tải danh sách: {type(exc).__name__}: {exc}")
                break

            new_page_urls = [url for url in urls if url not in seen_urls]
            if not new_page_urls:
                print(f"Trang {page}: không tìm thấy URL mới, dừng phân trang.")
                break

            seen_urls.update(new_page_urls)
            pending_urls = [
                url for url in new_page_urls if make_document_id(url) not in existing_ids
            ]

            print(
                f"Trang {page}: {len(urls)} URL, "
                f"mới {len(new_page_urls)}, cần tải {len(pending_urls)}."
            )

            for batch in chunked(pending_urls, args.batch_size):
                worker = lambda url: crawl_detail(
                    url,
                    timeout=args.timeout,
                    min_chars=args.min_chars,
                    delay=args.delay,
                )

                if args.workers == 1:
                    results = map(worker, batch)
                else:
                    executor = ThreadPoolExecutor(max_workers=args.workers)
                    results = executor.map(worker, batch)

                try:
                    for url, item, error in results:
                        document_id = make_document_id(url)
                        if error:
                            print(f"  SKIP {document_id}: {error}")
                            continue

                        assert item is not None
                        output_file.write(json.dumps(item, ensure_ascii=False) + "\n")
                        output_file.flush()
                        existing_ids.add(item["id"])
                        saved += 1
                        print(f"  SAVE {item['id']} ({len(item['context'])} ký tự)")
                finally:
                    if args.workers != 1:
                        executor.shutdown(wait=True)

            pages_processed += 1
            page += 1

            if args.page_delay > 0:
                time.sleep(args.page_delay)

    print(
        f"Hoàn tất. Đã lưu mới {saved} bản ghi. "
        f"Tổng số ID hiện có: {len(existing_ids)}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crawl mục Tin tức - Sự kiện của Ngân hàng Nhà nước Việt Nam."
    )
    parser.add_argument(
        "--output",
        default="data/finance_sbv_news.jsonl",
        help="Đường dẫn file JSONL output.",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="Trang danh sách bắt đầu crawl.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=5,
        help="Số trang tối đa; đặt 0 để crawl đến khi hết trang.",
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--timeout", type=int, default=40)
    parser.add_argument(
        "--delay",
        type=float,
        default=0.8,
        help="Khoảng nghỉ trước mỗi request bài chi tiết.",
    )
    parser.add_argument(
        "--page-delay",
        type=float,
        default=1.0,
        help="Khoảng nghỉ giữa hai trang danh sách.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=250,
        help="Bỏ qua bài có context ngắn hơn giá trị này.",
    )
    args = parser.parse_args()

    if args.start_page < 1:
        parser.error("--start-page phải >= 1")
    if args.max_pages < 0:
        parser.error("--max-pages phải >= 0")
    if args.workers < 1:
        parser.error("--workers phải >= 1")
    if args.batch_size < 1:
        parser.error("--batch-size phải >= 1")
    return args


if __name__ == "__main__":
    crawl(parse_args())