
from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup, Tag
from playwright.sync_api import (
    BrowserContext,
    Error as PlaywrightError,
    Page,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


CATEGORY_URL = "https://money.vtv.vn/tai-chinh.htm"
BASE_URL = "https://money.vtv.vn"
DOMAIN = "finance"

ARTICLE_PATH_RE = re.compile(
    r"^/(?!video/).+-\d{8,}\.htm$",
    re.IGNORECASE,
)
SPACE_RE = re.compile(r"[ \t\u00a0]+")
MULTI_BLANK_RE = re.compile(r"\n{3,}")
FINGERPRINT_RE = re.compile(r"\W+", re.UNICODE)

STOP_PREFIXES = (
    "bài liên quan",
    "bình luận",
    "cùng chuyên mục",
    "đọc nhiều",
    "theo dõi vtv money",
    "tin liên quan",
    "từ khóa",
    "từ khoá",
)

CONTENT_SELECTORS = (
    '[itemprop="articleBody"]',
    'div[class*="detail__content"]',
    'div[class*="detail-content"]',
    'div[class*="article__content"]',
    'div[class*="article-content"]',
    'div[class*="content-detail"]',
    'div[class*="content_detail"]',
    'div[class*="news-content"]',
    'div[class*="news_content"]',
    'main article',
    'article',
)

SUMMARY_SELECTORS = (
    '[itemprop="description"]',
    'h2[class*="sapo"]',
    'div[class*="sapo"]',
    'h2[class*="lead"]',
    'div[class*="lead"]',
    'h2[class*="description"]',
    '[class*="summary"]',
)

REMOVE_ATTR_FRAGMENTS = (
    "advert",
    "banner",
    "comment",
    "footer",
    "keyword",
    "popular",
    "recommend",
    "related",
    "share",
    "sidebar",
    "social",
    "sponsor",
    "tag",
)


@dataclass(frozen=True)
class ArticleRecord:
    id: str
    domain: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "domain": self.domain,
            "text": self.text,
        }


@dataclass(frozen=True)
class ErrorRecord:
    id: str
    url: str
    error: str

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "url": self.url,
            "error": self.error,
        }


def decode_html_entities(value: str, max_rounds: int = 3) -> str:
    text = value or ""
    for _ in range(max_rounds):
        decoded = html.unescape(text)
        if decoded == text:
            break
        text = decoded
    return text


def normalize_inline(value: str) -> str:
    text = decode_html_entities(value or "")
    text = text.replace("\u200b", "").replace("\ufeff", "")
    return SPACE_RE.sub(" ", text).strip()


def canonicalize_url(raw_url: str) -> str:
    parts = urlsplit(urljoin(BASE_URL, raw_url))
    host = parts.netloc.casefold()
    if host == "www.money.vtv.vn":
        host = "money.vtv.vn"
    path = re.sub(r"/{2,}", "/", parts.path).rstrip("/")
    return urlunsplit(("https", host, path, "", ""))


def is_article_url(url: str) -> bool:
    parsed = urlsplit(url)
    return (
        parsed.netloc.casefold() == "money.vtv.vn"
        and bool(ARTICLE_PATH_RE.match(parsed.path))
    )


def stable_id(url: str) -> str:
    digest = hashlib.blake2b(
        canonicalize_url(url).encode("utf-8"),
        digest_size=8,
    ).hexdigest()
    return f"finance_vtv_money_{digest}"


def extract_listing_urls(html_text: str) -> list[str]:
    """
    Lấy bài thuộc khối Tài chính, dừng trước chuyên mục kế tiếp.

    Sau mỗi lần bấm "Xem thêm", HTML của khối được nối thêm bài nên hàm này
    trả cả các batch đã tải.
    """
    soup = BeautifulSoup(html_text, "lxml")
    root: Tag | BeautifulSoup = soup.find("main") or soup

    urls: list[str] = []
    seen: set[str] = set()
    started = False
    stop_headings = {
        "chứng khoán",
        "thị trường",
        "video",
        "cùng chuyên mục",
    }

    for heading in root.find_all(["h1", "h2", "h3", "h4"]):
        heading_text = normalize_inline(
            heading.get_text(" ", strip=True)
        )
        lowered = heading_text.casefold()

        if not started and lowered == "tài chính":
            started = True
            continue

        if started and urls and lowered in stop_headings:
            break

        if not started:
            continue

        anchor = heading.find("a", href=True)
        if not anchor:
            continue

        url = canonicalize_url(str(anchor["href"]))
        if not is_article_url(url) or url in seen:
            continue

        seen.add(url)
        urls.append(url)

    # Fallback: nếu CMS thay heading, lấy URL bài duy nhất trong main.
    if not urls:
        for anchor in root.select('a[href$=".htm"]'):
            if not isinstance(anchor, Tag):
                continue

            url = canonicalize_url(str(anchor["href"]))
            if is_article_url(url) and url not in seen:
                seen.add(url)
                urls.append(url)

    return urls


class LoadMoreDiscovery:
    def __init__(
        self,
        timeout: float,
        browser_channel: str,
        headed: bool,
        profile_dir: str,
    ) -> None:
        self.timeout_ms = int(timeout * 1000)
        self.browser_channel = browser_channel
        self.headed = headed
        self.profile_dir = str(Path(profile_dir).resolve())

        self._playwright = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None

    def __enter__(self) -> "LoadMoreDiscovery":
        Path(self.profile_dir).mkdir(parents=True, exist_ok=True)
        self._playwright = sync_playwright().start()

        kwargs: dict[str, Any] = {
            "user_data_dir": self.profile_dir,
            "headless": not self.headed,
            "locale": "vi-VN",
            "timezone_id": "Asia/Ho_Chi_Minh",
            "viewport": {"width": 1440, "height": 1000},
            "timeout": self.timeout_ms,
        }
        if self.browser_channel != "chromium":
            kwargs["channel"] = self.browser_channel

        try:
            self.context = (
                self._playwright.chromium.launch_persistent_context(
                    **kwargs
                )
            )
        except PlaywrightError:
            kwargs.pop("channel", None)
            try:
                self.context = (
                    self._playwright.chromium.launch_persistent_context(
                        **kwargs
                    )
                )
            except PlaywrightError as exc:
                self._playwright.stop()
                raise RuntimeError(
                    "Không khởi động được Chrome/Chromium. "
                    "Hãy chạy `python -m playwright install chromium`."
                ) from exc

        self.context.set_default_timeout(self.timeout_ms)
        self.page = (
            self.context.pages[0]
            if self.context.pages
            else self.context.new_page()
        )
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            if self.context is not None:
                self.context.close()
        finally:
            if self._playwright is not None:
                self._playwright.stop()

    def _get_page(self) -> Page:
        if self.page is None:
            raise RuntimeError("Browser chưa được khởi tạo.")
        return self.page

    @staticmethod
    def _visible_load_more(page: Page):
        locator = page.get_by_text("Xem thêm", exact=True)
        for index in range(locator.count()):
            candidate = locator.nth(index)
            try:
                if candidate.is_visible():
                    return candidate
            except PlaywrightError:
                continue
        return None

    def discover_batches(
        self,
        start_page: int,
        pages: int,
    ) -> list[str]:
        """
        Page 1 là batch HTML ban đầu.
        Mỗi lần bấm Xem thêm thành công tạo batch kế tiếp.

        --start-page 2 --pages 10 xử lý batch 2..11.
        """
        page = self._get_page()

        logging.info("Mở danh sách: %s", CATEGORY_URL)
        try:
            response = page.goto(
                CATEGORY_URL,
                wait_until="domcontentloaded",
                timeout=self.timeout_ms,
            )
        except PlaywrightTimeoutError as exc:
            raise RuntimeError(
                f"Timeout khi mở {CATEGORY_URL}"
            ) from exc

        if response is not None and response.status >= 400:
            raise RuntimeError(
                f"Trang danh sách trả HTTP {response.status}."
            )

        try:
            page.locator("main, h1").first.wait_for(
                state="attached",
                timeout=self.timeout_ms,
            )
        except PlaywrightTimeoutError:
            pass

        target_last_page = start_page + pages - 1
        batches: dict[int, list[str]] = {}

        current_urls = extract_listing_urls(page.content())
        if not current_urls:
            raise RuntimeError(
                "Không tìm thấy URL bài ở batch đầu tiên."
            )

        batches[1] = current_urls
        logging.info(
            "[vtv_money] Batch 1: %s URL.",
            len(current_urls),
        )

        current_page_number = 1

        while current_page_number < target_last_page:
            before = current_urls
            before_set = set(before)

            button = self._visible_load_more(page)
            if button is None:
                logging.warning(
                    "Không còn nút Xem thêm tại batch %s.",
                    current_page_number,
                )
                break

            try:
                button.scroll_into_view_if_needed()
                button.click(timeout=self.timeout_ms)
            except PlaywrightError as exc:
                logging.warning(
                    "Không bấm được Xem thêm tại batch %s: %s",
                    current_page_number,
                    exc,
                )
                break

            deadline = time.monotonic() + self.timeout_ms / 1000.0
            grown_urls = before

            while time.monotonic() < deadline:
                page.wait_for_timeout(500)
                candidate_urls = extract_listing_urls(page.content())
                if len(set(candidate_urls) - before_set) > 0:
                    grown_urls = candidate_urls
                    break

            new_urls = [
                url for url in grown_urls
                if url not in before_set
            ]

            if not new_urls:
                logging.warning(
                    "Bấm Xem thêm nhưng không xuất hiện URL mới; "
                    "dừng tại batch %s.",
                    current_page_number,
                )
                break

            current_page_number += 1
            batches[current_page_number] = new_urls
            current_urls = grown_urls

            logging.info(
                "[vtv_money] Batch %s: %s URL mới, "
                "tổng trên DOM=%s.",
                current_page_number,
                len(new_urls),
                len(current_urls),
            )

        selected: list[str] = []
        seen: set[str] = set()

        for page_number in range(
            start_page,
            target_last_page + 1,
        ):
            for url in batches.get(page_number, []):
                if url not in seen:
                    seen.add(url)
                    selected.append(url)

        return selected


def build_session(timeout: float) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=10,
        pool_maxsize=10,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0.0.0 Safari/537.36"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/avif,image/webp,*/*;q=0.8"
            ),
            "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.6",
        }
    )
    session.request_timeout = timeout  # type: ignore[attr-defined]
    return session


def fetch_html(session: requests.Session, url: str) -> str:
    timeout = float(getattr(session, "request_timeout", 30.0))
    response = session.get(url, timeout=timeout)
    if response.status_code in (401, 403):
        raise RuntimeError(f"HTTP {response.status_code}: bị từ chối.")
    response.raise_for_status()

    if not response.encoding or response.encoding.casefold() in {
        "iso-8859-1",
        "latin-1",
    }:
        response.encoding = response.apparent_encoding or "utf-8"

    return response.text


def attrs_text(node: Tag) -> str:
    if node.attrs is None:
        return ""
    classes = node.get("class", [])
    class_text = (
        classes
        if isinstance(classes, str)
        else " ".join(str(value) for value in classes)
    )
    return " ".join(
        (
            str(node.get("id", "")),
            class_text,
            str(node.get("role", "")),
        )
    ).casefold()


def remove_unwanted(node: Tag) -> None:
    for child in reversed(list(node.find_all(True))):
        if child.attrs is None:
            continue
        attr_text = attrs_text(child)
        if any(key in attr_text for key in REMOVE_ATTR_FRAGMENTS):
            child.decompose()
            continue
        if child.name in {
            "script", "style", "noscript", "iframe", "svg", "form"
        }:
            child.decompose()


def should_stop(text: str) -> bool:
    lowered = normalize_inline(text).casefold()
    return any(lowered.startswith(prefix) for prefix in STOP_PREFIXES)


def collect_blocks(node: Tag) -> list[str]:
    clone_soup = BeautifulSoup(str(node), "html.parser")
    clone = clone_soup.find()
    if not isinstance(clone, Tag):
        return []

    remove_unwanted(clone)

    output: list[str] = []
    seen: set[str] = set()

    for element in clone.find_all(
        ["p", "h2", "h3", "blockquote", "figcaption"]
    ):
        text = normalize_inline(element.get_text(" ", strip=True))
        if not text:
            continue
        if should_stop(text):
            break
        if len(text) < 18:
            continue

        fingerprint = FINGERPRINT_RE.sub("", text.casefold())
        if not fingerprint or fingerprint in seen:
            continue
        seen.add(fingerprint)
        output.append(text)

    return output


def extract_title(soup: BeautifulSoup) -> str:
    h1 = soup.find("h1")
    if h1:
        title = normalize_inline(h1.get_text(" ", strip=True))
        if title:
            return title

    meta = soup.find("meta", attrs={"property": "og:title"})
    if meta and meta.get("content"):
        return normalize_inline(str(meta["content"]))

    return ""


def extract_summary(soup: BeautifulSoup, title: str) -> str:
    for selector in SUMMARY_SELECTORS:
        for node in soup.select(selector):
            if not isinstance(node, Tag):
                continue
            text = normalize_inline(node.get_text(" ", strip=True))
            if (
                len(text) >= 35
                and text.casefold() != title.casefold()
                and not should_stop(text)
            ):
                return text
    return ""


def extract_body(soup: BeautifulSoup) -> list[str]:
    for selector in CONTENT_SELECTORS:
        for node in soup.select(selector):
            if not isinstance(node, Tag):
                continue
            blocks = collect_blocks(node)
            if sum(len(block) for block in blocks) >= 100:
                return blocks

    root: Tag | BeautifulSoup = soup.find("main") or soup
    best_blocks: list[str] = []
    best_chars = 0

    for node in root.find_all(["article", "div", "section"]):
        if not isinstance(node, Tag):
            continue
        if len(node.find_all("p")) < 2:
            continue

        blocks = collect_blocks(node)
        chars = sum(len(block) for block in blocks)
        if chars > best_chars:
            best_chars = chars
            best_blocks = blocks

    return best_blocks


def build_text(
    title: str,
    summary: str,
    body: Iterable[str],
) -> str:
    values = [title, summary, *body]
    output: list[str] = []
    seen: set[str] = set()

    for value in values:
        text = normalize_inline(value)
        if not text:
            continue
        fingerprint = FINGERPRINT_RE.sub("", text.casefold())
        if not fingerprint or fingerprint in seen:
            continue
        seen.add(fingerprint)
        output.append(text)

    if not output:
        return ""
    if len(output) == 1:
        return output[0]

    return MULTI_BLANK_RE.sub(
        "\n\n",
        output[0] + "\n\n" + "\n".join(output[1:]),
    ).strip()


def extract_article(
    session: requests.Session,
    url: str,
    min_text_chars: int,
) -> ArticleRecord:
    soup = BeautifulSoup(fetch_html(session, url), "lxml")
    title = extract_title(soup)
    if not title:
        raise RuntimeError("Không tìm thấy tiêu đề.")

    summary = extract_summary(soup, title)
    body = extract_body(soup)
    text = build_text(title, summary, body)

    if len(text) < min_text_chars:
        raise RuntimeError(
            f"Text chỉ có {len(text)} ký tự; "
            f"yêu cầu tối thiểu {min_text_chars} ký tự."
        )

    return ArticleRecord(
        id=stable_id(url),
        domain=DOMAIN,
        text=text,
    )


def load_json_array(
    path: Path,
    required_keys: set[str],
) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError(f"{path} phải chứa JSON array.")

    return [
        record
        for record in data
        if isinstance(record, dict)
        and required_keys.issubset(record)
    ]


def atomic_write_json(
    path: Path,
    records: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            list(records),
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def polite_sleep(delay: float, jitter: float) -> None:
    seconds = max(
        0.0,
        delay + random.uniform(0.0, max(0.0, jitter)),
    )
    if seconds:
        time.sleep(seconds)


def run(args: argparse.Namespace) -> tuple[int, int, int]:
    data_path = Path(args.data_output)
    error_path = Path(args.error_output)

    if args.overwrite:
        article_map: dict[str, dict[str, Any]] = {}
        error_map: dict[str, dict[str, Any]] = {}
    else:
        article_map = {
            str(record["id"]): record
            for record in load_json_array(
                data_path,
                {"id", "domain", "text"},
            )
        }
        error_map = {
            str(record["id"]): record
            for record in load_json_array(
                error_path,
                {"id", "url", "error"},
            )
        }

    with LoadMoreDiscovery(
        timeout=args.timeout,
        browser_channel=args.browser_channel,
        headed=args.headed,
        profile_dir=args.profile_dir,
    ) as discovery:
        urls = discovery.discover_batches(
            start_page=args.start_page,
            pages=args.pages,
        )

    if args.max_articles is not None:
        urls = urls[: args.max_articles]

    logging.info(
        "[vtv_money] Tổng URL thuộc các batch yêu cầu: %s",
        len(urls),
    )

    session = build_session(args.timeout)
    saved = 0
    skipped = 0
    failed = 0
    changes = 0

    try:
        for index, url in enumerate(urls, start=1):
            record_id = stable_id(url)

            if record_id in article_map and not args.refresh:
                skipped += 1
                continue

            try:
                logging.info(
                    "[vtv_money][%s/%s] Crawl: %s",
                    index,
                    len(urls),
                    url,
                )
                article = extract_article(
                    session,
                    url,
                    args.min_text_chars,
                )
                article_map[article.id] = article.to_dict()
                error_map.pop(article.id, None)
                saved += 1
            except Exception as exc:
                error_map[record_id] = ErrorRecord(
                    id=record_id,
                    url=url,
                    error=repr(exc),
                ).to_dict()
                logging.error("Lỗi %s: %r", url, exc)
                failed += 1

            changes += 1
            if changes >= args.checkpoint_every:
                atomic_write_json(
                    data_path,
                    article_map.values(),
                )
                atomic_write_json(
                    error_path,
                    error_map.values(),
                )
                logging.info(
                    "Checkpoint | articles=%s | errors=%s",
                    len(article_map),
                    len(error_map),
                )
                changes = 0

            polite_sleep(args.delay, args.jitter)
    finally:
        atomic_write_json(data_path, article_map.values())
        atomic_write_json(error_path, error_map.values())

    return saved, skipped, failed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Crawl VTV Money bằng nút Xem thêm và lưu JSON data/errors."
        )
    )
    # Giữ --sites để tương thích lệnh cũ.
    parser.add_argument(
        "--sites",
        nargs="+",
        choices=("vtv_money",),
        default=["vtv_money"],
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="Batch bắt đầu; batch 1 là danh sách ban đầu.",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=1,
        help="Số batch cần crawl kể từ --start-page.",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--data-output",
        default="finance_articles.json",
    )
    parser.add_argument(
        "--error-output",
        default="finance_errors.json",
    )
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--browser-channel",
        choices=("chrome", "msedge", "chromium"),
        default="chrome",
    )
    parser.add_argument(
        "--profile-dir",
        default=".vtv_money_browser_profile",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Hiện cửa sổ trình duyệt để kiểm tra nút Xem thêm.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.start_page < 1:
        raise ValueError("--start-page phải >= 1")
    if args.pages < 1:
        raise ValueError("--pages phải >= 1")
    if args.max_articles is not None and args.max_articles < 1:
        raise ValueError("--max-articles phải >= 1")
    if args.min_text_chars < 0:
        raise ValueError("--min-text-chars không được âm")
    if args.delay < 0 or args.jitter < 0:
        raise ValueError("--delay và --jitter không được âm")
    if args.timeout <= 0:
        raise ValueError("--timeout phải > 0")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every phải >= 1")
    if Path(args.data_output).resolve() == Path(
        args.error_output
    ).resolve():
        raise ValueError(
            "File data và error phải khác nhau."
        )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        validate_args(args)
        saved, skipped, failed = run(args)
    except KeyboardInterrupt:
        logging.warning("Đã dừng bởi người dùng.")
        return 130
    except Exception as exc:
        logging.exception("Crawler dừng do lỗi: %s", exc)
        return 1

    logging.info(
        "Hoàn tất | saved=%s | skipped=%s | failed=%s | "
        "data=%s | errors=%s",
        saved,
        skipped,
        failed,
        args.data_output,
        args.error_output,
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())