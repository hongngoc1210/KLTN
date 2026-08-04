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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup, Tag
from playwright.sync_api import (
    BrowserContext,
    Error as PlaywrightError,
    Locator,
    Page,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DOMAIN = "finance"

SITE_URLS = {
    "vneconomy": "https://vneconomy.vn/chung-khoan.htm",
    "cafef": "https://cafef.vn/tai-chinh-quoc-te.chn",
    "cafef_stock": "https://cafef.vn/thi-truong-chung-khoan.chn",
}

SPACE_RE = re.compile(r"[ \t\u00a0]+")
MULTI_BLANK_RE = re.compile(r"\n{3,}")
FINGERPRINT_RE = re.compile(r"\W+", re.UNICODE)

ARTICLE_TYPES = {
    "article",
    "newsarticle",
    "reportagenewsarticle",
    "analysisnewsarticle",
    "blogposting",
}

CONTENT_SELECTORS_COMMON = (
    '[itemprop="articleBody"]',
    '[data-role="content"]',
    '[data-role="article-content"]',
    '[data-testid="article-content"]',
    'div[class*="article-body"]',
    'div[class*="article_body"]',
    'div[class*="article-content"]',
    'div[class*="article_content"]',
    'div[class*="content-detail"]',
    'div[class*="content_detail"]',
    'div[class*="detail-content"]',
    'div[class*="detail_content"]',
    'div[class*="news-content"]',
    'div[class*="news_content"]',
    'article',
)

SUMMARY_SELECTORS_COMMON = (
    '[itemprop="description"]',
    '[data-role="sapo"]',
    '[data-role="summary"]',
    '[class*="sapo"]',
    '[class*="summary"]',
    '[class*="lead"]',
    '[class*="description"]',
)

REMOVE_ATTR_FRAGMENTS = (
    "advert",
    "banner",
    "comment",
    "footer",
    "keyword",
    "most-read",
    "mostread",
    "popular",
    "recommend",
    "related",
    "share",
    "sidebar",
    "social",
    "sponsor",
    "tag",
)

STOP_PREFIXES = (
    "bài liên quan",
    "cùng chuyên mục",
    "đọc bài theo từ khoá",
    "đọc bài theo từ khóa",
    "đọc nhiều",
    "đọc nhiều nhất",
    "đọc tiếp",
    "đừng bỏ lỡ",
    "link bài gốc",
    "mới nhất trên",
    "nội dung được tài trợ",
    "tham khảo",
    "tin liên quan",
    "từ khóa",
    "từ khoá",
)

SKIP_EXACT = {
    "chia sẻ",
    "copy link",
    "lấy link",
    "tin mới",
    "mới nhất",
    "xem thêm",
    "đọc ngay",
}

VNECONOMY_STOP_HEADINGS = {
    "mới nhất trên vneconomy",
    "nổi bật trên vneconomy",
    "ấn phẩm",
    "multimedia",
    "đọc nhiều nhất",
    "ai điểm tin",
    "dòng sự kiện",
}

CAFEF_STOP_TEXTS = {
    "công ty tin tức lãnh đạo",
    "cùng chuyên mục",
}


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


def decode_html_entities(text: str, max_rounds: int = 3) -> str:
    """
    Giải mã HTML entities, kể cả trường hợp bị encode nhiều lần.

    Ví dụ:
    - &#x1EA5;       -> ấ
    - &amp;#x1EA5;  -> ấ
    """
    value = text or ""

    for _ in range(max_rounds):
        decoded = html.unescape(value)
        if decoded == value:
            break
        value = decoded

    return value


def normalize_inline(text: str) -> str:
    text = decode_html_entities(text or "")
    text = text.replace("\u200b", "").replace("\ufeff", "")
    return SPACE_RE.sub(" ", text).strip()


def normalize_multiline(text: str) -> str:
    lines = [normalize_inline(line) for line in (text or "").replace("\r", "\n").split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def canonicalize_url(raw_url: str, base_url: str) -> str:
    parts = urlsplit(urljoin(base_url, raw_url))
    scheme = "https"
    host = parts.netloc.lower()
    path = re.sub(r"/{2,}", "/", parts.path).rstrip("/")
    return urlunsplit((scheme, host, path, "", ""))


def stable_id(site: str, url: str) -> str:
    digest = hashlib.blake2b(
        canonicalize_url(url, SITE_URLS[site]).encode("utf-8"),
        digest_size=8,
    ).hexdigest()
    return f"finance_{site}_{digest}"


def iter_json_objects(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from iter_json_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_json_objects(child)


def jsonld_types(obj: dict[str, Any]) -> set[str]:
    raw_type = obj.get("@type", "")
    if isinstance(raw_type, str):
        return {raw_type.casefold()}
    if isinstance(raw_type, list):
        return {str(item).casefold() for item in raw_type}
    return set()


def clean_jsonld_body(raw_body: str) -> str:
    if "<" in raw_body and ">" in raw_body:
        soup = BeautifulSoup(raw_body, "html.parser")
        return normalize_multiline(soup.get_text("\n"))
    return normalize_multiline(raw_body)


def extract_jsonld_article(
    soup: BeautifulSoup,
) -> tuple[str, str, str] | None:
    """
    Trả về (title, summary, body) nếu JSON-LD có NewsArticle/Article.
    """
    best: tuple[int, str, str, str] | None = None

    for script in soup.select('script[type="application/ld+json"]'):
        raw = script.string or script.get_text()
        if not raw.strip():
            continue

        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue

        for obj in iter_json_objects(data):
            if not (jsonld_types(obj) & ARTICLE_TYPES):
                continue

            title = normalize_inline(
                str(obj.get("headline") or obj.get("name") or "")
            )
            summary = normalize_inline(str(obj.get("description") or ""))
            body = clean_jsonld_body(str(obj.get("articleBody") or ""))

            score = len(body) * 10 + len(summary) + len(title)
            if title and (best is None or score > best[0]):
                best = (score, title, summary, body)

    if best is None:
        return None

    return best[1], best[2], best[3]


def attrs_text(node: Tag) -> str:
    if node.attrs is None:
        return ""

    classes = node.get("class", [])
    if isinstance(classes, str):
        class_text = classes
    else:
        class_text = " ".join(str(value) for value in classes)

    return " ".join(
        (
            str(node.get("id", "")),
            class_text,
            str(node.get("role", "")),
            str(node.get("data-role", "")),
        )
    ).casefold()


def remove_unwanted_descendants(node: Tag) -> None:
    # Duyệt từ node sâu nhất lên trên. Nếu xóa container cha trước,
    # BeautifulSoup sẽ làm vô hiệu hóa các Tag con đã nằm trong danh sách.
    for child in reversed(list(node.find_all(True))):
        if child.attrs is None:
            continue

        attr_text = attrs_text(child)
        if any(fragment in attr_text for fragment in REMOVE_ATTR_FRAGMENTS):
            child.decompose()
            continue

        if child.name in {"script", "style", "noscript", "iframe", "svg", "form"}:
            child.decompose()


def should_stop_text(text: str) -> bool:
    lowered = normalize_inline(text).casefold()
    return any(lowered.startswith(prefix) for prefix in STOP_PREFIXES)


def should_skip_text(text: str) -> bool:
    normalized = normalize_inline(text)
    lowered = normalized.casefold()

    if not normalized:
        return True
    if lowered in SKIP_EXACT:
        return True
    if should_stop_text(normalized):
        return True
    if len(normalized) < 18:
        return True

    return False


def collect_text_blocks(node: Tag) -> list[str]:
    """
    Lấy các đoạn văn theo thứ tự và dừng trước phần liên quan/từ khóa.
    """
    clone_soup = BeautifulSoup(str(node), "html.parser")
    clone = clone_soup.find()
    if not isinstance(clone, Tag):
        return []

    remove_unwanted_descendants(clone)

    blocks: list[str] = []
    seen: set[str] = set()

    elements = clone.find_all(
        ["p", "h2", "h3", "blockquote", "figcaption"],
        recursive=True,
    )

    for element in elements:
        text = normalize_inline(element.get_text(" ", strip=True))
        if not text:
            continue

        if should_stop_text(text):
            break
        if should_skip_text(text):
            continue

        fingerprint = FINGERPRINT_RE.sub("", text.casefold())
        if not fingerprint or fingerprint in seen:
            continue

        seen.add(fingerprint)
        blocks.append(text)

    # Một số CMS để nội dung trực tiếp trong các div, không có thẻ p.
    if not blocks:
        for raw_line in clone.get_text("\n").splitlines():
            text = normalize_inline(raw_line)
            if should_stop_text(text):
                break
            if should_skip_text(text):
                continue

            fingerprint = FINGERPRINT_RE.sub("", text.casefold())
            if not fingerprint or fingerprint in seen:
                continue

            seen.add(fingerprint)
            blocks.append(text)

    return blocks


def candidate_score(node: Tag) -> tuple[float, list[str]]:
    blocks = collect_text_blocks(node)
    if not blocks:
        return -1.0, []

    chars = sum(len(block) for block in blocks)
    attr_text = attrs_text(node)
    p_count = len(node.find_all("p"))

    score = float(chars) + p_count * 75.0

    if any(
        key in attr_text
        for key in ("article", "content", "detail", "body", "news")
    ):
        score += 1500.0

    if any(
        key in attr_text
        for key in (
            "comment",
            "footer",
            "header",
            "latest",
            "popular",
            "recommend",
            "related",
            "sidebar",
        )
    ):
        score -= 3500.0

    # Container quá rộng thường chứa cả menu/footer.
    total_text = len(normalize_inline(node.get_text(" ", strip=True)))
    score -= min(max(0, total_text - chars) * 0.06, 2500.0)

    return score, blocks


def find_body_blocks(
    soup: BeautifulSoup,
    selectors: Sequence[str],
) -> list[str]:
    # Ưu tiên selector rõ ràng.
    for selector in selectors:
        for node in soup.select(selector):
            if not isinstance(node, Tag):
                continue

            blocks = collect_text_blocks(node)
            if sum(len(block) for block in blocks) >= 100:
                return blocks

    # Fallback: chấm điểm container.
    root: Tag | BeautifulSoup = soup.find("main") or soup
    best_score = -1.0
    best_blocks: list[str] = []

    for node in root.find_all(["article", "div", "section"]):
        if not isinstance(node, Tag):
            continue

        p_count = len(node.find_all("p"))
        if not (2 <= p_count <= 200):
            continue

        score, blocks = candidate_score(node)
        if score > best_score:
            best_score = score
            best_blocks = blocks

    return best_blocks


def extract_title(soup: BeautifulSoup) -> str:
    h1 = soup.find("h1")
    if h1:
        title = normalize_inline(h1.get_text(" ", strip=True))
        if title:
            return title

    for attrs in (
        {"property": "og:title"},
        {"name": "twitter:title"},
    ):
        meta = soup.find("meta", attrs=attrs)
        if meta and meta.get("content"):
            return normalize_inline(str(meta["content"]))

    return ""


def extract_summary(
    soup: BeautifulSoup,
    selectors: Sequence[str],
    title: str,
) -> str:
    for selector in selectors:
        for node in soup.select(selector):
            if not isinstance(node, Tag):
                continue

            text = normalize_inline(node.get_text(" ", strip=True))
            if (
                len(text) >= 35
                and text.casefold() != title.casefold()
                and not should_stop_text(text)
            ):
                return text

    # Fallback: đoạn h4 đầu tiên gần vùng tiêu đề, phổ biến ở VnEconomy.
    h1 = soup.find("h1")
    if h1:
        for node in h1.find_all_next(["h4", "p"], limit=8):
            text = normalize_inline(node.get_text(" ", strip=True))
            if (
                35 <= len(text) <= 1000
                and text.casefold() != title.casefold()
                and not should_stop_text(text)
            ):
                return text

    return ""


def build_article_text(
    title: str,
    summary: str,
    body_blocks: Iterable[str],
) -> str:
    title = normalize_inline(title)
    summary = normalize_inline(summary)

    output: list[str] = []
    seen: set[str] = set()

    for text in [title, summary, *body_blocks]:
        text = normalize_inline(text)
        if not text:
            continue

        fingerprint = FINGERPRINT_RE.sub("", text.casefold())
        if not fingerprint or fingerprint in seen:
            continue

        seen.add(fingerprint)
        output.append(text)

    if not output:
        return ""

    # Tiêu đề cách nội dung một dòng trống; các đoạn thân cách nhau một dòng.
    if len(output) == 1:
        return output[0]

    return MULTI_BLANK_RE.sub(
        "\n\n",
        output[0] + "\n\n" + "\n".join(output[1:]),
    ).strip()


def build_session(
    timeout: float,
    user_agent: str | None = None,
) -> requests.Session:
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
            "User-Agent": user_agent
            or (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0.0.0 Safari/537.36"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/avif,image/webp,*/*;q=0.8"
            ),
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.7,en;q=0.6",
            "Cache-Control": "no-cache",
        }
    )
    session.request_timeout = timeout  # type: ignore[attr-defined]
    return session


def fetch_html(session: requests.Session, url: str) -> str:
    timeout = float(getattr(session, "request_timeout", 30.0))
    response = session.get(url, timeout=timeout)

    if response.status_code in (401, 403):
        raise RuntimeError(
            f"HTTP {response.status_code}: website từ chối truy cập."
        )

    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if "html" not in content_type.casefold() and not response.text.lstrip().startswith("<"):
        raise RuntimeError(
            f"Response không phải HTML: Content-Type={content_type!r}."
        )

    if not response.encoding or response.encoding.casefold() in {
        "iso-8859-1",
        "latin-1",
    }:
        response.encoding = response.apparent_encoding or "utf-8"

    return response.text


def polite_sleep(delay: float, jitter: float) -> None:
    seconds = max(0.0, delay + random.uniform(0.0, max(0.0, jitter)))
    if seconds:
        time.sleep(seconds)


class SiteCrawler(ABC):
    name: str
    base_url: str
    category_url: str
    content_selectors: Sequence[str] = CONTENT_SELECTORS_COMMON
    summary_selectors: Sequence[str] = SUMMARY_SELECTORS_COMMON

    def __init__(
        self,
        session: requests.Session,
        min_text_chars: int,
    ) -> None:
        self.session = session
        self.min_text_chars = min_text_chars

    @abstractmethod
    def list_page_candidates(self, page_number: int) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def is_article_url(self, url: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def extract_urls_from_listing(self, html: str) -> list[str]:
        raise NotImplementedError

    def record_id(self, url: str) -> str:
        return stable_id(self.name, url)

    def discover_page(self, page_number: int) -> tuple[str, list[str]]:
        """
        Một số website đổi kiểu phân trang. Thử lần lượt các URL ứng viên
        và chọn URL đầu tiên trả về danh sách bài hợp lệ.
        """
        errors: list[str] = []

        for page_url in self.list_page_candidates(page_number):
            try:
                logging.info("[%s] Đọc danh sách: %s", self.name, page_url)
                html = fetch_html(self.session, page_url)
                urls = self.extract_urls_from_listing(html)

                if urls:
                    return page_url, urls

                errors.append(f"{page_url}: không tìm thấy URL bài")
            except Exception as exc:
                errors.append(f"{page_url}: {exc!r}")

        raise RuntimeError(
            "Không đọc được trang danh sách. " + " | ".join(errors)
        )

    def extract_article(self, url: str) -> ArticleRecord:
        html = fetch_html(self.session, url)
        soup = BeautifulSoup(html, "lxml")

        jsonld = extract_jsonld_article(soup)

        if jsonld is not None:
            jsonld_title, jsonld_summary, jsonld_body = jsonld
        else:
            jsonld_title = ""
            jsonld_summary = ""
            jsonld_body = ""

        title = jsonld_title or extract_title(soup)
        if not title:
            raise RuntimeError("Không tìm thấy tiêu đề bài viết.")

        summary = jsonld_summary or extract_summary(
            soup,
            self.summary_selectors,
            title,
        )

        if jsonld_body:
            body_blocks = [
                normalize_inline(line)
                for line in jsonld_body.splitlines()
                if normalize_inline(line)
            ]
        else:
            body_blocks = find_body_blocks(
                soup,
                self.content_selectors,
            )

        text = build_article_text(title, summary, body_blocks)

        if len(text) < self.min_text_chars:
            raise RuntimeError(
                f"Text chỉ có {len(text)} ký tự; "
                f"yêu cầu tối thiểu {self.min_text_chars} ký tự."
            )

        return ArticleRecord(
            id=self.record_id(url),
            domain=DOMAIN,
            text=text,
        )


class VnEconomyCrawler(SiteCrawler):
    name = "vneconomy"
    base_url = "https://vneconomy.vn"
    category_url = "https://vneconomy.vn/chung-khoan.htm"

    content_selectors = (
        '[itemprop="articleBody"]',
        'div[class*="detail__content"]',
        'div[class*="detail-content"]',
        'div[class*="article__content"]',
        'div[class*="article-content"]',
        'div[class*="content-detail"]',
        *CONTENT_SELECTORS_COMMON,
    )

    summary_selectors = (
        'h4[class*="detail"]',
        'div[class*="detail__summary"]',
        'div[class*="article__sapo"]',
        *SUMMARY_SELECTORS_COMMON,
    )

    EXCLUDED_PATHS = {
        "",
        "/chung-khoan.htm",
        "/tai-chinh.htm",
        "/the-gioi.htm",
        "/thi-truong.htm",
        "/dau-tu.htm",
    }

    def list_page_candidates(self, page_number: int) -> list[str]:
        if page_number <= 1:
            return [self.category_url]

        index = page_number - 1
        return [
            f"{self.category_url}?page={index}",
            f"{self.category_url}?trang={index}",
        ]

    def is_article_url(self, url: str) -> bool:
        parsed = urlsplit(url)
        path = parsed.path.rstrip("/").casefold()

        if parsed.netloc.casefold() not in {"vneconomy.vn", "www.vneconomy.vn"}:
            return False
        if path in self.EXCLUDED_PATHS:
            return False
        if not path.endswith(".htm"):
            return False
        if path.count("/") != 1:
            return False
        if len(path) < 12:
            return False

        return True

    def extract_urls_from_listing(self, html: str) -> list[str]:
        soup = BeautifulSoup(html, "lxml")
        root: Tag | BeautifulSoup = soup.find("main") or soup

        urls: list[str] = []
        seen: set[str] = set()
        started = False

        for heading in root.find_all(["h1", "h2", "h3", "h4"]):
            text = normalize_inline(heading.get_text(" ", strip=True))
            lowered = text.casefold()

            if not started and "chứng khoán" in lowered:
                started = True

            if (
                started
                and urls
                and lowered in VNECONOMY_STOP_HEADINGS
            ):
                break

            anchor = heading.find("a", href=True)
            if not anchor:
                continue

            url = canonicalize_url(str(anchor["href"]), self.base_url)
            if not self.is_article_url(url):
                continue
            if url in seen:
                continue

            seen.add(url)
            urls.append(url)

        # Fallback khi CMS đổi heading.
        if not urls:
            for anchor in root.select('a[href$=".htm"]'):
                if not isinstance(anchor, Tag):
                    continue

                title = normalize_inline(anchor.get_text(" ", strip=True))
                if len(title) < 15:
                    continue

                url = canonicalize_url(str(anchor["href"]), self.base_url)
                if self.is_article_url(url) and url not in seen:
                    seen.add(url)
                    urls.append(url)

        return urls


class CafeFCrawler(SiteCrawler):
    name = "cafef"
    base_url = "https://cafef.vn"
    category_slug = "tai-chinh-quoc-te"
    category_heading = "tài chính quốc tế"
    category_url = "https://cafef.vn/tai-chinh-quoc-te.chn"

    content_selectors = (
        '[itemprop="articleBody"]',
        'div[class*="contentdetail"]',
        'div[class*="content-detail"]',
        'div[class*="detail-content"]',
        'div[class*="news-content"]',
        'div[class*="article-content"]',
        '#mainContent',
        *CONTENT_SELECTORS_COMMON,
    )

    summary_selectors = (
        '[class*="knc-sapo"]',
        '[class*="detail-sapo"]',
        '[class*="sapo"]',
        *SUMMARY_SELECTORS_COMMON,
    )

    ARTICLE_PATH_RE = re.compile(
        r"^/.+-\d{8,}\.chn$",
        re.IGNORECASE,
    )

    def list_page_candidates(self, page_number: int) -> list[str]:
        if page_number <= 1:
            return [self.category_url]

        return [
            f"{self.base_url}/{self.category_slug}/trang-{page_number}.chn",
            f"{self.category_url}?page={page_number}",
            f"{self.category_url}?trang={page_number}",
        ]

    def is_article_url(self, url: str) -> bool:
        parsed = urlsplit(url)
        if parsed.netloc.casefold() not in {"cafef.vn", "www.cafef.vn"}:
            return False
        return bool(self.ARTICLE_PATH_RE.match(parsed.path))

    def extract_urls_from_listing(self, html: str) -> list[str]:
        soup = BeautifulSoup(html, "lxml")
        root: Tag | BeautifulSoup = soup.find("main") or soup

        urls: list[str] = []
        seen: set[str] = set()
        started = False

        for element in root.find_all(["h1", "h2", "h3", "h4"]):
            text = normalize_inline(element.get_text(" ", strip=True))
            lowered = text.casefold()

            if not started and self.category_heading in lowered:
                started = True

            if (
                started
                and urls
                and lowered in CAFE_F_STOP_HEADINGS
            ):
                break

            anchor = element.find("a", href=True)
            if not anchor:
                continue

            url = canonicalize_url(str(anchor["href"]), self.base_url)
            if not self.is_article_url(url):
                continue
            if url in seen:
                continue

            seen.add(url)
            urls.append(url)

        # Fallback: CafeF có ID số dài ở cuối slug nên lọc rất chắc.
        if not urls:
            for anchor in root.select('a[href*=".chn"]'):
                if not isinstance(anchor, Tag):
                    continue

                url = canonicalize_url(str(anchor["href"]), self.base_url)
                if self.is_article_url(url) and url not in seen:
                    seen.add(url)
                    urls.append(url)

        return urls



class CafeFStockCrawler(CafeFCrawler):
    """
    CafeF - Thị trường chứng khoán.

    Dùng chung parser bài chi tiết với CafeFCrawler vì cấu trúc title, sapo,
    article body và khu vực "Cùng chuyên mục" giống nhau.
    """

    name = "cafef_stock"
    category_slug = "thi-truong-chung-khoan"
    category_heading = "thị trường chứng khoán"
    category_url = "https://cafef.vn/thi-truong-chung-khoan.chn"



class CafeFStockLoadMoreDiscovery:
    """
    Khám phá URL CafeF Stock bằng cách bấm nút "Xem thêm".

    Batch 1 là danh sách hiển thị ban đầu. Mỗi lần bấm thành công và xuất hiện
    URL bài mới tạo ra batch tiếp theo.

    Ví dụ:
        --start-page 2 --pages 10
    sẽ chọn URL mới của batch 2 đến batch 11.
    """

    LOAD_MORE_TEXTS = (
        "Xem thêm",
        "XEM THÊM",
        "Tải thêm",
        "TẢI THÊM",
    )

    LOAD_MORE_SELECTORS = (
        'button[class*="loadmore"]',
        'button[class*="load-more"]',
        'button[class*="viewmore"]',
        'button[class*="view-more"]',
        'a[class*="loadmore"]',
        'a[class*="load-more"]',
        'a[class*="viewmore"]',
        'a[class*="view-more"]',
        '[data-role*="load-more"]',
        '[data-action*="load-more"]',
    )

    def __init__(
        self,
        crawler: CafeFStockCrawler,
        timeout: float,
        browser_channel: str,
        headed: bool,
        profile_dir: str,
        load_more_wait_ms: int,
    ) -> None:
        self.crawler = crawler
        self.timeout_ms = int(timeout * 1000)
        self.browser_channel = browser_channel
        self.headed = headed
        self.profile_dir = str(Path(profile_dir).resolve())
        self.load_more_wait_ms = load_more_wait_ms

        self._playwright = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None

    def __enter__(self) -> "CafeFStockLoadMoreDiscovery":
        Path(self.profile_dir).mkdir(parents=True, exist_ok=True)
        self._playwright = sync_playwright().start()

        launch_kwargs: dict[str, Any] = {
            "user_data_dir": self.profile_dir,
            "headless": not self.headed,
            "locale": "vi-VN",
            "timezone_id": "Asia/Ho_Chi_Minh",
            "viewport": {"width": 1440, "height": 1100},
            "timeout": self.timeout_ms,
        }

        if self.browser_channel != "chromium":
            launch_kwargs["channel"] = self.browser_channel

        try:
            self.context = (
                self._playwright.chromium.launch_persistent_context(
                    **launch_kwargs
                )
            )
        except PlaywrightError:
            # Nếu Chrome/Edge channel không tồn tại, fallback Chromium bundled.
            launch_kwargs.pop("channel", None)
            try:
                self.context = (
                    self._playwright.chromium.launch_persistent_context(
                        **launch_kwargs
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
        self.page.set_default_navigation_timeout(self.timeout_ms)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            if self.context is not None:
                self.context.close()
        finally:
            if self._playwright is not None:
                self._playwright.stop()

    def _require_page(self) -> Page:
        if self.page is None:
            raise RuntimeError("Playwright browser chưa được khởi tạo.")
        return self.page

    @staticmethod
    def _is_safe_click_target(locator: Locator) -> bool:
        """
        Tránh click một link điều hướng thông thường có chữ "Xem thêm".

        Nút load-more thường là button/div, hoặc anchor có href rỗng, #,
        javascript: hay cùng URL hiện tại.
        """
        try:
            tag_name = locator.evaluate(
                "(element) => element.tagName.toLowerCase()"
            )
            if tag_name != "a":
                return True

            href = locator.get_attribute("href")
            if href is None:
                return True

            normalized = href.strip().casefold()
            return (
                not normalized
                or normalized == "#"
                or normalized.startswith("javascript:")
            )
        except PlaywrightError:
            return False

    def _candidate_buttons(self, page: Page) -> list[Locator]:
        candidates: list[Locator] = []
        seen_handles: set[str] = set()

        def add_locator(locator: Locator) -> None:
            try:
                count = locator.count()
            except PlaywrightError:
                return

            for index in range(count):
                candidate = locator.nth(index)
                try:
                    if not candidate.is_visible():
                        continue

                    key = candidate.evaluate(
                        """(element) => [
                            element.tagName,
                            element.id || '',
                            element.className || '',
                            (element.textContent || '').trim()
                        ].join('|')"""
                    )
                    if key in seen_handles:
                        continue
                    seen_handles.add(key)
                    candidates.append(candidate)
                except PlaywrightError:
                    continue

        for label in self.LOAD_MORE_TEXTS:
            add_locator(page.get_by_text(label, exact=True))

        for selector in self.LOAD_MORE_SELECTORS:
            add_locator(page.locator(selector))

        return candidates

    def _click_and_wait_for_new_urls(
        self,
        page: Page,
        previous_urls: list[str],
    ) -> list[str] | None:
        previous_set = set(previous_urls)

        for button in self._candidate_buttons(page):
            if not self._is_safe_click_target(button):
                continue

            try:
                button.scroll_into_view_if_needed()
                button.click(timeout=self.timeout_ms)
            except PlaywrightError:
                continue

            deadline = (
                time.monotonic()
                + max(self.load_more_wait_ms, 1000) / 1000.0
            )

            while time.monotonic() < deadline:
                page.wait_for_timeout(400)
                current_urls = self.crawler.extract_urls_from_listing(
                    page.content()
                )
                if set(current_urls) - previous_set:
                    return current_urls

            # Nút có thể vừa bị vô hiệu hóa hoặc thay bằng node mới.
            # Nếu click không sinh URL mới, thử candidate tiếp theo.

        return None

    def discover_batches(
        self,
        start_page: int,
        pages: int,
    ) -> list[str]:
        page = self._require_page()
        category_url = self.crawler.category_url

        logging.info(
            "[cafef_stock] Mở danh sách bằng trình duyệt: %s",
            category_url,
        )

        try:
            response = page.goto(
                category_url,
                wait_until="domcontentloaded",
                timeout=self.timeout_ms,
            )
        except PlaywrightTimeoutError as exc:
            raise RuntimeError(
                f"Timeout khi mở trang danh sách {category_url}."
            ) from exc
        except PlaywrightError as exc:
            raise RuntimeError(
                f"Không mở được trang danh sách {category_url}: {exc}"
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

        cumulative_urls = self.crawler.extract_urls_from_listing(
            page.content()
        )
        if not cumulative_urls:
            raise RuntimeError(
                "Không tìm thấy URL CafeF Stock trong batch đầu tiên."
            )

        batches[1] = list(cumulative_urls)
        logging.info(
            "[cafef_stock] Batch 1: %s URL.",
            len(cumulative_urls),
        )

        current_batch = 1

        while current_batch < target_last_page:
            expanded_urls = self._click_and_wait_for_new_urls(
                page,
                cumulative_urls,
            )

            if expanded_urls is None:
                logging.warning(
                    "[cafef_stock] Không còn nút Xem thêm hoặc click "
                    "không sinh URL mới tại batch %s.",
                    current_batch,
                )
                break

            previous_set = set(cumulative_urls)
            new_urls = [
                url for url in expanded_urls
                if url not in previous_set
            ]

            if not new_urls:
                logging.warning(
                    "[cafef_stock] Không có URL mới sau batch %s; dừng.",
                    current_batch,
                )
                break

            current_batch += 1
            batches[current_batch] = new_urls
            cumulative_urls = expanded_urls

            logging.info(
                "[cafef_stock] Batch %s: %s URL mới, tổng DOM=%s.",
                current_batch,
                len(new_urls),
                len(cumulative_urls),
            )

        selected: list[str] = []
        selected_seen: set[str] = set()

        for batch_number in range(
            start_page,
            target_last_page + 1,
        ):
            for url in batches.get(batch_number, []):
                if url in selected_seen:
                    continue
                selected_seen.add(url)
                selected.append(url)

        return selected


# Giữ tên riêng để tránh typo trong class.
CAFE_F_STOP_HEADINGS = {
    "cùng chuyên mục",
}


def load_json_records(
    path: Path,
    required_keys: set[str],
) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"File JSON lỗi {path}: {exc}"
        ) from exc

    if not isinstance(data, list):
        raise RuntimeError(
            f"{path} phải chứa JSON array, không phải {type(data).__name__}."
        )

    output: list[dict[str, Any]] = []
    for index, record in enumerate(data):
        if not isinstance(record, dict):
            logging.warning(
                "Bỏ record không phải object tại %s[%s].",
                path,
                index,
            )
            continue

        if not required_keys.issubset(record):
            logging.warning(
                "Bỏ record thiếu trường tại %s[%s]: %s",
                path,
                index,
                required_keys - set(record),
            )
            continue

        output.append(record)

    return output


def atomic_write_json(
    path: Path,
    records: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")

    payload = json.dumps(
        list(records),
        ensure_ascii=False,
        indent=2,
    )
    temp_path.write_text(payload + "\n", encoding="utf-8")
    temp_path.replace(path)


def save_state(
    data_path: Path,
    error_path: Path,
    article_map: dict[str, dict[str, Any]],
    error_map: dict[str, dict[str, Any]],
) -> None:
    atomic_write_json(data_path, article_map.values())
    atomic_write_json(error_path, error_map.values())


def build_crawlers(
    site_names: Iterable[str],
    session: requests.Session,
    min_text_chars: int,
) -> list[SiteCrawler]:
    crawler_types: dict[str, type[SiteCrawler]] = {
        "vneconomy": VnEconomyCrawler,
        "cafef": CafeFCrawler,
        "cafef_stock": CafeFStockCrawler,
    }

    return [
        crawler_types[name](
            session=session,
            min_text_chars=min_text_chars,
        )
        for name in site_names
    ]


def run(args: argparse.Namespace) -> tuple[int, int, int]:
    data_path = Path(args.data_output)
    error_path = Path(args.error_output)

    if args.overwrite:
        article_map: dict[str, dict[str, Any]] = {}
        error_map: dict[str, dict[str, Any]] = {}
    else:
        existing_articles = load_json_records(
            data_path,
            {"id", "domain", "text"},
        )
        existing_errors = load_json_records(
            error_path,
            {"id", "url", "error"},
        )
        article_map = {
            str(record["id"]): record
            for record in existing_articles
        }
        error_map = {
            str(record["id"]): record
            for record in existing_errors
        }

    session = build_session(args.timeout)
    crawlers = build_crawlers(
        args.sites,
        session,
        args.min_text_chars,
    )

    saved = 0
    skipped = 0
    failed = 0
    changes_since_checkpoint = 0

    try:
        for crawler in crawlers:
            discovered: list[str] = []

            if crawler.name == "cafef_stock":
                category_error_id = crawler.record_id(
                    crawler.category_url
                )

                try:
                    with CafeFStockLoadMoreDiscovery(
                        crawler=crawler,
                        timeout=args.timeout,
                        browser_channel=args.browser_channel,
                        headed=args.headed,
                        profile_dir=args.browser_profile_dir,
                        load_more_wait_ms=args.load_more_wait_ms,
                    ) as discovery:
                        discovered = discovery.discover_batches(
                            start_page=args.start_page,
                            pages=args.pages,
                        )

                    # Nếu lần trước discovery lỗi nhưng giờ thành công.
                    error_map.pop(category_error_id, None)

                except Exception as exc:
                    failed += 1
                    changes_since_checkpoint += 1
                    error_map[category_error_id] = ErrorRecord(
                        id=category_error_id,
                        url=crawler.category_url,
                        error=repr(exc),
                    ).to_dict()
                    logging.error(
                        "[cafef_stock] Lỗi khám phá URL bằng Xem thêm: %r",
                        exc,
                    )
                    discovered = []

            else:
                discovered_seen: set[str] = set()
                end_page = args.start_page + args.pages

                for page_number in range(
                    args.start_page,
                    end_page,
                ):
                    page_candidates = crawler.list_page_candidates(
                        page_number
                    )
                    representative_url = page_candidates[0]
                    page_error_id = crawler.record_id(
                        representative_url
                    )

                    try:
                        used_page_url, page_urls = (
                            crawler.discover_page(page_number)
                        )
                        logging.info(
                            "[%s] Trang %s: %s URL từ %s",
                            crawler.name,
                            page_number,
                            len(page_urls),
                            used_page_url,
                        )

                        error_map.pop(page_error_id, None)

                        new_on_page = 0
                        for url in page_urls:
                            if url in discovered_seen:
                                continue
                            discovered_seen.add(url)
                            discovered.append(url)
                            new_on_page += 1

                        if new_on_page == 0:
                            logging.info(
                                "[%s] Trang %s không có URL mới; "
                                "dừng phân trang.",
                                crawler.name,
                                page_number,
                            )
                            break

                    except Exception as exc:
                        failed += 1
                        changes_since_checkpoint += 1
                        error_map[page_error_id] = ErrorRecord(
                            id=page_error_id,
                            url=representative_url,
                            error=repr(exc),
                        ).to_dict()
                        logging.error(
                            "[%s] Lỗi trang danh sách %s: %r",
                            crawler.name,
                            representative_url,
                            exc,
                        )
                        continue

                    polite_sleep(args.delay, args.jitter)

            if args.max_articles_per_site is not None:
                discovered = discovered[
                    : args.max_articles_per_site
                ]

            logging.info(
                "[%s] Tổng URL sẽ xét: %s",
                crawler.name,
                len(discovered),
            )

            for index, url in enumerate(discovered, start=1):
                record_id = crawler.record_id(url)

                if (
                    record_id in article_map
                    and not args.refresh
                ):
                    skipped += 1
                    continue

                try:
                    logging.info(
                        "[%s][%s/%s] Crawl: %s",
                        crawler.name,
                        index,
                        len(discovered),
                        url,
                    )
                    article = crawler.extract_article(url)
                    article_map[article.id] = article.to_dict()
                    error_map.pop(article.id, None)

                    saved += 1
                    changes_since_checkpoint += 1

                except Exception as exc:
                    failed += 1
                    changes_since_checkpoint += 1
                    error_map[record_id] = ErrorRecord(
                        id=record_id,
                        url=url,
                        error=repr(exc),
                    ).to_dict()
                    logging.error(
                        "[%s] Lỗi bài %s: %r",
                        crawler.name,
                        url,
                        exc,
                    )

                if (
                    changes_since_checkpoint
                    >= args.checkpoint_every
                ):
                    save_state(
                        data_path,
                        error_path,
                        article_map,
                        error_map,
                    )
                    logging.info(
                        "Checkpoint | articles=%s | errors=%s",
                        len(article_map),
                        len(error_map),
                    )
                    changes_since_checkpoint = 0

                polite_sleep(args.delay, args.jitter)

    finally:
        save_state(
            data_path,
            error_path,
            article_map,
            error_map,
        )

    return saved, skipped, failed

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Crawl VnEconomy Chứng khoán, CafeF Tài chính quốc tế "
            "và CafeF Thị trường chứng khoán sang hai file JSON."
        )
    )
    parser.add_argument(
        "--sites",
        nargs="+",
        choices=("vneconomy", "cafef", "cafef_stock"),
        default=["vneconomy", "cafef", "cafef_stock"],
        help=(
            "Nguồn cần crawl: vneconomy, cafef hoặc cafef_stock; "
            "mặc định crawl cả ba."
        ),
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help=("Trang/batch bắt đầu crawl. Với cafef_stock, đây là batch ""Xem thêm; mặc định: 1."),
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=1,
        help=("Số trang hoặc số batch cần crawl; mặc định: 1."),
    )
    parser.add_argument(
        "--max-articles-per-site",
        type=int,
        default=None,
        help="Giới hạn số URL bài mỗi website sau khi khám phá.",
    )
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=200,
        help=(
            "Đưa bài vào file lỗi nếu text ngắn hơn ngưỡng; "
            "mặc định: 200. Dùng 0 để không kiểm tra."
        ),
    )
    parser.add_argument(
        "--data-output",
        default="finance_articles.json",
        help="File JSON chứa bài crawl thành công.",
    )
    parser.add_argument(
        "--error-output",
        default="finance_errors.json",
        help="File JSON chứa URL lỗi.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Số giây nghỉ tối thiểu giữa request; mặc định: 2.",
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=1.0,
        help="Khoảng nghỉ ngẫu nhiên cộng thêm 0..jitter giây.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout mỗi request, tính bằng giây.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Ghi checkpoint sau mỗi N thay đổi; mặc định: 10.",
    )
    parser.add_argument(
        "--browser-channel",
        choices=("chrome", "msedge", "chromium"),
        default="chrome",
        help=(
            "Trình duyệt dùng để bấm Xem thêm cho cafef_stock; "
            "mặc định: chrome."
        ),
    )
    parser.add_argument(
        "--browser-profile-dir",
        default=".cafef_stock_browser_profile",
        help="Thư mục lưu cookie/profile Playwright cho cafef_stock.",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help=(
            "Hiện cửa sổ trình duyệt khi bấm Xem thêm. "
            "Mặc định chạy headless."
        ),
    )
    parser.add_argument(
        "--load-more-wait-ms",
        type=int,
        default=12000,
        help=(
            "Thời gian tối đa chờ URL mới sau mỗi lần bấm Xem thêm; "
            "mặc định: 12000 ms."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Tạo lại cả hai file từ đầu.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Crawl lại cả bài đã có trong file chính.",
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
    if (
        args.max_articles_per_site is not None
        and args.max_articles_per_site < 1
    ):
        raise ValueError("--max-articles-per-site phải >= 1")
    if args.min_text_chars < 0:
        raise ValueError("--min-text-chars không được âm")
    if args.delay < 0 or args.jitter < 0:
        raise ValueError("--delay và --jitter không được âm")
    if args.timeout <= 0:
        raise ValueError("--timeout phải > 0")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every phải >= 1")
    if args.load_more_wait_ms < 1000:
        raise ValueError("--load-more-wait-ms phải >= 1000")

    data_path = Path(args.data_output).resolve()
    error_path = Path(args.error_output).resolve()
    if data_path == error_path:
        raise ValueError(
            "--data-output và --error-output phải là hai file khác nhau."
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
        logging.warning("Đã dừng bởi người dùng; checkpoint cuối đã được ghi.")
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