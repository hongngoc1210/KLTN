from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
import urllib.robotparser
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator, Sequence
from urllib.parse import urljoin, urlparse, urlunparse

try:
    import truststore

    truststore.inject_into_ssl()
    TRUSTSTORE_ENABLED = True
except ImportError:
    TRUSTSTORE_ENABLED = False

import requests
import yaml
from bs4 import BeautifulSoup
from docx import Document
from docx.document import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf  # type: ignore


DOMAIN = "education"
PDF_MAGIC = b"%PDF"
DOCX_MAGIC = b"PK\x03\x04"
LEGACY_DOC_MAGIC = bytes.fromhex("D0CF11E0A1B11AE1")

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36 "
    "UniversityEducationDatasetCrawler/0.1"
)

GENERIC_CONTENT_SELECTORS = [
    "article .entry-content",
    "article .post-content",
    "article .content",
    "article",
    "main .entry-content",
    "main .post-content",
    "main .content",
    ".node-content",
    ".field-name-body",
    ".views-field-body",
    ".news-detail",
    ".detail-content",
    ".post-content",
    ".entry-content",
    ".content-detail",
    ".article-content",
    ".article-detail",
    ".page-content",
    "main",
]

GENERIC_REMOVE_SELECTORS = [
    "script",
    "style",
    "noscript",
    "svg",
    "canvas",
    "iframe",
    "nav",
    "footer",
    "header",
    "form",
    "button",
    "input",
    "select",
    "textarea",
    ".breadcrumb",
    ".breadcrumbs",
    ".share",
    ".social",
    ".related",
    ".related-posts",
    ".post-navigation",
    ".comments",
    ".comment",
    ".sidebar",
    "aside",
]

ATTACHMENT_EXTENSIONS = (
    ".pdf",
    ".docx",
    ".doc",
)


@dataclass
class SiteConfig:
    key: str
    name: str
    base_url: str
    start_urls: list[str]
    id_prefix: str
    detail_patterns: list[str]
    pagination_patterns: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)
    list_link_selectors: list[str] = field(default_factory=list)
    content_selectors: list[str] = field(default_factory=list)
    remove_selectors: list[str] = field(default_factory=list)
    attachment_patterns: list[str] = field(default_factory=list)
    renderer: str = "requests"
    prefer_attachments: bool = True
    allow_html_fallback: bool = True
    id_pattern: str | None = None
    max_pages: int = 10
    max_docs: int = 100
    min_chars: int = 300
    enabled: bool = True

    detail_regexes: list[re.Pattern[str]] = field(
        init=False,
        repr=False,
    )
    pagination_regexes: list[re.Pattern[str]] = field(
        init=False,
        repr=False,
    )
    exclude_regexes: list[re.Pattern[str]] = field(
        init=False,
        repr=False,
    )
    attachment_regexes: list[re.Pattern[str]] = field(
        init=False,
        repr=False,
    )
    id_regex: re.Pattern[str] | None = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self.detail_regexes = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.detail_patterns
        ]
        self.pagination_regexes = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.pagination_patterns
        ]
        self.exclude_regexes = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.exclude_patterns
        ]
        self.attachment_regexes = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.attachment_patterns
        ]
        self.id_regex = (
            re.compile(self.id_pattern, re.IGNORECASE)
            if self.id_pattern
            else None
        )


def clean_inline_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = [
        re.sub(r"[ \t]+", " ", line).strip()
        for line in text.splitlines()
    ]

    output: list[str] = []
    previous_blank = False

    for line in lines:
        if not line:
            if output and not previous_blank:
                output.append("")
            previous_blank = True
            continue

        output.append(line)
        previous_blank = False

    return "\n".join(output).strip()


def canonicalize_url(
    url: str,
    base_url: str | None = None,
) -> str:
    absolute = urljoin(base_url or url, url)
    parsed = urlparse(absolute)

    # Fragment không làm thay đổi tài liệu.
    parsed = parsed._replace(fragment="")

    # Chuẩn hóa slash cuối nhưng không làm hỏng root.
    path = re.sub(r"/{2,}", "/", parsed.path or "/")

    if path != "/" and path.endswith("/"):
        path = path[:-1]

    return urlunparse(parsed._replace(path=path))


def same_host(url: str, base_url: str) -> bool:
    return (
        (urlparse(url).hostname or "").lower()
        == (urlparse(base_url).hostname or "").lower()
    )


def matches_any(
    value: str,
    patterns: Sequence[re.Pattern[str]],
) -> bool:
    return any(pattern.search(value) for pattern in patterns)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )


def load_existing_ids(path: Path) -> set[str]:
    existing: set[str] = set()

    if not path.exists():
        return existing

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"CẢNH BÁO: JSON lỗi tại "
                    f"{path}:{line_number}, bỏ qua."
                )
                continue

            record_id = record.get("id")

            if isinstance(record_id, str):
                existing.add(record_id)

    return existing


def build_session(
    retries: int,
    backoff: float,
    user_agent: str,
    verify_ssl: bool,
) -> requests.Session:
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=backoff,
        status_forcelist=(
            408,
            425,
            429,
            500,
            502,
            503,
            504,
        ),
        allowed_methods=frozenset({"GET", "HEAD"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=20,
        pool_maxsize=20,
    )

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": (
                "text/html,application/xhtml+xml,"
                "application/xml;q=0.9,"
                "application/pdf,"
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document,"
                "application/octet-stream,*/*;q=0.8"
            ),
            "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.7",
            "Connection": "keep-alive",
        }
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session._crawler_verify_ssl = verify_ssl  # type: ignore[attr-defined]

    return session


def get_verify_ssl(session: requests.Session) -> bool:
    return bool(
        getattr(session, "_crawler_verify_ssl", True)
    )


def fetch_requests(
    session: requests.Session,
    url: str,
    timeout: tuple[float, float],
    referer: str | None = None,
    stream: bool = False,
) -> requests.Response:
    headers = {"Referer": referer} if referer else None

    response = session.get(
        url,
        headers=headers,
        timeout=timeout,
        allow_redirects=True,
        stream=stream,
        verify=get_verify_ssl(session),
    )
    response.raise_for_status()
    return response


def check_robots(
    session: requests.Session,
    site: SiteConfig,
    policy: str,
    timeout: tuple[float, float],
    user_agent: str,
) -> urllib.robotparser.RobotFileParser | None:
    if policy == "off":
        return None

    robots_url = f"{site.base_url}/robots.txt"
    parser = urllib.robotparser.RobotFileParser()
    parser.set_url(robots_url)

    try:
        response = fetch_requests(
            session=session,
            url=robots_url,
            timeout=timeout,
        )
        parser.parse(response.text.splitlines())
    except Exception as error:
        if policy == "strict":
            raise RuntimeError(
                f"[{site.key}] Không đọc được robots.txt: {error}"
            ) from error

        print(
            f"CẢNH BÁO [{site.key}]: "
            f"Không đọc được robots.txt: {error}"
        )
        return None

    for start_url in site.start_urls:
        if not parser.can_fetch(user_agent, start_url):
            raise RuntimeError(
                f"[{site.key}] robots.txt không cho phép: "
                f"{start_url}"
            )

    return parser


def ensure_robots_allowed(
    parser: urllib.robotparser.RobotFileParser | None,
    url: str,
    user_agent: str,
) -> None:
    if parser is not None and not parser.can_fetch(
        user_agent,
        url,
    ):
        raise RuntimeError(
            f"robots.txt không cho phép crawl: {url}"
        )


class PageFetcher:
    """
    Fetch HTML bằng requests hoặc Playwright tùy renderer của site.
    File đính kèm vẫn ưu tiên requests; với renderer=playwright,
    crawler có thể dùng APIRequestContext cùng cookie khi cần.
    """

    def __init__(
        self,
        session: requests.Session,
        site: SiteConfig,
        args: argparse.Namespace,
    ) -> None:
        self.session = session
        self.site = site
        self.args = args
        self.playwright = None
        self.context = None

        if site.renderer == "playwright":
            try:
                from playwright.sync_api import sync_playwright
            except ImportError as error:
                raise RuntimeError(
                    "Site yêu cầu Playwright. Hãy cài: "
                    "pip install playwright && "
                    "python -m playwright install chromium"
                ) from error

            self.playwright = sync_playwright().start()
            profile_dir = (
                Path(args.profile_root)
                / site.key
            )
            profile_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            launch_kwargs: dict[str, Any] = {
                "user_data_dir": str(profile_dir),
                "headless": not args.headed,
                "locale": "vi-VN",
                "user_agent": args.user_agent,
                "ignore_https_errors": args.insecure,
                "viewport": {
                    "width": 1440,
                    "height": 1000,
                },
            }

            if args.channel:
                launch_kwargs["channel"] = args.channel

            self.context = (
                self.playwright.chromium
                .launch_persistent_context(**launch_kwargs)
            )

    def close(self) -> None:
        if self.context is not None:
            self.context.close()

        if self.playwright is not None:
            self.playwright.stop()

    def get_html(
        self,
        url: str,
        referer: str | None = None,
    ) -> tuple[bytes, str]:
        if self.site.renderer == "requests":
            response = fetch_requests(
                session=self.session,
                url=url,
                timeout=self.args.timeout,
                referer=referer,
            )
            return response.content, response.url

        assert self.context is not None

        page = self.context.new_page()

        try:
            page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=int(
                    self.args.read_timeout * 1000
                ),
            )

            try:
                page.wait_for_load_state(
                    "networkidle",
                    timeout=15_000,
                )
            except Exception:
                pass

            page.wait_for_timeout(
                int(self.args.render_wait * 1000)
            )

            return (
                page.content().encode("utf-8"),
                page.url,
            )

        finally:
            page.close()

    def download(
        self,
        url: str,
        referer: str,
        max_file_mb: float,
    ) -> tuple[bytes, dict[str, str], str]:
        if self.site.renderer == "playwright":
            assert self.context is not None

            response = self.context.request.get(
                url,
                headers={
                    "Referer": referer,
                    "Accept": (
                        "application/pdf,"
                        "application/vnd.openxmlformats-officedocument."
                        "wordprocessingml.document,"
                        "application/octet-stream,*/*"
                    ),
                },
                timeout=int(
                    self.args.read_timeout * 1000
                ),
                fail_on_status_code=False,
            )

            if not response.ok:
                raise RuntimeError(
                    f"Download HTTP {response.status}: "
                    f"{response.status_text}"
                )

            content = response.body()
            headers = {
                key.lower(): value
                for key, value in response.headers.items()
            }
            final_url = response.url

        else:
            response = fetch_requests(
                session=self.session,
                url=url,
                timeout=self.args.timeout,
                referer=referer,
                stream=True,
            )

            limit = int(max_file_mb * 1024 * 1024)
            chunks: list[bytes] = []
            total = 0

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if not chunk:
                    continue

                total += len(chunk)

                if total > limit:
                    raise ValueError(
                        f"File vượt quá {max_file_mb:g} MB."
                    )

                chunks.append(chunk)

            content = b"".join(chunks)
            headers = {
                key.lower(): value
                for key, value in response.headers.items()
            }
            final_url = response.url

        if len(content) > int(max_file_mb * 1024 * 1024):
            raise ValueError(
                f"File vượt quá {max_file_mb:g} MB."
            )

        return content, headers, final_url


def load_sites(config_path: Path) -> dict[str, SiteConfig]:
    raw = yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    )

    if not isinstance(raw, dict):
        raise ValueError(
            "universities.yaml phải là mapping."
        )

    items = raw.get("universities")

    if not isinstance(items, list):
        raise ValueError(
            "Thiếu danh sách universities trong YAML."
        )

    sites: dict[str, SiteConfig] = {}

    for item in items:
        if not isinstance(item, dict):
            continue

        site = SiteConfig(
            key=str(item["key"]),
            name=str(item.get("name", item["key"])),
            base_url=str(item["base_url"]),
            start_urls=[
                str(url)
                for url in item.get("start_urls", [])
            ],
            id_prefix=str(
                item.get(
                    "id_prefix",
                    f"education_{item['key']}",
                )
            ),
            detail_patterns=[
                str(value)
                for value in item.get(
                    "detail_patterns",
                    [],
                )
            ],
            pagination_patterns=[
                str(value)
                for value in item.get(
                    "pagination_patterns",
                    [],
                )
            ],
            exclude_patterns=[
                str(value)
                for value in item.get(
                    "exclude_patterns",
                    [],
                )
            ],
            list_link_selectors=[
                str(value)
                for value in item.get(
                    "list_link_selectors",
                    [],
                )
            ],
            content_selectors=[
                str(value)
                for value in item.get(
                    "content_selectors",
                    [],
                )
            ],
            remove_selectors=[
                str(value)
                for value in item.get(
                    "remove_selectors",
                    [],
                )
            ],
            attachment_patterns=[
                str(value)
                for value in item.get(
                    "attachment_patterns",
                    [],
                )
            ],
            renderer=str(
                item.get("renderer", "requests")
            ).lower(),
            prefer_attachments=bool(
                item.get(
                    "prefer_attachments",
                    True,
                )
            ),
            allow_html_fallback=bool(
                item.get(
                    "allow_html_fallback",
                    True,
                )
            ),
            id_pattern=(
                str(item["id_pattern"])
                if item.get("id_pattern")
                else None
            ),
            max_pages=int(
                item.get("max_pages", 10)
            ),
            max_docs=int(
                item.get("max_docs", 100)
            ),
            min_chars=int(
                item.get("min_chars", 300)
            ),
            enabled=bool(
                item.get("enabled", True)
            ),
        )

        if not site.start_urls:
            raise ValueError(
                f"[{site.key}] start_urls đang rỗng."
            )

        if not site.detail_patterns:
            raise ValueError(
                f"[{site.key}] detail_patterns đang rỗng."
            )

        if site.renderer not in {
            "requests",
            "playwright",
        }:
            raise ValueError(
                f"[{site.key}] renderer không hợp lệ: "
                f"{site.renderer}"
            )

        sites[site.key] = site

    return sites


def extract_links(
    html: bytes,
    page_url: str,
    selectors: Sequence[str],
) -> list[str]:
    soup = BeautifulSoup(html, "lxml")

    anchors = []

    if selectors:
        for selector in selectors:
            try:
                anchors.extend(
                    soup.select(f"{selector}[href]")
                )
                anchors.extend(
                    element
                    for container in soup.select(selector)
                    for element in container.select("a[href]")
                )
            except Exception:
                continue

    if not anchors:
        anchors = soup.select("a[href]")

    urls: list[str] = []
    seen: set[str] = set()

    for anchor in anchors:
        href = anchor.get("href")

        if not href:
            continue

        href = str(href).strip()

        if href.startswith(
            ("mailto:", "tel:", "javascript:")
        ):
            continue

        url = canonicalize_url(href, page_url)

        if url not in seen:
            seen.add(url)
            urls.append(url)

    return urls


def classify_list_links(
    urls: Sequence[str],
    site: SiteConfig,
) -> tuple[list[str], list[str]]:
    detail_urls: list[str] = []
    page_urls: list[str] = []

    seen_details: set[str] = set()
    seen_pages: set[str] = set()

    for url in urls:
        if not same_host(url, site.base_url):
            continue

        if matches_any(url, site.exclude_regexes):
            continue

        if matches_any(url, site.detail_regexes):
            if url not in seen_details:
                seen_details.add(url)
                detail_urls.append(url)
            continue

        if (
            site.pagination_regexes
            and matches_any(
                url,
                site.pagination_regexes,
            )
        ):
            if url not in seen_pages:
                seen_pages.add(url)
                page_urls.append(url)

    return detail_urls, page_urls


def collect_detail_urls(
    fetcher: PageFetcher,
    site: SiteConfig,
    robots_parser: urllib.robotparser.RobotFileParser | None,
    args: argparse.Namespace,
    max_pages: int,
    max_docs: int,
) -> list[str]:
    queue = [
        canonicalize_url(url)
        for url in site.start_urls
    ]
    visited_pages: set[str] = set()
    detail_urls: list[str] = []
    seen_details: set[str] = set()

    while queue:
        if max_pages > 0 and len(visited_pages) >= max_pages:
            break

        page_url = queue.pop(0)

        if page_url in visited_pages:
            continue

        ensure_robots_allowed(
            robots_parser,
            page_url,
            args.user_agent,
        )

        # Một link phân trang cũ hoặc tạm thời lỗi không nên làm dừng
        # toàn bộ site (và các trường tiếp theo trong cùng lượt crawl).
        try:
            html, final_url = fetcher.get_html(page_url)
        except Exception as error:
            visited_pages.add(page_url)
            print(
                f"CẢNH BÁO [{site.key}]: "
                f"Không tải được trang danh sách {page_url}: "
                f"{error}"
            )
            polite_sleep(
                args.page_delay,
                args.jitter,
            )
            continue

        final_url = canonicalize_url(final_url)
        visited_pages.add(page_url)

        links = extract_links(
            html=html,
            page_url=final_url,
            selectors=site.list_link_selectors,
        )

        page_details, next_pages = classify_list_links(
            links,
            site,
        )

        new_count = 0

        for detail_url in page_details:
            if detail_url in seen_details:
                continue

            seen_details.add(detail_url)
            detail_urls.append(detail_url)
            new_count += 1

            if max_docs > 0 and len(detail_urls) >= max_docs:
                break

        print(
            f"[{site.key}] Trang danh sách "
            f"{len(visited_pages)}: "
            f"{len(page_details)} link bài, "
            f"mới {new_count}, "
            f"tổng {len(detail_urls)}."
        )

        if max_docs > 0 and len(detail_urls) >= max_docs:
            break

        for next_url in next_pages:
            if (
                next_url not in visited_pages
                and next_url not in queue
            ):
                queue.append(next_url)

        polite_sleep(
            args.page_delay,
            args.jitter,
        )

    return detail_urls


def make_document_id(
    url: str,
    site: SiteConfig,
) -> str:
    if site.id_regex is not None:
        match = site.id_regex.search(url)

        if match:
            value = (
                match.group(1)
                if match.lastindex
                else match.group(0)
            )
            safe_value = re.sub(
                r"[^a-zA-Z0-9_-]+",
                "_",
                value,
            ).strip("_")

            if safe_value:
                return (
                    f"{site.id_prefix}_"
                    f"{safe_value.lower()}"
                )

    digest = hashlib.sha256(
        canonicalize_url(url).encode("utf-8")
    ).hexdigest()[:24]

    return f"{site.id_prefix}_{digest}"


def extract_attachment_urls(
    html: bytes,
    detail_url: str,
    site: SiteConfig,
) -> list[str]:
    soup = BeautifulSoup(html, "lxml")

    urls: list[str] = []
    seen: set[str] = set()

    for anchor in soup.select("a[href]"):
        href = str(anchor.get("href", "")).strip()
        label = clean_inline_text(
            anchor.get_text(" ", strip=True)
        ).lower()

        if not href:
            continue

        url = canonicalize_url(href, detail_url)
        parsed_path = urlparse(url).path.lower()
        combined = f"{url} {label}".lower()

        extension_match = parsed_path.endswith(
            ATTACHMENT_EXTENSIONS
        )
        label_match = bool(
            re.search(
                r"\btải\b|\bdownload\b|"
                r"file\s*đính\s*kèm|"
                r"văn\s*bản\s*gốc",
                combined,
                re.IGNORECASE,
            )
        )
        configured_match = matches_any(
            combined,
            site.attachment_regexes,
        )

        if not (
            extension_match
            or configured_match
            or (
                label_match
                and re.search(
                    r"\.(pdf|docx|doc)(?:$|[?&#])",
                    combined,
                    re.IGNORECASE,
                )
            )
        ):
            continue

        if url not in seen:
            seen.add(url)
            urls.append(url)

    return urls


def iter_docx_blocks(
    document: DocxDocument,
) -> Iterator[Paragraph | Table]:
    body = document.element.body

    for child in body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield Table(child, document)


def extract_docx_text(content: bytes) -> str:
    if not content.startswith(DOCX_MAGIC):
        raise ValueError(
            "Nội dung không phải DOCX/ZIP hợp lệ."
        )

    document = Document(BytesIO(content))
    parts: list[str] = []

    for block in iter_docx_blocks(document):
        if isinstance(block, Paragraph):
            text = clean_inline_text(block.text)

            if text:
                parts.append(text)

        elif isinstance(block, Table):
            for row in block.rows:
                cells = [
                    clean_inline_text(cell.text)
                    for cell in row.cells
                ]

                compact: list[str] = []

                for cell_text in cells:
                    if not cell_text:
                        continue

                    if (
                        not compact
                        or compact[-1] != cell_text
                    ):
                        compact.append(cell_text)

                if compact:
                    parts.append(" | ".join(compact))

    return clean_text("\n".join(parts))


def extract_pdf_text(content: bytes) -> str:
    if not content.lstrip().startswith(PDF_MAGIC):
        raise ValueError(
            "Nội dung không phải PDF hợp lệ."
        )

    document = pymupdf.open(
        stream=content,
        filetype="pdf",
    )
    pages: list[str] = []

    try:
        for page in document:
            try:
                text = page.get_text(
                    "text",
                    sort=True,
                )
            except TypeError:
                text = page.get_text("text")

            text = clean_text(text)

            if text:
                pages.append(text)

    finally:
        document.close()

    result = clean_text("\n\n".join(pages))

    if len(result) < 100:
        raise ValueError(
            "PDF gần như không có lớp text; "
            "có thể là PDF scan."
        )

    return result


def detect_file_type(
    content: bytes,
    headers: dict[str, str],
    url: str,
) -> str:
    content_type = headers.get(
        "content-type",
        "",
    ).lower()
    content_disposition = headers.get(
        "content-disposition",
        "",
    ).lower()
    combined = (
        f"{url} {content_type} "
        f"{content_disposition}"
    ).lower()

    if content.lstrip().startswith(PDF_MAGIC):
        return "pdf"

    if content.startswith(DOCX_MAGIC):
        return "docx"

    if content.startswith(LEGACY_DOC_MAGIC):
        return "doc"

    if "application/pdf" in combined or ".pdf" in combined:
        return "pdf"

    if (
        "wordprocessingml" in combined
        or ".docx" in combined
    ):
        return "docx"

    if "msword" in combined or ".doc" in combined:
        return "doc"

    if "text/html" in content_type:
        return "html"

    return "unknown"


def extract_attachment_text(
    fetcher: PageFetcher,
    attachment_url: str,
    detail_url: str,
    max_file_mb: float,
) -> tuple[str, str]:
    content, headers, final_url = fetcher.download(
        url=attachment_url,
        referer=detail_url,
        max_file_mb=max_file_mb,
    )

    file_type = detect_file_type(
        content=content,
        headers=headers,
        url=final_url,
    )

    if file_type == "pdf":
        return extract_pdf_text(content), "pdf"

    if file_type == "docx":
        return extract_docx_text(content), "docx"

    if file_type == "doc":
        raise ValueError(
            "Định dạng .doc cũ chưa được hỗ trợ."
        )

    if file_type == "html":
        preview = clean_inline_text(
            content[:500].decode(
                "utf-8",
                errors="replace",
            )
        )
        raise ValueError(
            "Link tải trả về HTML thay vì file. "
            f"Preview: {preview[:150]}"
        )

    raise ValueError(
        "Không nhận diện được loại file."
    )


def extract_html_content(
    html: bytes,
    site: SiteConfig,
) -> str:
    soup = BeautifulSoup(html, "lxml")

    for selector in (
        GENERIC_REMOVE_SELECTORS
        + site.remove_selectors
    ):
        try:
            for element in soup.select(selector):
                element.decompose()
        except Exception:
            continue

    selectors = (
        site.content_selectors
        + [
            selector
            for selector in GENERIC_CONTENT_SELECTORS
            if selector not in site.content_selectors
        ]
    )

    candidates: list[str] = []

    for selector in selectors:
        try:
            for element in soup.select(selector):
                text = clean_text(
                    element.get_text(
                        "\n",
                        strip=True,
                    )
                )

                if text:
                    candidates.append(text)
        except Exception:
            continue

    if not candidates and soup.body is not None:
        candidates.append(
            clean_text(
                soup.body.get_text(
                    "\n",
                    strip=True,
                )
            )
        )

    return max(
        candidates,
        key=len,
        default="",
    )


def deduplicate_texts(
    texts: Sequence[str],
) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()

    for text in texts:
        normalized = clean_text(text)

        if not normalized:
            continue

        digest = hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()

        if digest in seen:
            continue

        seen.add(digest)
        output.append(normalized)

    return output


def crawl_one_document(
    fetcher: PageFetcher,
    site: SiteConfig,
    detail_url: str,
    robots_parser: urllib.robotparser.RobotFileParser | None,
    args: argparse.Namespace,
) -> tuple[str, str]:
    ensure_robots_allowed(
        robots_parser,
        detail_url,
        args.user_agent,
    )

    html, final_url = fetcher.get_html(detail_url)
    final_url = canonicalize_url(final_url)

    html_text = extract_html_content(
        html,
        site,
    )
    attachment_urls = extract_attachment_urls(
        html,
        final_url,
        site,
    )

    if args.max_files_per_doc > 0:
        attachment_urls = attachment_urls[
            :args.max_files_per_doc
        ]

    attachment_texts: list[str] = []
    source_types: list[str] = []
    attachment_errors: list[str] = []

    for index, attachment_url in enumerate(
        attachment_urls,
        start=1,
    ):
        try:
            text, source_type = extract_attachment_text(
                fetcher=fetcher,
                attachment_url=attachment_url,
                detail_url=final_url,
                max_file_mb=args.max_file_mb,
            )

            if len(text) < site.min_chars:
                raise ValueError(
                    f"Text file chỉ có "
                    f"{len(text)} ký tự."
                )

            attachment_texts.append(text)
            source_types.append(source_type)

        except Exception as error:
            attachment_errors.append(
                f"file {index}: {error}"
            )

    attachment_texts = deduplicate_texts(
        attachment_texts
    )

    if site.prefer_attachments and attachment_texts:
        return (
            clean_text(
                "\n\n".join(attachment_texts)
            ),
            "+".join(
                dict.fromkeys(source_types)
            ),
        )

    if (
        site.allow_html_fallback
        and len(html_text) >= site.min_chars
    ):
        return html_text, "html"

    if attachment_texts:
        return (
            clean_text(
                "\n\n".join(attachment_texts)
            ),
            "+".join(
                dict.fromkeys(source_types)
            ),
        )

    details: list[str] = []

    if not attachment_urls:
        details.append(
            "Không tìm thấy PDF/DOCX đính kèm."
        )

    if html_text:
        details.append(
            f"HTML chỉ có {len(html_text)} ký tự."
        )
    else:
        details.append(
            "Không trích được text HTML."
        )

    details.extend(attachment_errors)

    raise RuntimeError(" | ".join(details))


def polite_sleep(delay: float, jitter: float) -> None:
    time.sleep(
        max(
            0.0,
            delay + random.uniform(0.0, jitter),
        )
    )


def resolve_site_limits(
    site: SiteConfig,
    args: argparse.Namespace,
) -> tuple[int, int]:
    max_pages = (
        args.max_pages
        if args.max_pages is not None
        else site.max_pages
    )
    max_docs = (
        args.max_docs
        if args.max_docs is not None
        else site.max_docs
    )

    return max_pages, max_docs


def crawl_site(
    site: SiteConfig,
    args: argparse.Namespace,
    existing_ids: set[str],
) -> tuple[int, int, int]:
    print(f"\n=== {site.name} ({site.key}) ===")

    session = build_session(
        retries=args.retries,
        backoff=args.backoff,
        user_agent=args.user_agent,
        verify_ssl=not args.insecure,
    )

    robots_parser = check_robots(
        session=session,
        site=site,
        policy=args.robots_policy,
        timeout=args.timeout,
        user_agent=args.user_agent,
    )

    fetcher = PageFetcher(
        session=session,
        site=site,
        args=args,
    )

    saved = 0
    skipped = 0
    failed = 0

    try:
        max_pages, max_docs = resolve_site_limits(
            site,
            args,
        )

        detail_urls = collect_detail_urls(
            fetcher=fetcher,
            site=site,
            robots_parser=robots_parser,
            args=args,
            max_pages=max_pages,
            max_docs=max_docs,
        )

        print(
            f"[{site.key}] Sẽ xử lý "
            f"{len(detail_urls)} URL chi tiết."
        )

        for index, detail_url in enumerate(
            detail_urls,
            start=1,
        ):
            document_id = make_document_id(
                detail_url,
                site,
            )

            if document_id in existing_ids:
                skipped += 1
                print(
                    f"[{site.key} "
                    f"{index}/{len(detail_urls)}] "
                    f"SKIP {document_id}"
                )
                continue

            try:
                text, source_type = crawl_one_document(
                    fetcher=fetcher,
                    site=site,
                    detail_url=detail_url,
                    robots_parser=robots_parser,
                    args=args,
                )

                append_jsonl(
                    args.output,
                    {
                        "id": document_id,
                        "domain": DOMAIN,
                        "text": text,
                    },
                )
                existing_ids.add(document_id)
                saved += 1

                print(
                    f"[{site.key} "
                    f"{index}/{len(detail_urls)}] "
                    f"OK {document_id} | "
                    f"{source_type.upper()} | "
                    f"{len(text)} ký tự"
                )

            except Exception as error:
                failed += 1

                append_jsonl(
                    args.error_output,
                    {
                        "id": document_id,
                        "source": site.key,
                        "url": detail_url,
                        "error": repr(error),
                    },
                )

                print(
                    f"[{site.key} "
                    f"{index}/{len(detail_urls)}] "
                    f"ERROR {document_id}: {error}"
                )

            polite_sleep(
                args.delay,
                args.jitter,
            )

    finally:
        fetcher.close()
        session.close()

    print(
        f"[{site.key}] Lưu mới={saved}, "
        f"bỏ qua={skipped}, lỗi={failed}"
    )

    return saved, skipped, failed


def parse_selected_sites(
    raw_value: str,
    sites: dict[str, SiteConfig],
) -> list[SiteConfig]:
    if raw_value.strip().lower() == "all":
        return [
            site
            for site in sites.values()
            if site.enabled
        ]

    keys = [
        value.strip()
        for value in raw_value.split(",")
        if value.strip()
    ]

    unknown = [
        key
        for key in keys
        if key not in sites
    ]

    if unknown:
        raise ValueError(
            "Không có site trong YAML: "
            + ", ".join(unknown)
        )

    return [
        sites[key]
        for key in keys
        if sites[key].enabled
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("universities.yaml"),
    )
    parser.add_argument(
        "--sites",
        default="all",
        help=(
            "all hoặc danh sách key, ví dụ "
            "uit,ptit,hust."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/raw/education_universities.jsonl"
        ),
    )
    parser.add_argument(
        "--error-output",
        type=Path,
        default=Path(
            "data/raw/education_universities_errors.jsonl"
        ),
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help=(
            "Ghi đè max_pages của YAML; "
            "0 là không giới hạn."
        ),
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help=(
            "Ghi đè max_docs của YAML; "
            "0 là không giới hạn."
        ),
    )
    parser.add_argument(
        "--max-files-per-doc",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--max-file-mb",
        type=float,
        default=100,
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.5,
    )
    parser.add_argument(
        "--page-delay",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=20,
    )
    parser.add_argument(
        "--read-timeout",
        type=float,
        default=120,
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--backoff",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--robots-policy",
        choices=("strict", "warn", "off"),
        default="warn",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help=(
            "Tắt xác minh SSL. Chỉ dùng khi "
            "thực sự gặp lỗi certificate."
        ),
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help=(
            "Hiện Chromium cho các site "
            "renderer=playwright."
        ),
    )
    parser.add_argument(
        "--channel",
        default="chromium",
    )
    parser.add_argument(
        "--profile-root",
        default=(
            "data/browser_profiles/universities"
        ),
    )
    parser.add_argument(
        "--render-wait",
        type=float,
        default=2.0,
        help=(
            "Số giây chờ sau khi trang Playwright "
            "render xong."
        ),
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
    )

    args = parser.parse_args()
    args.timeout = (
        args.connect_timeout,
        args.read_timeout,
    )
    args.channel = (
        args.channel.strip()
        if args.channel
        else None
    )

    return args


def main() -> None:
    args = parse_args()

    print(
        "TLS certificate store: "
        + (
            "system trust store"
            if TRUSTSTORE_ENABLED
            else "Requests/certifi"
        )
    )

    sites = load_sites(args.config)
    selected_sites = parse_selected_sites(
        args.sites,
        sites,
    )

    if not selected_sites:
        raise RuntimeError(
            "Không có trường nào được chọn."
        )

    existing_ids = load_existing_ids(
        args.output
    )

    print(
        f"Đã có {len(existing_ids)} "
        f"bản ghi trong output."
    )
    print(
        "Nguồn sẽ crawl: "
        + ", ".join(
            site.key
            for site in selected_sites
        )
    )

    total_saved = 0
    total_skipped = 0
    total_failed = 0

    for site in selected_sites:
        saved, skipped, failed = crawl_site(
            site=site,
            args=args,
            existing_ids=existing_ids,
        )
        total_saved += saved
        total_skipped += skipped
        total_failed += failed

    print("\n=== HOÀN TẤT TẤT CẢ NGUỒN ===")
    print(f"- Lưu mới: {total_saved}")
    print(f"- Bỏ qua: {total_skipped}")
    print(f"- Lỗi: {total_failed}")
    print(f"- Output: {args.output}")
    print(f"- Error log: {args.error_output}")


if __name__ == "__main__":
    main()
