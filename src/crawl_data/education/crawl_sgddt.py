from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
import urllib.robotparser
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import unquote, urljoin, urlparse

from docx import Document
from docx.document import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from playwright.sync_api import (
    APIResponse,
    BrowserContext,
    Locator,
    Page,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)

try:
    import pymupdf
except ImportError:
    # Tương thích với các phiên bản PyMuPDF cũ.
    import fitz as pymupdf  # type: ignore


DOMAIN = "education"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

DETAIL_PATH_RE = re.compile(
    r"/vbct/(\d+)/(\d+)(?:/|$)",
    re.IGNORECASE,
)

PDF_MAGIC = b"%PDF"
DOCX_MAGIC = b"PK\x03\x04"
LEGACY_DOC_MAGIC = bytes.fromhex("D0CF11E0A1B11AE1")

CHALLENGE_MARKERS = (
    "what code is in the image",
    "prevent automated spam submission",
    "support id is",
    "mã trong hình",
    "xác minh bạn là con người",
    "captcha",
)


@dataclass(frozen=True)
class SiteConfig:
    key: str
    name: str
    base_url: str
    start_url: str
    id_prefix: str

    @property
    def robots_url(self) -> str:
        return f"{self.base_url}/robots.txt"


SITES: dict[str, SiteConfig] = {
    "hanoi": SiteConfig(
        key="hanoi",
        name="Sở GD&ĐT Hà Nội",
        base_url="https://hanoi.edu.vn",
        start_url="https://hanoi.edu.vn/van-ban/vbmb/0",
        id_prefix="education_hanoi",
    ),
    "hcm": SiteConfig(
        key="hcm",
        name="Sở GD&ĐT TP.HCM",
        base_url="https://hcm.edu.vn",
        start_url="https://hcm.edu.vn/van-ban/vbmb/0",
        id_prefix="education_hcm",
    ),
}


def clean_inline_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    """
    Chuẩn hóa khoảng trắng nhưng chưa chia tài liệu thành context nhỏ.
    """
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


def make_document_id(
    detail_url: str,
    site: SiteConfig,
) -> str:
    match = DETAIL_PATH_RE.search(
        urlparse(detail_url).path
    )

    if match:
        # group(2) là ID văn bản, group(1) là ID chuyên mục.
        return f"{site.id_prefix}_{match.group(2)}"

    digest = hashlib.sha256(
        detail_url.encode("utf-8")
    ).hexdigest()[:24]

    return f"{site.id_prefix}_{digest}"


def normalize_detail_url(
    raw_url: str,
    site: SiteConfig,
) -> str | None:
    absolute_url = urljoin(site.base_url, raw_url)
    parsed = urlparse(absolute_url)

    if (
        (parsed.hostname or "").lower()
        != (urlparse(site.base_url).hostname or "").lower()
    ):
        return None

    if not DETAIL_PATH_RE.search(parsed.path):
        return None

    return parsed._replace(
        query="",
        fragment="",
    ).geturl()


def wait_page_ready(
    page: Page,
    timeout_ms: int,
) -> None:
    page.wait_for_load_state(
        "domcontentloaded",
        timeout=timeout_ms,
    )

    try:
        page.wait_for_load_state(
            "networkidle",
            timeout=min(timeout_ms, 15_000),
        )
    except PlaywrightTimeoutError:
        # Một số trang có request analytics chạy liên tục.
        pass

    page.wait_for_timeout(1_800)


def save_debug(
    page: Page,
    debug_dir: Path,
    name: str,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(
        r"[^a-zA-Z0-9_.-]+",
        "_",
        name,
    )

    try:
        page.screenshot(
            path=str(debug_dir / f"{safe_name}.png"),
            full_page=True,
        )
    except Exception:
        pass

    try:
        (debug_dir / f"{safe_name}.html").write_text(
            page.content(),
            encoding="utf-8",
        )
    except Exception:
        pass

    try:
        (debug_dir / f"{safe_name}.txt").write_text(
            page.locator("body").inner_text(),
            encoding="utf-8",
        )
    except Exception:
        pass


def body_text_lower(page: Page) -> str:
    try:
        return clean_inline_text(
            page.locator("body").inner_text()
        ).lower()
    except Exception:
        return ""


def is_human_challenge(page: Page) -> bool:
    text = body_text_lower(page)

    return any(
        marker in text
        for marker in CHALLENGE_MARKERS
    )


def extract_detail_urls(
    page: Page,
    site: SiteConfig,
) -> list[str]:
    """
    Lấy link chi tiết từ href/data-url/onclick và HTML đã render.
    """
    raw_values = page.evaluate(
        """
        () => {
            const attrs = [
                "href",
                "data-href",
                "data-url",
                "routerlink",
                "ng-reflect-router-link",
                "onclick"
            ];

            const values = [];

            for (const element of document.querySelectorAll("*")) {
                for (const attr of attrs) {
                    const value = element.getAttribute(attr);

                    if (value && value.includes("/vbct/")) {
                        values.push(value);
                    }
                }
            }

            const html = document.documentElement.innerHTML;
            const matches = html.match(
                /(?:https?:\\/\\/[^"'<>\\s]+)?\\/[^"'<>\\s]*\\/vbct\\/\\d+\\/\\d+/g
            ) || [];

            values.push(...matches);
            return values;
        }
        """
    )

    urls: list[str] = []
    seen: set[str] = set()

    for raw in raw_values:
        if not isinstance(raw, str):
            continue

        match = re.search(
            r"(?:https?://[^\"'()<>\s]+)?"
            r"/[^\"'()<>\s]*/vbct/\d+/\d+",
            raw,
            re.IGNORECASE,
        )

        candidate = match.group(0) if match else raw
        detail_url = normalize_detail_url(
            candidate,
            site,
        )

        if detail_url and detail_url not in seen:
            seen.add(detail_url)
            urls.append(detail_url)

    return urls


def wait_for_document_list(
    page: Page,
    site: SiteConfig,
    timeout_ms: int,
    headed: bool,
    debug_dir: Path,
) -> None:
    """
    Chờ danh sách render. Nếu gặp CAPTCHA ở TP.HCM, cho phép người dùng
    giải thủ công trên cửa sổ Chromium.
    """
    deadline = time.time() + timeout_ms / 1000

    while time.time() < deadline:
        urls = extract_detail_urls(page, site)

        if urls:
            return

        if is_human_challenge(page):
            save_debug(
                page,
                debug_dir,
                f"{site.key}_human_challenge",
            )

            if not headed:
                raise RuntimeError(
                    f"{site.name} đang yêu cầu CAPTCHA. "
                    "Hãy chạy lại với --headed và dùng cùng "
                    "--profile-root để lưu phiên."
                )

            print(
                f"\n{site.name} đang yêu cầu xác minh CAPTCHA."
            )
            print(
                "Hãy giải CAPTCHA trong cửa sổ trình duyệt. "
                "Sau khi thấy trang danh sách văn bản, "
                "quay lại terminal và nhấn Enter."
            )

            input("Nhấn Enter sau khi đã giải CAPTCHA... ")

            try:
                wait_page_ready(page, timeout_ms)
            except Exception:
                pass

            # Cho phép vòng lặp kiểm tra lại.
            deadline = time.time() + timeout_ms / 1000
            continue

        page.wait_for_timeout(750)

    save_debug(
        page,
        debug_dir,
        f"{site.key}_no_document_list",
    )

    raise RuntimeError(
        f"Không tìm thấy link /vbct/ trên {site.name}. "
        f"Đã lưu screenshot và HTML tại {debug_dir}."
    )


def page_signature(
    page: Page,
    site: SiteConfig,
) -> list[str]:
    return extract_detail_urls(page, site)[:5]


def locator_is_usable(locator: Locator) -> bool:
    try:
        if locator.count() == 0:
            return False

        target = locator.first

        if not target.is_visible():
            return False

        if target.get_attribute("disabled") is not None:
            return False

        if target.get_attribute("aria-disabled") == "true":
            return False

        class_name = target.get_attribute("class") or ""

        if re.search(
            r"\bdisabled\b",
            class_name,
            re.IGNORECASE,
        ):
            return False

        return True

    except Exception:
        return False


def mark_pagination_target(
    page: Page,
    next_page_number: int,
) -> bool:
    """
    Chọn phần tử phân trang tốt nhất:
    - ưu tiên đúng số trang kế tiếp;
    - nếu không có, ưu tiên nút >>, », ›, Tiếp, Sau;
    - tăng điểm nếu ancestor có class/id chứa page/paging/pagination.
    """
    return bool(
        page.evaluate(
            """
            nextPageNumber => {
                const normalize = value =>
                    (value || "").replace(/\\s+/g, " ").trim();

                const visible = element => {
                    const style = getComputedStyle(element);
                    const rect = element.getBoundingClientRect();

                    return (
                        style.display !== "none" &&
                        style.visibility !== "hidden" &&
                        rect.width > 2 &&
                        rect.height > 2
                    );
                };

                const clickableAncestor = element => {
                    let node = element;

                    for (
                        let depth = 0;
                        depth < 6 && node;
                        depth += 1
                    ) {
                        const style = getComputedStyle(node);
                        const role = node.getAttribute("role");

                        if (
                            node.tagName === "A" ||
                            node.tagName === "BUTTON" ||
                            node.hasAttribute("onclick") ||
                            role === "button" ||
                            role === "link" ||
                            style.cursor === "pointer"
                        ) {
                            return node;
                        }

                        node = node.parentElement;
                    }

                    return element;
                };

                const paginationScore = element => {
                    let score = 0;
                    let node = element;

                    for (
                        let depth = 0;
                        depth < 6 && node;
                        depth += 1
                    ) {
                        const value = [
                            node.id || "",
                            node.className || "",
                            node.getAttribute("aria-label") || ""
                        ].join(" ").toLowerCase();

                        if (
                            /page|paging|pagination|pager/.test(value)
                        ) {
                            score += 1000;
                        }

                        node = node.parentElement;
                    }

                    score += element.getBoundingClientRect().top;
                    return score;
                };

                document.querySelectorAll(
                    "[data-edu-crawl-next='1']"
                ).forEach(element => {
                    element.removeAttribute(
                        "data-edu-crawl-next"
                    );
                });

                const all = [
                    ...document.querySelectorAll(
                        "a,button,li,span,[role='button']"
                    )
                ].filter(visible);

                const exact = all.filter(element => {
                    return (
                        normalize(element.innerText)
                        === String(nextPageNumber)
                    );
                });

                const arrowLabels = new Set([
                    ">>",
                    ">",
                    "»",
                    "›",
                    "tiếp",
                    "sau",
                    "next"
                ]);

                const arrows = all.filter(element => {
                    const text = normalize(
                        element.innerText
                    ).toLowerCase();

                    const aria = (
                        element.getAttribute("aria-label")
                        || ""
                    ).toLowerCase();

                    const title = (
                        element.getAttribute("title")
                        || ""
                    ).toLowerCase();

                    return (
                        arrowLabels.has(text) ||
                        /next|tiếp|sau/.test(aria) ||
                        /next|tiếp|sau/.test(title)
                    );
                });

                const candidates = exact.length
                    ? exact
                    : arrows;

                if (!candidates.length) {
                    return false;
                }

                candidates.sort(
                    (a, b) =>
                        paginationScore(b)
                        - paginationScore(a)
                );

                const target = clickableAncestor(
                    candidates[0]
                );

                target.setAttribute(
                    "data-edu-crawl-next",
                    "1"
                );

                return true;
            }
            """,
            next_page_number,
        )
    )


def click_next_page(
    page: Page,
    site: SiteConfig,
    current_page_number: int,
    timeout_ms: int,
    debug_dir: Path,
) -> bool:
    before_signature = page_signature(page, site)
    before_url = page.url

    page.evaluate(
        "window.scrollTo(0, document.body.scrollHeight)"
    )
    page.wait_for_timeout(800)

    if not mark_pagination_target(
        page,
        current_page_number + 1,
    ):
        return False

    target = page.locator(
        "[data-edu-crawl-next='1']"
    ).first

    if not locator_is_usable(target):
        return False

    try:
        target.scroll_into_view_if_needed(
            timeout=5_000
        )
        target.click(timeout=10_000)
    except Exception:
        try:
            target.click(
                timeout=5_000,
                force=True,
            )
        except Exception:
            return False

    deadline = time.time() + timeout_ms / 1000

    while time.time() < deadline:
        page.wait_for_timeout(500)

        if is_human_challenge(page):
            # CAPTCHA có thể tái xuất hiện khi chuyển trang.
            return False

        after_signature = page_signature(
            page,
            site,
        )

        if (
            after_signature
            and after_signature != before_signature
        ):
            return True

        if page.url != before_url:
            try:
                wait_page_ready(page, timeout_ms)
            except Exception:
                pass

            after_signature = page_signature(
                page,
                site,
            )

            if after_signature:
                return True

    save_debug(
        page,
        debug_dir,
        f"{site.key}_next_timeout_page_{current_page_number}",
    )

    return False


def collect_detail_urls(
    page: Page,
    site: SiteConfig,
    max_pages: int,
    max_docs: int,
    timeout_ms: int,
    headed: bool,
    debug_dir: Path,
) -> list[str]:
    page.goto(
        site.start_url,
        wait_until="domcontentloaded",
        timeout=timeout_ms,
    )
    wait_page_ready(page, timeout_ms)

    wait_for_document_list(
        page=page,
        site=site,
        timeout_ms=timeout_ms,
        headed=headed,
        debug_dir=debug_dir,
    )

    all_urls: list[str] = []
    seen: set[str] = set()
    page_number = 1

    while True:
        urls = extract_detail_urls(
            page,
            site,
        )

        new_urls = [
            url
            for url in urls
            if url not in seen
        ]

        for url in new_urls:
            seen.add(url)
            all_urls.append(url)

            if (
                max_docs > 0
                and len(all_urls) >= max_docs
            ):
                break

        print(
            f"[{site.key}] Trang {page_number}: "
            f"{len(urls)} URL, mới {len(new_urls)}, "
            f"tổng {len(all_urls)}."
        )

        if (
            max_docs > 0
            and len(all_urls) >= max_docs
        ):
            break

        if (
            max_pages > 0
            and page_number >= max_pages
        ):
            break

        moved = click_next_page(
            page=page,
            site=site,
            current_page_number=page_number,
            timeout_ms=timeout_ms,
            debug_dir=debug_dir,
        )

        if not moved:
            if is_human_challenge(page):
                if not headed:
                    raise RuntimeError(
                        f"{site.name} yêu cầu CAPTCHA khi "
                        "chuyển trang. Chạy lại với --headed."
                    )

                print(
                    f"{site.name} yêu cầu CAPTCHA khi "
                    "chuyển trang."
                )
                input(
                    "Giải CAPTCHA, trở lại danh sách rồi "
                    "nhấn Enter... "
                )

                wait_page_ready(page, timeout_ms)

                moved = click_next_page(
                    page=page,
                    site=site,
                    current_page_number=page_number,
                    timeout_ms=timeout_ms,
                    debug_dir=debug_dir,
                )

            if not moved:
                print(
                    f"[{site.key}] Không tìm thấy hoặc "
                    "không bấm được trang tiếp theo."
                )
                break

        page_number += 1

    return all_urls


def check_robots(
    context: BrowserContext,
    site: SiteConfig,
    policy: str,
    timeout_ms: int,
) -> urllib.robotparser.RobotFileParser | None:
    if policy == "off":
        return None

    try:
        response = context.request.get(
            site.robots_url,
            timeout=timeout_ms,
            fail_on_status_code=False,
        )

        if not response.ok:
            raise RuntimeError(
                f"HTTP {response.status}"
            )

        content_type = response.headers.get(
            "content-type",
            "",
        ).lower()

        text = response.text()

        if (
            "text/html" in content_type
            and "<html" in text.lower()
        ):
            raise RuntimeError(
                "robots.txt trả về HTML/challenge."
            )

        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(site.robots_url)
        parser.parse(text.splitlines())

        if not parser.can_fetch(
            USER_AGENT,
            site.start_url,
        ):
            raise RuntimeError(
                f"robots.txt không cho phép crawl "
                f"{site.start_url}"
            )

        return parser

    except Exception as error:
        if policy == "strict":
            raise RuntimeError(
                f"Không kiểm tra được robots.txt: {error}"
            ) from error

        print(
            f"CẢNH BÁO [{site.key}]: "
            f"Không kiểm tra được robots.txt: {error}"
        )
        print(
            "Tiếp tục vì --robots-policy=warn."
        )

        return None


def ensure_robots_allowed(
    parser: urllib.robotparser.RobotFileParser | None,
    url: str,
) -> None:
    if (
        parser is not None
        and not parser.can_fetch(USER_AGENT, url)
    ):
        raise RuntimeError(
            f"robots.txt không cho phép crawl: {url}"
        )


def extract_attachment_candidates(
    page: Page,
) -> list[dict[str, str]]:
    """
    Lấy các liên kết "Tải file" trên trang chi tiết.

    Link file thường trỏ sang quantrisgd.shieldix.app nhưng crawler
    không hard-code host, chỉ dựa vào text và extension.
    """
    raw = page.evaluate(
        """
        () => {
            const result = [];

            for (const element of document.querySelectorAll(
                "a[href],button,[role='button']"
            )) {
                const text = (
                    element.innerText
                    || element.textContent
                    || ""
                ).replace(/\\s+/g, " ").trim();

                const values = [
                    element.getAttribute("href"),
                    element.getAttribute("data-href"),
                    element.getAttribute("data-url"),
                    element.getAttribute("onclick")
                ].filter(Boolean);

                for (const value of values) {
                    result.push({
                        value,
                        text
                    });
                }
            }

            return result;
        }
        """
    )

    candidates: list[dict[str, str]] = []
    seen: set[str] = set()

    for item in raw:
        value = item.get("value", "")
        label = clean_inline_text(
            item.get("text", "")
        )

        if not isinstance(value, str):
            continue

        value_lower = unquote(
            f"{value} {label}"
        ).lower()

        is_download_label = bool(
            re.search(
                r"\btải\s*file\b|\btai\s*file\b|"
                r"file\s*đính\s*kèm|file\s*dinh\s*kem",
                value_lower,
                re.IGNORECASE,
            )
        )

        has_supported_extension = bool(
            re.search(
                r"\.(pdf|docx|doc)(?:$|[?&#])",
                value_lower,
                re.IGNORECASE,
            )
        )

        if not (
            is_download_label
            or has_supported_extension
        ):
            continue

        match = re.search(
            r"(https?://[^\"'()<>\s]+|"
            r"/[^\"'()<>\s]+)",
            value,
        )

        candidate = (
            match.group(0)
            if match
            else value
        )

        absolute_url = urljoin(
            page.url,
            candidate,
        ).rstrip(".,;)]}")

        if absolute_url not in seen:
            seen.add(absolute_url)
            candidates.append(
                {
                    "url": absolute_url,
                    "label": label,
                }
            )

    return candidates


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
            "File không phải DOCX/ZIP hợp lệ."
        )

    document = Document(BytesIO(content))
    parts: list[str] = []

    for block in iter_docx_blocks(document):
        if isinstance(block, Paragraph):
            text = clean_inline_text(
                block.text
            )

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
                    parts.append(
                        " | ".join(compact)
                    )

    return clean_text("\n".join(parts))


def extract_pdf_text(content: bytes) -> str:
    if not content.lstrip().startswith(
        PDF_MAGIC
    ):
        raise ValueError(
            "File không phải PDF hợp lệ."
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

    result = clean_text(
        "\n\n".join(pages)
    )

    if len(result) < 100:
        raise ValueError(
            "PDF gần như không có lớp text; "
            "có thể là PDF scan."
        )

    return result


def response_bytes_with_limit(
    response: APIResponse,
    max_file_mb: float,
) -> bytes:
    if not response.ok:
        raise RuntimeError(
            f"Download HTTP {response.status}: "
            f"{response.status_text}"
        )

    content = response.body()
    limit = int(max_file_mb * 1024 * 1024)

    if len(content) > limit:
        raise ValueError(
            f"File vượt quá {max_file_mb:g} MB."
        )

    return content


def detect_file_type(
    content: bytes,
    response: APIResponse,
    url: str,
    label: str,
) -> str:
    headers = {
        key.lower(): value
        for key, value in response.headers.items()
    }

    content_type = headers.get(
        "content-type",
        "",
    ).lower()

    content_disposition = unquote(
        headers.get(
            "content-disposition",
            "",
        )
    ).lower()

    combined = unquote(
        f"{url} {label} "
        f"{content_type} {content_disposition}"
    ).lower()

    stripped = content.lstrip()

    if stripped.startswith(PDF_MAGIC):
        return "pdf"

    if content.startswith(DOCX_MAGIC):
        return "docx"

    if content.startswith(LEGACY_DOC_MAGIC):
        return "doc"

    if "application/pdf" in combined:
        return "pdf"

    if (
        "wordprocessingml" in combined
        or ".docx" in combined
    ):
        return "docx"

    if (
        "application/msword" in combined
        or re.search(
            r"\.doc(?:$|[?&#\s])",
            combined,
        )
    ):
        return "doc"

    if "text/html" in content_type:
        return "html"

    return "unknown"


def download_and_extract(
    context: BrowserContext,
    attachment: dict[str, str],
    referer: str,
    timeout_ms: int,
    max_file_mb: float,
) -> tuple[str, str]:
    url = attachment["url"]
    label = attachment.get("label", "")

    response = context.request.get(
        url,
        headers={
            "Referer": referer,
            "Accept": (
                "application/pdf,"
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document,"
                "application/msword,"
                "application/octet-stream,*/*"
            ),
        },
        timeout=timeout_ms,
        fail_on_status_code=False,
    )

    content = response_bytes_with_limit(
        response,
        max_file_mb=max_file_mb,
    )

    file_type = detect_file_type(
        content=content,
        response=response,
        url=url,
        label=label,
    )

    if file_type == "pdf":
        return extract_pdf_text(content), "pdf"

    if file_type == "docx":
        return extract_docx_text(content), "docx"

    if file_type == "doc":
        raise ValueError(
            "File dùng định dạng .doc cũ; "
            "python-docx không đọc được."
        )

    if file_type == "html":
        preview = clean_inline_text(
            content[:500].decode(
                "utf-8",
                errors="replace",
            )
        )

        raise ValueError(
            "Link tải trả về HTML/challenge thay vì file. "
            f"Preview: {preview[:150]}"
        )

    raise ValueError(
        "Không xác định được loại file tải về."
    )


def deduplicate_texts(
    texts: list[str],
) -> list[str]:
    unique: list[str] = []
    seen_hashes: set[str] = set()

    for text in texts:
        normalized = clean_text(text)

        if not normalized:
            continue

        digest = hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()

        if digest in seen_hashes:
            continue

        seen_hashes.add(digest)
        unique.append(normalized)

    return unique


def extract_html_fallback(
    page: Page,
) -> str:
    """
    Chỉ dùng khi bật --allow-html-fallback.
    Trang chi tiết thường chỉ có metadata, không phải toàn văn.
    """
    text = page.evaluate(
        """
        () => {
            const selectors = [
                "main",
                "article",
                "[role='main']",
                ".detail",
                ".document-detail",
                ".vanban-detail",
                ".content"
            ];

            let root = null;

            for (const selector of selectors) {
                const candidate =
                    document.querySelector(selector);

                if (
                    candidate
                    && candidate.innerText
                    && candidate.innerText.trim().length > 200
                ) {
                    root = candidate;
                    break;
                }
            }

            if (!root) {
                root = document.body;
            }

            const clone = root.cloneNode(true);

            clone.querySelectorAll(
                [
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
                    "textarea"
                ].join(",")
            ).forEach(element => element.remove());

            return clone.innerText || "";
        }
        """
    )

    return clean_text(text)


def crawl_one_document(
    context: BrowserContext,
    site: SiteConfig,
    detail_url: str,
    robots_parser: urllib.robotparser.RobotFileParser | None,
    timeout_ms: int,
    min_chars: int,
    max_files_per_doc: int,
    max_file_mb: float,
    allow_html_fallback: bool,
    debug_dir: Path,
    headed: bool,
) -> tuple[str, str]:
    ensure_robots_allowed(
        robots_parser,
        detail_url,
    )

    page = context.new_page()

    try:
        page.goto(
            detail_url,
            wait_until="domcontentloaded",
            timeout=timeout_ms,
        )
        wait_page_ready(page, timeout_ms)

        if is_human_challenge(page):
            if not headed:
                raise RuntimeError(
                    f"{site.name} yêu cầu CAPTCHA trên "
                    "trang chi tiết. Chạy lại với --headed."
                )

            print(
                f"{site.name} yêu cầu CAPTCHA tại "
                f"{detail_url}"
            )
            input(
                "Giải CAPTCHA rồi nhấn Enter... "
            )
            wait_page_ready(page, timeout_ms)

        attachments = extract_attachment_candidates(
            page
        )

        if max_files_per_doc > 0:
            attachments = attachments[
                :max_files_per_doc
            ]

        texts: list[str] = []
        source_types: list[str] = []
        errors: list[str] = []

        for file_index, attachment in enumerate(
            attachments,
            start=1,
        ):
            try:
                text, source_type = download_and_extract(
                    context=context,
                    attachment=attachment,
                    referer=detail_url,
                    timeout_ms=timeout_ms,
                    max_file_mb=max_file_mb,
                )

                if len(text) < min_chars:
                    raise ValueError(
                        f"Text chỉ có "
                        f"{len(text)} ký tự."
                    )

                texts.append(text)
                source_types.append(source_type)

            except Exception as error:
                errors.append(
                    f"file {file_index}: {error}"
                )

        unique_texts = deduplicate_texts(
            texts
        )

        if unique_texts:
            return (
                clean_text(
                    "\n\n".join(unique_texts)
                ),
                "+".join(
                    dict.fromkeys(source_types)
                ),
            )

        if allow_html_fallback:
            html_text = extract_html_fallback(
                page
            )

            if len(html_text) >= min_chars:
                return html_text, "html"

        save_debug(
            page,
            debug_dir,
            f"{site.key}_detail_failed_"
            f"{make_document_id(detail_url, site)}",
        )

        if not attachments:
            errors.append(
                "Không tìm thấy link tải file."
            )

        if not errors:
            errors.append(
                "Không có nội dung đạt yêu cầu."
            )

        raise RuntimeError(
            " | ".join(errors)
        )

    finally:
        page.close()


def crawl_site(
    playwright: Any,
    site: SiteConfig,
    args: argparse.Namespace,
) -> None:
    output_path = (
        Path(args.output_dir)
        / f"{site.id_prefix}.jsonl"
    )
    error_path = (
        Path(args.output_dir)
        / f"{site.id_prefix}_errors.jsonl"
    )
    debug_dir = (
        Path(args.debug_dir)
        / site.key
    )
    profile_dir = (
        Path(args.profile_root)
        / site.key
    )

    profile_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_ids = load_existing_ids(
        output_path
    )

    print(
        f"\n=== {site.name} ==="
    )
    print(
        f"Đã có {len(existing_ids)} "
        f"bản ghi trong {output_path}."
    )

    timeout_ms = int(
        args.timeout * 1_000
    )

    launch_kwargs: dict[str, Any] = {
        "user_data_dir": str(profile_dir),
        "headless": not args.headed,
        "locale": "vi-VN",
        "viewport": {
            "width": 1440,
            "height": 1_000,
        },
        "user_agent": USER_AGENT,
        "ignore_https_errors": (
            args.ignore_https_errors
        ),
    }

    if args.channel:
        launch_kwargs["channel"] = (
            args.channel
        )

    context = (
        playwright.chromium
        .launch_persistent_context(
            **launch_kwargs
        )
    )

    robots_parser = check_robots(
        context=context,
        site=site,
        policy=args.robots_policy,
        timeout_ms=timeout_ms,
    )

    pages = context.pages
    listing_page = (
        pages[0]
        if pages
        else context.new_page()
    )

    saved = 0
    skipped = 0
    failed = 0

    try:
        detail_urls = collect_detail_urls(
            page=listing_page,
            site=site,
            max_pages=args.max_pages,
            max_docs=args.max_docs,
            timeout_ms=timeout_ms,
            headed=args.headed,
            debug_dir=debug_dir,
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
                    context=context,
                    site=site,
                    detail_url=detail_url,
                    robots_parser=robots_parser,
                    timeout_ms=timeout_ms,
                    min_chars=args.min_chars,
                    max_files_per_doc=(
                        args.max_files_per_doc
                    ),
                    max_file_mb=args.max_file_mb,
                    allow_html_fallback=(
                        args.allow_html_fallback
                    ),
                    debug_dir=debug_dir,
                    headed=args.headed,
                )

                append_jsonl(
                    output_path,
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
                    error_path,
                    {
                        "id": document_id,
                        "url": detail_url,
                        "error": repr(error),
                    },
                )

                print(
                    f"[{site.key} "
                    f"{index}/{len(detail_urls)}] "
                    f"ERROR {document_id}: {error}"
                )

            time.sleep(
                max(
                    0.5,
                    args.delay
                    + random.uniform(
                        0,
                        args.jitter,
                    ),
                )
            )

    finally:
        context.close()

    print(
        f"\nHoàn tất {site.name}"
    )
    print(f"- Lưu mới: {saved}")
    print(f"- Bỏ qua: {skipped}")
    print(f"- Lỗi: {failed}")
    print(f"- Output: {output_path}")
    print(f"- Error log: {error_path}")
    print(f"- Debug: {debug_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--site",
        choices=("hanoi", "hcm", "both"),
        default="both",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
    )
    parser.add_argument(
        "--debug-dir",
        default="data/debug/so_gddt",
    )
    parser.add_argument(
        "--profile-root",
        default="data/browser_profiles/so_gddt",
        help=(
            "Thư mục lưu cookie/browser profile. "
            "Giữ nguyên đường dẫn này để tái sử dụng "
            "phiên CAPTCHA."
        ),
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=1,
        help=(
            "Số trang danh sách tối đa; "
            "0 là không giới hạn."
        ),
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=10,
        help=(
            "Tổng số văn bản tối đa cho mỗi nguồn; "
            "0 là không giới hạn."
        ),
    )
    parser.add_argument(
        "--max-files-per-doc",
        type=int,
        default=10,
        help=(
            "Số file đính kèm tối đa cho một văn bản; "
            "0 là không giới hạn."
        ),
    )
    parser.add_argument(
        "--max-file-mb",
        type=float,
        default=100,
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.5,
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=90,
        help="Timeout theo giây.",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help=(
            "Hiện Chromium. Bắt buộc cho lần đầu "
            "khi TP.HCM yêu cầu CAPTCHA."
        ),
    )
    parser.add_argument(
        "--channel",
        default="chromium",
        help=(
            "Browser channel của Playwright. "
            "Để chuỗi rỗng nếu muốn dùng executable "
            "mặc định."
        ),
    )
    parser.add_argument(
        "--ignore-https-errors",
        action="store_true",
        help=(
            "Bỏ qua lỗi chứng chỉ HTTPS trong browser "
            "và API request context."
        ),
    )
    parser.add_argument(
        "--allow-html-fallback",
        action="store_true",
        help=(
            "Cho phép lưu text HTML khi không đọc được "
            "file. Mặc định tắt vì HTML thường chỉ có "
            "metadata."
        ),
    )
    parser.add_argument(
        "--robots-policy",
        choices=("strict", "warn", "off"),
        default="warn",
    )

    args = parser.parse_args()
    args.channel = (
        args.channel.strip()
        if args.channel
        else None
    )

    return args


def main() -> None:
    args = parse_args()

    if args.site == "both":
        selected_sites = [
            SITES["hanoi"],
            SITES["hcm"],
        ]
    else:
        selected_sites = [
            SITES[args.site]
        ]

    with sync_playwright() as playwright:
        for site in selected_sites:
            crawl_site(
                playwright=playwright,
                site=site,
                args=args,
            )


if __name__ == "__main__":
    main()