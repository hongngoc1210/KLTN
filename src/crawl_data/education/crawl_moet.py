from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
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
    # Tương thích môi trường cài PyMuPDF phiên bản cũ.
    import fitz as pymupdf  # type: ignore


BASE_URL = "https://tuyensinh.moet.gov.vn"
START_URL = "https://tuyensinh.moet.gov.vn/ts/tin-tuc"
DOMAIN = "education"

DETAIL_PREFIX = "/ts/tin-tuc/"

UUID_RE = re.compile(
    r"([0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12})"
)

FILE_EXT_RE = re.compile(
    r"\.(docx|pdf)(?:$|[?&#])",
    re.IGNORECASE,
)


def clean_inline_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    """
    Làm sạch cơ bản nhưng chưa chia nhỏ thành context.
    """
    text = text.replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = [
        re.sub(r"[ \t]+", " ", line).strip()
        for line in text.splitlines()
    ]

    result: list[str] = []
    previous_blank = False

    for line in lines:
        if not line:
            if result and not previous_blank:
                result.append("")
            previous_blank = True
            continue

        result.append(line)
        previous_blank = False

    return "\n".join(result).strip()


def make_document_id(url: str) -> str:
    """
    URL chi tiết thường kết thúc bằng UUID.
    """
    matches = UUID_RE.findall(url)

    if matches:
        return f"education_moet_news_{matches[-1].lower()}"

    digest = hashlib.sha256(
        url.encode("utf-8")
    ).hexdigest()[:24]

    return f"education_moet_news_{digest}"


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


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


def wait_page_ready(page: Page, timeout_ms: int) -> None:
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
        # Một số trang có analytics/request chạy nền liên tục.
        pass

    page.wait_for_timeout(2_000)


def save_debug(
    page: Page,
    debug_dir: Path,
    name: str,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)

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


def normalize_detail_url(raw_url: str) -> str | None:
    if not raw_url:
        return None

    absolute_url = urljoin(BASE_URL, raw_url)
    parsed = urlparse(absolute_url)

    if parsed.netloc.lower() != urlparse(BASE_URL).netloc.lower():
        return None

    path = parsed.path.rstrip("/")

    # Loại chính trang danh sách /ts/tin-tuc.
    if not path.startswith(f"{DETAIL_PREFIX}"):
        return None

    if path == START_URL.replace(BASE_URL, "").rstrip("/"):
        return None

    # Trang chi tiết hiện có UUID trong slug.
    if not UUID_RE.search(path):
        return None

    return parsed._replace(
        query="",
        fragment="",
    ).geturl()


def extract_detail_urls(page: Page) -> list[str]:
    """
    Tìm URL chi tiết trong href, routerLink, data-url, onclick
    và cả HTML đã render.
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

                    if (
                        value &&
                        value.includes("/ts/tin-tuc/")
                    ) {
                        values.push(value);
                    }
                }
            }

            const html = document.documentElement.innerHTML;

            const matches = html.match(
                /(?:https?:\\/\\/tuyensinh\\.moet\\.gov\\.vn)?\\/ts\\/tin-tuc\\/[^"'<>\\s]+/g
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

        # Trích URL trong trường hợp onclick chứa JavaScript.
        match = re.search(
            r"(?:https?://tuyensinh\.moet\.gov\.vn)?"
            r"/ts/tin-tuc/[^\"'()<>\s]+",
            raw,
            re.IGNORECASE,
        )

        candidate = match.group(0) if match else raw
        url = normalize_detail_url(candidate)

        if url and url not in seen:
            seen.add(url)
            urls.append(url)

    return urls


def page_signature(page: Page) -> list[str]:
    """
    Dùng các URL đầu trang để phát hiện danh sách đã chuyển trang.
    """
    return extract_detail_urls(page)[:5]


def locator_is_usable(locator: Locator) -> bool:
    try:
        if locator.count() == 0:
            return False

        target = locator.first

        if not target.is_visible():
            return False

        disabled = target.get_attribute("disabled")
        aria_disabled = target.get_attribute("aria-disabled")

        if disabled is not None or aria_disabled == "true":
            return False

        class_name = target.get_attribute("class") or ""

        if re.search(r"\bdisabled\b", class_name, re.IGNORECASE):
            return False

        return True
    except Exception:
        return False


def click_next_page(
    page: Page,
    current_page_number: int,
    timeout_ms: int,
) -> bool:
    """
    Phân trang có thể thay đổi URL hoặc chỉ cập nhật DOM.
    """
    before_signature = page_signature(page)
    before_url = page.url
    next_number = str(current_page_number + 1)

    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    page.wait_for_timeout(800)

    selectors = [
        f"button:text-is('{next_number}')",
        f"a:text-is('{next_number}')",
        f"[role='button']:text-is('{next_number}')",
        "button[aria-label*='next' i]",
        "a[aria-label*='next' i]",
        "button[title*='next' i]",
        "a[title*='next' i]",
        "button[aria-label*='sau' i]",
        "a[aria-label*='sau' i]",
        "button[title*='sau' i]",
        "a[title*='sau' i]",
        "button:has-text('Tiếp')",
        "a:has-text('Tiếp')",
        "button:has-text('Sau')",
        "a:has-text('Sau')",
        "li.ant-pagination-next button",
        "li.ant-pagination-next a",
        ".p-paginator-next",
        ".pagination-next",
        "button.next",
        "a.next",
    ]

    clicked = False

    for selector in selectors:
        locator = page.locator(selector)

        if not locator_is_usable(locator):
            continue

        try:
            locator.first.scroll_into_view_if_needed()
            locator.first.click(timeout=10_000)
            clicked = True
            break
        except Exception:
            continue

    if not clicked:
        # Fallback: tìm phần tử số trang bằng JavaScript.
        clicked = bool(
            page.evaluate(
                """
                nextNumber => {
                    const normalize = value =>
                        (value || "").replace(/\\s+/g, " ").trim();

                    const visible = element => {
                        const style = getComputedStyle(element);
                        const rect = element.getBoundingClientRect();

                        return (
                            style.display !== "none" &&
                            style.visibility !== "hidden" &&
                            rect.width > 3 &&
                            rect.height > 3
                        );
                    };

                    const candidates = [
                        ...document.querySelectorAll(
                            "a,button,li,span,[role='button']"
                        )
                    ].filter(element => {
                        return (
                            visible(element) &&
                            normalize(element.innerText) === nextNumber
                        );
                    });

                    candidates.sort((a, b) => {
                        return (
                            b.getBoundingClientRect().top -
                            a.getBoundingClientRect().top
                        );
                    });

                    for (const element of candidates) {
                        let node = element;

                        for (
                            let depth = 0;
                            depth < 5 && node;
                            depth += 1
                        ) {
                            const style = getComputedStyle(node);
                            const role = node.getAttribute("role");

                            if (
                                node.tagName === "A" ||
                                node.tagName === "BUTTON" ||
                                node.hasAttribute("onclick") ||
                                role === "button" ||
                                style.cursor === "pointer"
                            ) {
                                node.click();
                                return true;
                            }

                            node = node.parentElement;
                        }

                        element.click();
                        return true;
                    }

                    return false;
                }
                """,
                next_number,
            )
        )

    if not clicked:
        return False

    deadline = time.time() + timeout_ms / 1000

    while time.time() < deadline:
        page.wait_for_timeout(500)

        after_signature = page_signature(page)

        if (
            after_signature
            and after_signature != before_signature
        ):
            return True

        if page.url != before_url:
            wait_page_ready(page, timeout_ms)
            return True

    return False


def collect_detail_urls(
    page: Page,
    max_pages: int,
    max_docs: int,
    timeout_ms: int,
    debug_dir: Path,
) -> list[str]:
    page.goto(
        START_URL,
        wait_until="domcontentloaded",
        timeout=timeout_ms,
    )
    wait_page_ready(page, timeout_ms)

    all_urls: list[str] = []
    seen: set[str] = set()
    page_number = 1

    while True:
        urls = extract_detail_urls(page)

        if not urls:
            save_debug(
                page,
                debug_dir,
                f"no_detail_urls_page_{page_number}",
            )
            print(
                f"Trang {page_number}: không tìm thấy URL văn bản. "
                f"Đã lưu debug tại {debug_dir}."
            )
            break

        new_urls = [
            url
            for url in urls
            if url not in seen
        ]

        for url in new_urls:
            seen.add(url)
            all_urls.append(url)

            if max_docs > 0 and len(all_urls) >= max_docs:
                break

        print(
            f"Trang {page_number}: "
            f"{len(urls)} URL, mới {len(new_urls)}, "
            f"tổng {len(all_urls)}."
        )

        if max_docs > 0 and len(all_urls) >= max_docs:
            break

        if max_pages > 0 and page_number >= max_pages:
            break

        if not click_next_page(
            page=page,
            current_page_number=page_number,
            timeout_ms=timeout_ms,
        ):
            save_debug(
                page,
                debug_dir,
                f"cannot_next_after_page_{page_number}",
            )
            print(
                "Không tìm thấy hoặc không bấm được trang tiếp theo."
            )
            break

        page_number += 1

    return all_urls


def classify_attachment(
    url: str,
    link_text: str,
) -> str | None:
    combined = unquote(f"{url} {link_text}").lower()

    if ".docx" in combined:
        return "docx"

    if ".pdf" in combined:
        return "pdf"

    # Một số endpoint download không để extension trong path,
    # nhưng text của nút có dạng "Tải file DOCX/PDF".
    if "docx" in combined or "word" in combined:
        return "docx"

    if "pdf" in combined:
        return "pdf"

    return None


def extract_attachment_urls(
    page: Page,
) -> dict[str, str]:
    """
    Lấy file đính kèm từ href/data-url/routerLink.
    """
    raw_links = page.evaluate(
        """
        () => {
            const attrs = [
                "href",
                "data-href",
                "data-url",
                "data-download",
                "routerlink",
                "onclick"
            ];

            const result = [];

            for (const element of document.querySelectorAll(
                "a,button,[role='button']"
            )) {
                const text = (
                    element.innerText ||
                    element.textContent ||
                    ""
                ).trim();

                for (const attr of attrs) {
                    const value = element.getAttribute(attr);

                    if (value) {
                        result.push({
                            value,
                            text
                        });
                    }
                }
            }

            return result;
        }
        """
    )

    attachments: dict[str, str] = {}

    for item in raw_links:
        raw_url = item.get("value", "")
        link_text = item.get("text", "")

        if not isinstance(raw_url, str):
            continue

        # Trích URL nếu nằm trong onclick.
        url_match = re.search(
            r"(https?://[^\"'()<>\s]+|"
            r"/[^\"'()<>\s]+)",
            raw_url,
        )

        candidate = (
            url_match.group(0)
            if url_match
            else raw_url
        )

        absolute_url = urljoin(page.url, candidate)
        file_type = classify_attachment(
            absolute_url,
            link_text,
        )

        if file_type and file_type not in attachments:
            attachments[file_type] = absolute_url

    return attachments


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
    if not content.startswith(b"PK"):
        raise ValueError(
            "File tải về không phải DOCX/ZIP hợp lệ."
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

                compact_cells: list[str] = []

                for cell_text in cells:
                    if not cell_text:
                        continue

                    if (
                        not compact_cells
                        or compact_cells[-1] != cell_text
                    ):
                        compact_cells.append(cell_text)

                if compact_cells:
                    parts.append(" | ".join(compact_cells))

    return clean_text("\n".join(parts))


def extract_pdf_text(content: bytes) -> str:
    if not content.lstrip().startswith(b"%PDF"):
        raise ValueError(
            "File tải về không phải PDF hợp lệ."
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


def response_bytes(response: APIResponse) -> bytes:
    if not response.ok:
        raise RuntimeError(
            f"Download HTTP {response.status}: "
            f"{response.status_text}"
        )

    return response.body()


def detect_content_type(
    response: APIResponse,
    url: str,
    expected_type: str,
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
        headers.get("content-disposition", "")
    ).lower()

    combined = (
        f"{url} {content_type} "
        f"{content_disposition} {expected_type}"
    ).lower()

    if (
        "wordprocessingml" in combined
        or ".docx" in combined
    ):
        return "docx"

    if (
        "application/pdf" in combined
        or ".pdf" in combined
    ):
        return "pdf"

    return expected_type


def download_and_extract(
    context: BrowserContext,
    attachment_url: str,
    expected_type: str,
    referer: str,
    timeout_ms: int,
) -> str:
    """
    BrowserContext.request dùng chung cookie với browser context,
    hữu ích khi website/CDN chặn requests độc lập.
    """
    response = context.request.get(
        attachment_url,
        headers={
            "Referer": referer,
            "Accept": (
                "application/pdf,"
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document,"
                "application/octet-stream,*/*"
            ),
        },
        timeout=timeout_ms,
        fail_on_status_code=False,
    )

    content = response_bytes(response)
    detected_type = detect_content_type(
        response=response,
        url=attachment_url,
        expected_type=expected_type,
    )

    if detected_type == "docx":
        return extract_docx_text(content)

    if detected_type == "pdf":
        return extract_pdf_text(content)

    raise ValueError(
        f"Không nhận diện được loại file: {detected_type}"
    )


def extract_html_fallback(page: Page) -> str:
    """
    Fallback khi trang không có file đính kèm hoặc file tải lỗi.
    """
    text = page.evaluate(
        """
        () => {
            const selectors = [
                "main",
                "article",
                "[role='main']",
                ".document-detail",
                ".detail-content",
                ".content-detail",
                ".van-ban-chi-tiet",
                ".content"
            ];

            let root = null;

            for (const selector of selectors) {
                const candidate = document.querySelector(selector);

                if (
                    candidate &&
                    candidate.innerText &&
                    candidate.innerText.trim().length > 200
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
                    "button",
                    "input",
                    "select",
                    "textarea",
                    ".grecaptcha-badge"
                ].join(",")
            ).forEach(element => element.remove());

            return clone.innerText || "";
        }
        """
    )

    return clean_text(text)


def crawl_one_document(
    context: BrowserContext,
    detail_url: str,
    timeout_ms: int,
    min_chars: int,
    debug_dir: Path,
    document_id: str,
) -> tuple[str, str]:
    page = context.new_page()

    try:
        page.goto(
            detail_url,
            wait_until="domcontentloaded",
            timeout=timeout_ms,
        )
        wait_page_ready(page, timeout_ms)

        if "/account/login" in page.url:
            raise RuntimeError(
                "Website chuyển sang trang đăng nhập."
            )

        attachments = extract_attachment_urls(page)
        extraction_errors: list[str] = []

        # Ưu tiên DOCX, fallback PDF.
        for file_type in ("docx", "pdf"):
            attachment_url = attachments.get(file_type)

            if not attachment_url:
                continue

            try:
                text = download_and_extract(
                    context=context,
                    attachment_url=attachment_url,
                    expected_type=file_type,
                    referer=detail_url,
                    timeout_ms=timeout_ms,
                )

                if len(text) < min_chars:
                    raise ValueError(
                        f"Text quá ngắn: {len(text)} ký tự."
                    )

                return text, file_type

            except Exception as error:
                extraction_errors.append(
                    f"{file_type}: {error}"
                )

        # Fallback text HTML trên trang chi tiết.
        html_text = extract_html_fallback(page)

        if len(html_text) >= min_chars:
            return html_text, "html"

        save_debug(
            page,
            debug_dir,
            f"detail_failed_{document_id}",
        )

        message = (
            "Không trích được nội dung. "
            + " | ".join(extraction_errors)
        )

        if not attachments:
            message += " | Không tìm thấy DOCX/PDF."

        message += (
            f" | HTML chỉ có {len(html_text)} ký tự."
        )

        raise RuntimeError(message)

    finally:
        page.close()


def crawl(args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    error_path = Path(args.error_output)
    debug_dir = Path(args.debug_dir)

    existing_ids = load_existing_ids(output_path)
    print(
        f"Đã có {len(existing_ids)} bản ghi "
        f"trong file output."
    )

    timeout_ms = int(args.timeout * 1_000)

    with sync_playwright() as playwright:
        launch_kwargs: dict[str, Any] = {
            "headless": not args.headed,
        }

        if args.channel:
            launch_kwargs["channel"] = args.channel

        browser = playwright.chromium.launch(
            **launch_kwargs,
        )

        context = browser.new_context(
            locale="vi-VN",
            viewport={
                "width": 1440,
                "height": 1_000,
            },
            ignore_https_errors=args.ignore_https_errors,
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )

        listing_page = context.new_page()

        try:
            detail_urls = collect_detail_urls(
                page=listing_page,
                max_pages=args.max_pages,
                max_docs=args.max_docs,
                timeout_ms=timeout_ms,
                debug_dir=debug_dir,
            )
        finally:
            listing_page.close()

        print(
            f"Sẽ xử lý {len(detail_urls)} URL chi tiết."
        )

        saved = 0
        skipped = 0
        failed = 0

        for index, detail_url in enumerate(
            detail_urls,
            start=1,
        ):
            document_id = make_document_id(detail_url)

            if document_id in existing_ids:
                skipped += 1
                print(
                    f"[{index}/{len(detail_urls)}] "
                    f"SKIP {document_id}"
                )
                continue

            try:
                text, source_type = crawl_one_document(
                    context=context,
                    detail_url=detail_url,
                    timeout_ms=timeout_ms,
                    min_chars=args.min_chars,
                    debug_dir=debug_dir,
                    document_id=document_id,
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
                    f"[{index}/{len(detail_urls)}] "
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
                    f"[{index}/{len(detail_urls)}] "
                    f"ERROR {document_id}: {error}"
                )

            time.sleep(
                max(
                    0.5,
                    args.delay
                    + random.uniform(0, args.jitter),
                )
            )

        context.close()
        browser.close()

    print("\nHoàn tất")
    print(f"- Lưu mới: {saved}")
    print(f"- Bỏ qua: {skipped}")
    print(f"- Lỗi: {failed}")
    print(f"- Output: {output_path}")
    print(f"- Error log: {error_path}")
    print(f"- Debug: {debug_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        default="data/raw/education_moet.jsonl",
    )
    parser.add_argument(
        "--error-output",
        default="data/raw/education_moet_errors.jsonl",
    )
    parser.add_argument(
        "--debug-dir",
        default="data/debug/tuyensinh_moet",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=1,
        help="Số trang danh sách tối đa; 0 là không giới hạn.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=10,
        help="Số văn bản tối đa; 0 là không giới hạn.",
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
        default=60,
        help="Timeout theo giây.",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Hiện cửa sổ Chromium để kiểm tra.",
    )
    parser.add_argument(
        "--channel",
        default="chromium",
        help=(
            "Playwright browser channel. "
            "Để trống để dùng executable mặc định."
        ),
    )
    parser.add_argument(
        "--ignore-https-errors",
        action="store_true",
        help=(
            "Bỏ qua lỗi chứng chỉ HTTPS trong browser context. "
            "Chỉ dùng khi CDN tài liệu gặp lỗi certificate."
        ),
    )

    args = parser.parse_args()

    if args.channel is not None:
        args.channel = args.channel.strip() or None

    return args


if __name__ == "__main__":
    crawl(parse_args())