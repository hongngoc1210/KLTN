from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
import urllib.robotparser
from io import BytesIO
from pathlib import Path
from typing import Iterator
from urllib.parse import parse_qs, urljoin, urlparse

# Ưu tiên certificate store của Windows/macOS/Linux.
# Nếu chưa cài truststore, Requests sẽ dùng certifi như bình thường.
try:
    import truststore

    truststore.inject_into_ssl()
    TRUSTSTORE_ENABLED = True
except ImportError:
    TRUSTSTORE_ENABLED = False

import requests
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
    # Tương thích các bản PyMuPDF cũ.
    import fitz as pymupdf  # type: ignore


BASE_URL = "https://vqa.moet.gov.vn"
DEFAULT_START_URL = f"{BASE_URL}/vi/laws/"
ROBOTS_URL = f"{BASE_URL}/robots.txt"
DOMAIN = "education"

DETAIL_PATH_RE = re.compile(
    r"^/vi/laws/detail/.+-(\d+)/?$",
    re.IGNORECASE,
)

DOWNLOAD_QUERY_RE = re.compile(
    r"(?:^|[?&])download=1(?:&|$)",
    re.IGNORECASE,
)

PAGINATION_RE = re.compile(
    r"(?:/page-\d+/?(?:$|[?#])|[?&](?:page|p)=\d+)",
    re.IGNORECASE,
)

PDF_MAGIC = b"%PDF"
DOCX_MAGIC = b"PK\x03\x04"
LEGACY_DOC_MAGIC = bytes.fromhex("D0CF11E0A1B11AE1")


def clean_inline_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    """
    Chuẩn hóa khoảng trắng nhưng chưa tách thành các context nhỏ.
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


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    return parsed._replace(fragment="").geturl()


def build_session(
    user_agent: str,
    retries: int,
    backoff: float,
    verify_ssl: bool,
) -> requests.Session:
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=backoff,
        status_forcelist=(408, 425, 429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=10,
        pool_maxsize=10,
    )

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
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

    # Thuộc tính nội bộ của crawler.
    session._crawler_verify_ssl = verify_ssl  # type: ignore[attr-defined]

    return session


def get_verify_ssl(session: requests.Session) -> bool:
    return bool(
        getattr(session, "_crawler_verify_ssl", True)
    )


def check_robots(
    session: requests.Session,
    start_url: str,
    policy: str,
    timeout: tuple[float, float],
) -> urllib.robotparser.RobotFileParser | None:
    """
    policy:
      strict: không đọc được robots.txt thì dừng.
      warn: cảnh báo rồi tiếp tục.
      off: không kiểm tra.
    """
    if policy == "off":
        return None

    parser = urllib.robotparser.RobotFileParser()
    parser.set_url(ROBOTS_URL)

    try:
        response = session.get(
            ROBOTS_URL,
            timeout=timeout,
            verify=get_verify_ssl(session),
        )
        response.raise_for_status()
        parser.parse(response.text.splitlines())
    except requests.RequestException as error:
        message = f"Không tải được robots.txt: {error}"

        if policy == "strict":
            raise RuntimeError(message) from error

        print(f"CẢNH BÁO: {message}")
        print("Tiếp tục vì --robots-policy=warn.")
        return None

    if not parser.can_fetch("*", start_url):
        raise RuntimeError(
            f"robots.txt không cho phép crawl: {start_url}"
        )

    return parser


def ensure_allowed(
    parser: urllib.robotparser.RobotFileParser | None,
    url: str,
) -> None:
    if parser is not None and not parser.can_fetch("*", url):
        raise RuntimeError(
            f"robots.txt không cho phép crawl: {url}"
        )


def fetch(
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
        stream=stream,
        allow_redirects=True,
        verify=get_verify_ssl(session),
    )
    response.raise_for_status()
    return response


def same_host(url: str) -> bool:
    return (
        (urlparse(url).hostname or "").lower()
        == (urlparse(BASE_URL).hostname or "").lower()
    )


def extract_detail_urls(
    html: bytes,
    page_url: str,
) -> list[str]:
    soup = BeautifulSoup(html, "lxml")

    urls: list[str] = []
    seen: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        absolute_url = normalize_url(
            urljoin(page_url, anchor["href"])
        )

        if not same_host(absolute_url):
            continue

        if not DETAIL_PATH_RE.match(
            urlparse(absolute_url).path
        ):
            continue

        if absolute_url not in seen:
            seen.add(absolute_url)
            urls.append(absolute_url)

    return urls


def extract_pagination_urls(
    html: bytes,
    page_url: str,
    start_url: str,
) -> list[str]:
    """
    Tìm URL phân trang theo các dạng thường gặp của NukeViet:
      /page-2/
      /page-20/
      ?page=2
      ?p=2

    Trang hiện tại có thể không phân trang; khi đó trả về [].
    """
    soup = BeautifulSoup(html, "lxml")
    start_path = urlparse(start_url).path.rstrip("/") + "/"

    urls: list[str] = []
    seen: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        absolute_url = normalize_url(
            urljoin(page_url, anchor["href"])
        )
        parsed = urlparse(absolute_url)

        if not same_host(absolute_url):
            continue

        if "/vi/laws/detail/" in parsed.path:
            continue

        if not parsed.path.startswith(start_path):
            continue

        if not PAGINATION_RE.search(absolute_url):
            continue

        if absolute_url not in seen:
            seen.add(absolute_url)
            urls.append(absolute_url)

    return urls


def collect_detail_urls(
    session: requests.Session,
    start_url: str,
    max_pages: int,
    max_docs: int,
    timeout: tuple[float, float],
    robots_parser: urllib.robotparser.RobotFileParser | None,
    page_delay: float,
    jitter: float,
) -> list[str]:
    """
    BFS các trang danh sách để không phụ thuộc số trang được mã hóa
    dưới dạng page number hay offset.
    """
    queue = [normalize_url(start_url)]
    visited_pages: set[str] = set()

    detail_urls: list[str] = []
    seen_details: set[str] = set()

    while queue:
        if max_pages > 0 and len(visited_pages) >= max_pages:
            break

        page_url = queue.pop(0)

        if page_url in visited_pages:
            continue

        ensure_allowed(robots_parser, page_url)

        response = fetch(
            session=session,
            url=page_url,
            timeout=timeout,
        )
        visited_pages.add(page_url)

        page_details = extract_detail_urls(
            response.content,
            page_url,
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
            f"[Danh sách {len(visited_pages)}] "
            f"{len(page_details)} link, mới {new_count}, "
            f"tổng {len(detail_urls)}."
        )

        if max_docs > 0 and len(detail_urls) >= max_docs:
            break

        for pagination_url in extract_pagination_urls(
            response.content,
            page_url,
            start_url,
        ):
            if (
                pagination_url not in visited_pages
                and pagination_url not in queue
            ):
                queue.append(pagination_url)

        polite_sleep(page_delay, jitter)

    return detail_urls


def document_id_from_url(url: str) -> str:
    match = DETAIL_PATH_RE.match(
        urlparse(url).path
    )

    if match:
        return f"education_vqa_{match.group(1)}"

    digest = hashlib.sha256(
        url.encode("utf-8")
    ).hexdigest()[:24]

    return f"education_vqa_{digest}"


def is_download_url(url: str) -> bool:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)

    if query.get("download") == ["1"]:
        return True

    path_lower = parsed.path.lower()

    return path_lower.endswith(
        (".pdf", ".docx", ".doc")
    )


def extract_attachment_urls(
    html: bytes,
    detail_url: str,
) -> list[str]:
    """
    Trang VQA thường dùng:
        <detail-url>?download=1&id=0

    Một văn bản có thể có nhiều tệp với id=0, id=1...
    """
    soup = BeautifulSoup(html, "lxml")

    urls: list[str] = []
    seen: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        absolute_url = normalize_url(
            urljoin(detail_url, anchor["href"])
        )
        link_text = clean_inline_text(
            anchor.get_text(" ", strip=True)
        ).lower()

        if not (
            is_download_url(absolute_url)
            or "tải tập tin" in link_text
            or "tai tap tin" in link_text
        ):
            continue

        if not same_host(absolute_url):
            # Vẫn cho phép file ngoài host nếu website dẫn trực tiếp.
            pass

        if absolute_url not in seen:
            seen.add(absolute_url)
            urls.append(absolute_url)

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

                compact_cells: list[str] = []

                for cell_text in cells:
                    if not cell_text:
                        continue

                    # python-docx có thể lặp nội dung của merged cell.
                    if (
                        not compact_cells
                        or compact_cells[-1] != cell_text
                    ):
                        compact_cells.append(cell_text)

                if compact_cells:
                    parts.append(" | ".join(compact_cells))

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
    response: requests.Response,
    attachment_url: str,
) -> str:
    content_type = response.headers.get(
        "Content-Type",
        "",
    ).lower()

    content_disposition = response.headers.get(
        "Content-Disposition",
        "",
    ).lower()

    combined = (
        f"{attachment_url} "
        f"{content_type} "
        f"{content_disposition}"
    ).lower()

    stripped = content.lstrip()

    if stripped.startswith(PDF_MAGIC):
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


def download_bytes(
    response: requests.Response,
    max_file_mb: float,
) -> bytes:
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

    return b"".join(chunks)


def extract_attachment_text(
    session: requests.Session,
    attachment_url: str,
    detail_url: str,
    timeout: tuple[float, float],
    max_file_mb: float,
) -> tuple[str, str]:
    response = fetch(
        session=session,
        url=attachment_url,
        timeout=timeout,
        referer=detail_url,
        stream=True,
    )
    content = download_bytes(
        response,
        max_file_mb=max_file_mb,
    )

    file_type = detect_file_type(
        content,
        response,
        attachment_url,
    )

    if file_type == "pdf":
        return extract_pdf_text(content), "pdf"

    if file_type == "docx":
        return extract_docx_text(content), "docx"

    if file_type == "doc":
        raise ValueError(
            "File là định dạng .doc cũ. "
            "python-docx không đọc được định dạng này."
        )

    if file_type == "html":
        preview = clean_inline_text(
            content[:300].decode(
                response.encoding or "utf-8",
                errors="replace",
            )
        )
        raise ValueError(
            "Link tải trả về HTML thay vì tài liệu. "
            f"Preview: {preview[:150]}"
        )

    raise ValueError(
        "Không xác định được loại file tải về."
    )


def extract_detail_html_text(html: bytes) -> str:
    """
    Chỉ dùng khi người dùng chủ động bật --allow-html-fallback.
    Nội dung trang chi tiết thường là metadata, không phải toàn văn.
    """
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(
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
        ]
    ):
        tag.decompose()

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.select_one(".laws-detail")
        or soup.select_one(".detail")
        or soup.body
    )

    if main is None:
        return ""

    return clean_text(
        main.get_text("\n", strip=True)
    )


def deduplicate_texts(texts: list[str]) -> list[str]:
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


def crawl_one_document(
    session: requests.Session,
    detail_url: str,
    timeout: tuple[float, float],
    min_chars: int,
    max_files_per_doc: int,
    max_file_mb: float,
    allow_html_fallback: bool,
) -> tuple[str, str]:
    detail_response = fetch(
        session=session,
        url=detail_url,
        timeout=timeout,
    )
    detail_html = detail_response.content

    attachment_urls = extract_attachment_urls(
        detail_html,
        detail_url,
    )

    if max_files_per_doc > 0:
        attachment_urls = attachment_urls[
            :max_files_per_doc
        ]

    texts: list[str] = []
    source_types: list[str] = []
    errors: list[str] = []

    for index, attachment_url in enumerate(
        attachment_urls,
        start=1,
    ):
        try:
            text, source_type = extract_attachment_text(
                session=session,
                attachment_url=attachment_url,
                detail_url=detail_url,
                timeout=timeout,
                max_file_mb=max_file_mb,
            )

            if len(text) < min_chars:
                raise ValueError(
                    f"Text chỉ có {len(text)} ký tự."
                )

            texts.append(text)
            source_types.append(source_type)

        except Exception as error:
            errors.append(
                f"file {index}: {error}"
            )

    unique_texts = deduplicate_texts(texts)

    if unique_texts:
        return (
            clean_text("\n\n".join(unique_texts)),
            "+".join(dict.fromkeys(source_types)),
        )

    if allow_html_fallback:
        html_text = extract_detail_html_text(
            detail_html
        )

        if len(html_text) >= min_chars:
            return html_text, "html"

    if not attachment_urls:
        errors.append(
            "Không tìm thấy link file đính kèm."
        )

    if not errors:
        errors.append(
            "Không có nội dung đạt yêu cầu."
        )

    raise RuntimeError(
        " | ".join(errors)
    )


def load_existing_ids(path: Path) -> set[str]:
    existing: set[str] = set()

    if not path.exists():
        return existing

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
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


def append_jsonl(
    path: Path,
    record: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


def polite_sleep(
    delay: float,
    jitter: float,
) -> None:
    seconds = max(
        0.0,
        delay + random.uniform(0.0, jitter),
    )
    time.sleep(seconds)


def crawl(args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    error_path = Path(args.error_output)

    timeout = (
        float(args.connect_timeout),
        float(args.read_timeout),
    )

    print(
        "TLS certificate store: "
        + (
            "system trust store"
            if TRUSTSTORE_ENABLED
            else "Requests/certifi"
        )
    )

    if args.insecure:
        print(
            "CẢNH BÁO: Đã tắt xác minh SSL "
            "theo yêu cầu --insecure."
        )

    session = build_session(
        user_agent=args.user_agent,
        retries=args.retries,
        backoff=args.backoff,
        verify_ssl=not args.insecure,
    )

    robots_parser = check_robots(
        session=session,
        start_url=args.start_url,
        policy=args.robots_policy,
        timeout=timeout,
    )

    existing_ids = load_existing_ids(
        output_path
    )
    print(
        f"Đã có {len(existing_ids)} "
        f"bản ghi trong file output."
    )

    try:
        detail_urls = collect_detail_urls(
            session=session,
            start_url=args.start_url,
            max_pages=args.max_pages,
            max_docs=args.max_docs,
            timeout=timeout,
            robots_parser=robots_parser,
            page_delay=args.page_delay,
            jitter=args.jitter,
        )
    except Exception:
        session.close()
        raise

    print(
        f"Sẽ xử lý {len(detail_urls)} "
        f"URL chi tiết."
    )

    saved = 0
    skipped = 0
    failed = 0

    for index, detail_url in enumerate(
        detail_urls,
        start=1,
    ):
        document_id = document_id_from_url(
            detail_url
        )

        if document_id in existing_ids:
            skipped += 1
            print(
                f"[{index}/{len(detail_urls)}] "
                f"SKIP {document_id}"
            )
            continue

        try:
            ensure_allowed(
                robots_parser,
                detail_url,
            )

            text, source_type = crawl_one_document(
                session=session,
                detail_url=detail_url,
                timeout=timeout,
                min_chars=args.min_chars,
                max_files_per_doc=args.max_files_per_doc,
                max_file_mb=args.max_file_mb,
                allow_html_fallback=args.allow_html_fallback,
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

        polite_sleep(
            args.delay,
            args.jitter,
        )

    session.close()

    print("\nHoàn tất")
    print(f"- Lưu mới: {saved}")
    print(f"- Bỏ qua: {skipped}")
    print(f"- Lỗi: {failed}")
    print(f"- Output: {output_path}")
    print(f"- Error log: {error_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start-url",
        default=DEFAULT_START_URL,
        help=(
            "Trang danh sách cần crawl. "
            "Mặc định là toàn bộ /vi/laws/."
        ),
    )
    parser.add_argument(
        "--output",
        default="data/raw/education_vqa.jsonl",
    )
    parser.add_argument(
        "--error-output",
        default="data/raw/education_vqa_errors.jsonl",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
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
            "Số văn bản tối đa; "
            "0 là không giới hạn."
        ),
    )
    parser.add_argument(
        "--max-files-per-doc",
        type=int,
        default=5,
        help=(
            "Số file đính kèm tối đa "
            "trong mỗi văn bản; 0 là không giới hạn."
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
        help="Khoảng nghỉ giữa hai văn bản.",
    )
    parser.add_argument(
        "--page-delay",
        type=float,
        default=2.0,
        help="Khoảng nghỉ giữa các trang danh sách.",
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
        "--allow-html-fallback",
        action="store_true",
        help=(
            "Cho phép lưu text HTML khi không đọc "
            "được file đính kèm. Mặc định tắt vì "
            "trang chi tiết thường chỉ chứa metadata."
        ),
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help=(
            "Tắt xác minh SSL cho toàn bộ request. "
            "Chỉ dùng khi thực sự gặp lỗi certificate."
        ),
    )
    parser.add_argument(
        "--user-agent",
        default=(
            "VietnameseEducationDatasetCrawler/0.1 "
            "(research; contact: replace-with-your-email)"
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    crawl(parse_args())