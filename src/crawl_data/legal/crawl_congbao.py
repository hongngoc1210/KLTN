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
from urllib.parse import unquote, urljoin, urlparse

import fitz  # PyMuPDF

# Dùng certificate store của hệ điều hành. Trên Windows, cách này thường
# xử lý được các CDN mà browser truy cập được nhưng certifi của Requests
# không xây dựng được đầy đủ certificate chain.
try:
    import truststore

    truststore.inject_into_ssl()
    TRUSTSTORE_ENABLED = True
except ImportError:
    TRUSTSTORE_ENABLED = False

import requests
from bs4 import BeautifulSoup
from docx import Document
from docx.document import Document as _Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_URL = "https://congbao.chinhphu.vn"
LIST_URL = f"{BASE_URL}/van-ban-dang-cong-bao.htm"
ROBOTS_URL = f"{BASE_URL}/robots.txt"
CDN_HOST = "g7.cdnchinhphu.vn"

DOMAIN = "legal"


class CDNSSLError(RuntimeError):
    """SSL verification failed specifically for the Government CDN."""


# Ví dụ:
# /van-ban/van-ban-hop-nhat-so-66-vbhn-tt-bct-470025.htm
DETAIL_PATH_RE = re.compile(
    r"^/van-ban/(?!van-ban-dang-cong-bao(?:/|\.htm))"
    r"[^?#]+-(\d+)\.htm$",
    re.IGNORECASE,
)


def build_session(
    user_agent: str,
    retries: int,
    backoff: float,
    allow_insecure_cdn: bool,
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
                "image/avif,image/webp,*/*;q=0.8"
            ),
            "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.7",
            "Connection": "keep-alive",
        }
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    # Thuộc tính nội bộ chỉ dùng trong script này.
    session._allow_insecure_cdn = allow_insecure_cdn  # type: ignore[attr-defined]

    return session


def check_robots(
    session: requests.Session,
    user_agent: str,
    policy: str,
    timeout: tuple[float, float],
) -> urllib.robotparser.RobotFileParser | None:
    """
    policy:
      - strict: không tải được robots.txt thì dừng.
      - warn: cảnh báo rồi tiếp tục.
      - off: không kiểm tra.
    """
    if policy == "off":
        return None

    parser = urllib.robotparser.RobotFileParser()
    parser.set_url(ROBOTS_URL)

    try:
        response = session.get(ROBOTS_URL, timeout=timeout)
        response.raise_for_status()
        parser.parse(response.text.splitlines())
    except requests.RequestException as error:
        message = f"Không tải được robots.txt: {error}"

        if policy == "strict":
            raise RuntimeError(message) from error

        print(f"CẢNH BÁO: {message}")
        print("Tiếp tục vì --robots-policy=warn.")
        return None

    if not parser.can_fetch(user_agent, LIST_URL):
        raise RuntimeError(
            f"robots.txt không cho phép user-agent crawl {LIST_URL}"
        )

    return parser


def ensure_allowed(
    parser: urllib.robotparser.RobotFileParser | None,
    user_agent: str,
    url: str,
) -> None:
    if parser is not None and not parser.can_fetch(user_agent, url):
        raise RuntimeError(f"robots.txt không cho phép crawl: {url}")


def list_page_url(page_number: int) -> str:
    if page_number < 1:
        raise ValueError("page_number phải >= 1")

    if page_number == 1:
        return LIST_URL

    return (
        f"{BASE_URL}/van-ban-dang-cong-bao/"
        f"trang-{page_number}.htm"
    )


def get_response(
    session: requests.Session,
    url: str,
    timeout: tuple[float, float],
    referer: str | None = None,
    stream: bool = False,
) -> requests.Response:
    headers = {"Referer": referer} if referer else None
    hostname = (urlparse(url).hostname or "").lower()

    allow_insecure = bool(
        getattr(session, "_allow_insecure_cdn", False)
    )

    # Chỉ cho phép verify=False với đúng CDN này và chỉ khi người dùng
    # chủ động truyền --allow-insecure-cdn.
    verify = not (
        allow_insecure and hostname == CDN_HOST
    )

    try:
        response = session.get(
            url,
            headers=headers,
            timeout=timeout,
            stream=stream,
            allow_redirects=True,
            verify=verify,
        )
        response.raise_for_status()
        return response

    except requests.exceptions.SSLError as error:
        if hostname == CDN_HOST:
            raise CDNSSLError(
                "Không xác minh được chứng chỉ SSL của "
                f"{CDN_HOST}. Hãy cài truststore/pip-system-certs "
                "hoặc chỉ dùng --allow-insecure-cdn như phương án "
                "cuối cùng."
            ) from error

        raise


def extract_detail_urls(
    html: bytes,
    page_url: str,
) -> list[str]:
    """
    Lấy các URL chi tiết bằng pattern URL, không phụ thuộc CSS class.
    """
    soup = BeautifulSoup(html, "lxml")

    urls: list[str] = []
    seen: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        absolute_url = urljoin(page_url, anchor["href"])
        parsed = urlparse(absolute_url)

        if parsed.netloc.lower() != urlparse(BASE_URL).netloc.lower():
            continue

        normalized_path = parsed.path.rstrip("/")

        if not DETAIL_PATH_RE.match(normalized_path):
            continue

        clean_url = parsed._replace(
            query="",
            fragment="",
        ).geturl()

        if clean_url not in seen:
            seen.add(clean_url)
            urls.append(clean_url)

    return urls


def detail_id_from_url(url: str) -> str:
    parsed = urlparse(url)
    match = DETAIL_PATH_RE.match(parsed.path.rstrip("/"))

    if match:
        return f"legal_congbao_{match.group(1)}"

    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    return f"legal_congbao_{digest}"


def classify_download_link(
    href: str,
    link_text: str,
) -> str | None:
    """
    CDN dùng URL /api/download/stream?... nên extension có thể nằm trong
    query file_name hoặc trong text của thẻ a.
    """
    combined = unquote(f"{href} {link_text}").lower()

    if ".docx" in combined:
        return "docx"

    if re.search(r"\.pdf(?:\b|$|[?&#])", combined):
        return "pdf"

    return None


def extract_download_urls(
    html: bytes,
    detail_url: str,
) -> dict[str, str]:
    soup = BeautifulSoup(html, "lxml")
    candidates: dict[str, str] = {}

    for anchor in soup.find_all("a", href=True):
        href = urljoin(detail_url, anchor["href"])
        link_text = anchor.get_text(" ", strip=True)

        file_type = classify_download_link(href, link_text)

        if file_type and file_type not in candidates:
            candidates[file_type] = href

    return candidates


def iter_docx_blocks(parent: _Document) -> Iterator[Paragraph | Table]:
    """
    Duyệt paragraph và table theo đúng thứ tự xuất hiện trong DOCX.
    """
    parent_element = parent.element.body

    for child in parent_element.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


def extract_docx_text(content: bytes) -> str:
    if not content.startswith(b"PK"):
        raise ValueError(
            "Dữ liệu tải về không có định dạng DOCX/ZIP hợp lệ."
        )

    document = Document(BytesIO(content))
    blocks: list[str] = []

    for block in iter_docx_blocks(document):
        if isinstance(block, Paragraph):
            text = block.text.strip()

            if text:
                blocks.append(text)

        elif isinstance(block, Table):
            for row in block.rows:
                cells = [
                    clean_inline_text(cell.text)
                    for cell in row.cells
                ]

                # Loại các cell lặp do merged cell của python-docx.
                compact_cells: list[str] = []

                for cell_text in cells:
                    if not cell_text:
                        continue

                    if not compact_cells or compact_cells[-1] != cell_text:
                        compact_cells.append(cell_text)

                if compact_cells:
                    blocks.append(" | ".join(compact_cells))

    return clean_text("\n".join(blocks))


def extract_pdf_text(content: bytes) -> str:
    if not content.lstrip().startswith(b"%PDF"):
        raise ValueError("Dữ liệu tải về không phải PDF hợp lệ.")

    document = fitz.open(stream=content, filetype="pdf")
    page_texts: list[str] = []

    try:
        for page in document:
            try:
                text = page.get_text("text", sort=True)
            except TypeError:
                # Tương thích một số phiên bản PyMuPDF cũ.
                text = page.get_text("text")

            text = clean_text(text)

            if text:
                page_texts.append(text)
    finally:
        document.close()

    result = clean_text("\n\n".join(page_texts))

    if len(result) < 100:
        raise ValueError(
            "PDF gần như không có lớp text; có thể là PDF scan. "
            "Crawler này không tự OCR."
        )

    return result


def clean_inline_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    """
    Chỉ làm sạch cơ bản, chưa chia nhỏ thành context.
    """
    text = text.replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = [
        re.sub(r"[ \t]+", " ", line).strip()
        for line in text.split("\n")
    ]

    result_lines: list[str] = []
    previous_blank = False

    for line in lines:
        if not line:
            if result_lines and not previous_blank:
                result_lines.append("")
            previous_blank = True
            continue

        result_lines.append(line)
        previous_blank = False

    return "\n".join(result_lines).strip()


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


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def polite_sleep(delay: float, jitter: float) -> None:
    seconds = max(0.0, delay + random.uniform(0.0, jitter))
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
        + ("system trust store" if TRUSTSTORE_ENABLED else "Requests/certifi")
    )

    if args.allow_insecure_cdn:
        print(
            "CẢNH BÁO: SSL verification bị tắt riêng cho "
            f"{CDN_HOST}."
        )

    session = build_session(
        user_agent=args.user_agent,
        retries=args.retries,
        backoff=args.backoff,
        allow_insecure_cdn=args.allow_insecure_cdn,
    )

    robots_parser = check_robots(
        session=session,
        user_agent=args.user_agent,
        policy=args.robots_policy,
        timeout=timeout,
    )

    existing_ids = load_existing_ids(output_path)
    print(f"Đã có {len(existing_ids)} bản ghi trong file output.")

    seen_detail_urls: set[str] = set()
    saved = 0
    skipped = 0
    failed = 0
    discovered = 0

    stop = False

    for page_number in range(
        args.start_page,
        args.start_page + args.max_pages
        if args.max_pages > 0
        else 10**9,
    ):
        page_url = list_page_url(page_number)
        ensure_allowed(
            robots_parser,
            args.user_agent,
            page_url,
        )

        try:
            response = get_response(
                session=session,
                url=page_url,
                timeout=timeout,
            )
        except requests.RequestException as error:
            append_jsonl(
                error_path,
                {
                    "stage": "list",
                    "url": page_url,
                    "error": repr(error),
                },
            )
            print(f"[Trang {page_number}] ERROR: {error}")
            failed += 1
            break

        detail_urls = extract_detail_urls(
            response.content,
            page_url,
        )

        new_detail_urls = [
            url
            for url in detail_urls
            if url not in seen_detail_urls
        ]

        print(
            f"[Trang {page_number}] "
            f"{len(detail_urls)} URL chi tiết, "
            f"mới {len(new_detail_urls)}."
        )

        if not detail_urls:
            print("Không tìm thấy văn bản; dừng phân trang.")
            break

        if not new_detail_urls:
            print("Trang không có URL mới; dừng để tránh lặp vô hạn.")
            break

        for detail_url in new_detail_urls:
            seen_detail_urls.add(detail_url)
            discovered += 1

            if args.max_docs > 0 and discovered > args.max_docs:
                stop = True
                break

            document_id = detail_id_from_url(detail_url)

            if document_id in existing_ids:
                skipped += 1
                print(f"  SKIP {document_id}")
                continue

            try:
                ensure_allowed(
                    robots_parser,
                    args.user_agent,
                    detail_url,
                )

                detail_response = get_response(
                    session=session,
                    url=detail_url,
                    timeout=timeout,
                    referer=page_url,
                )

                download_urls = extract_download_urls(
                    detail_response.content,
                    detail_url,
                )

                if not download_urls:
                    raise ValueError(
                        "Không tìm thấy link DOCX hoặc PDF "
                        "trên trang chi tiết."
                    )

                text: str | None = None
                used_type: str | None = None
                extraction_errors: list[str] = []

                # Ưu tiên DOCX, sau đó mới fallback PDF.
                for file_type in ("docx", "pdf"):
                    download_url = download_urls.get(file_type)

                    if not download_url:
                        continue

                    try:
                        file_response = get_response(
                            session=session,
                            url=download_url,
                            timeout=timeout,
                            referer=detail_url,
                        )

                        content = file_response.content

                        if file_type == "docx":
                            candidate_text = extract_docx_text(content)
                        else:
                            candidate_text = extract_pdf_text(content)

                        if len(candidate_text) < args.min_chars:
                            raise ValueError(
                                f"Text quá ngắn: "
                                f"{len(candidate_text)} ký tự."
                            )

                        text = candidate_text
                        used_type = file_type
                        break

                    except CDNSSLError:
                        # DOCX và PDF đều nằm trên cùng CDN, nên thử file
                        # còn lại chỉ làm crawler chờ lâu hơn.
                        raise

                    except Exception as file_error:
                        extraction_errors.append(
                            f"{file_type}: {file_error}"
                        )

                if text is None:
                    raise RuntimeError(
                        "Không trích được nội dung. "
                        + " | ".join(extraction_errors)
                    )

                record = {
                    "id": document_id,
                    "domain": DOMAIN,
                    "text": text,
                }

                append_jsonl(output_path, record)
                existing_ids.add(document_id)
                saved += 1

                print(
                    f"  OK {document_id} | "
                    f"{used_type.upper()} | "
                    f"{len(text)} ký tự"
                )

            except CDNSSLError as error:
                failed += 1

                append_jsonl(
                    error_path,
                    {
                        "stage": "download_ssl",
                        "id": document_id,
                        "url": detail_url,
                        "error": repr(error),
                    },
                )

                print(f"  SSL ERROR {document_id}: {error}")
                print(
                    "Dừng crawler ngay để không lặp lỗi SSL "
                    "trên tất cả văn bản."
                )
                stop = True
                break

            except Exception as error:
                failed += 1

                append_jsonl(
                    error_path,
                    {
                        "stage": "document",
                        "id": document_id,
                        "url": detail_url,
                        "error": repr(error),
                    },
                )

                print(f"  ERROR {document_id}: {error}")

            polite_sleep(args.delay, args.jitter)

        if stop:
            break

        polite_sleep(args.page_delay, args.jitter)

    session.close()

    print("\nHoàn tất")
    print(f"- URL đã xét: {discovered}")
    print(f"- Lưu mới: {saved}")
    print(f"- Bỏ qua: {skipped}")
    print(f"- Lỗi: {failed}")
    print(f"- Output: {output_path}")
    print(f"- Error log: {error_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        default="data/raw/legal_congbao.jsonl",
    )
    parser.add_argument(
        "--error-output",
        default="data/raw/legal_congbao_errors.jsonl",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
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
        help="Số URL văn bản tối đa; 0 là không giới hạn.",
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
        help="Khoảng nghỉ cơ bản giữa hai văn bản.",
    )
    parser.add_argument(
        "--page-delay",
        type=float,
        default=3.0,
        help="Khoảng nghỉ cơ bản giữa hai trang danh sách.",
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
        default=2,
        help="Số lần retry; giảm để tránh chờ lâu khi SSL lỗi.",
    )
    parser.add_argument(
        "--backoff",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--allow-insecure-cdn",
        action="store_true",
        help=(
            "Tắt xác minh SSL chỉ cho g7.cdnchinhphu.vn. "
            "Chỉ dùng khi system trust store vẫn không xử lý được."
        ),
    )
    parser.add_argument(
        "--robots-policy",
        choices=("strict", "warn", "off"),
        default="warn",
    )
    parser.add_argument(
        "--user-agent",
        default=(
            "VietnameseLegalDatasetCrawler/0.1 "
            "(research; contact: replace-with-your-email)"
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    crawl(parse_args())