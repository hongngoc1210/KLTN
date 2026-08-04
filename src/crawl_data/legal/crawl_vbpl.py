from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from playwright.sync_api import (
    BrowserContext,
    Locator,
    Page,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)

BASE_URL = "https://vbpl.vn"
START_URL = "https://vbpl.vn/van-ban/trung-uong"
DETAIL_FRAGMENT = "/van-ban/chi-tiet/"
DOMAIN = "legal"

UUID_RE = re.compile(
    r"([0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12})"
)

LEGAL_TITLE_RE = re.compile(
    r"^(Hiến pháp|Bộ luật|Luật|Pháp lệnh|Lệnh|"
    r"Nghị quyết|Nghị định|Quyết định|Thông tư)",
    re.IGNORECASE,
)


def normalize_space(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def make_document_id(url: str) -> str:
    matches = UUID_RE.findall(url)
    if matches:
        return f"legal_{matches[-1].lower()}"

    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    return f"legal_{digest}"


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_existing_ids(path: Path) -> set[str]:
    result: set[str] = set()

    if not path.exists():
        return result

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                record = json.loads(line)
                value = record.get("id")
                if isinstance(value, str):
                    result.add(value)
            except (json.JSONDecodeError, AttributeError):
                continue

    return result


def wait_page_ready(page: Page, timeout_ms: int) -> None:
    page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)

    try:
        page.wait_for_load_state(
            "networkidle",
            timeout=min(timeout_ms, 15_000),
        )
    except PlaywrightTimeoutError:
        pass

    # Website render danh sách bằng JavaScript.
    page.wait_for_timeout(2500)


def save_debug(page: Page, debug_dir: Path, name: str) -> None:
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


def extract_urls_from_dom(page: Page) -> list[str]:
    """
    Tìm URL chi tiết trong href và các thuộc tính thường dùng bởi SPA.

    Selector cũ chỉ xét:
        a[href*='/van-ban/chi-tiet/']

    Trang hiện tại có thể điều hướng bằng click handler nên cần xét thêm
    data-url, routerLink, onclick và HTML đã render.
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
                    if (value && value.includes("/van-ban/chi-tiet/")) {
                        values.push(value);
                    }
                }
            }

            const html = document.documentElement.innerHTML;
            const matches = html.match(
                /(?:https?:\\/\\/vbpl\\.vn)?\\/van-ban\\/chi-tiet\\/[^"'<>\\s]+/g
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
            r"(https?://vbpl\.vn)?/van-ban/chi-tiet/[^\"'()<>\s]+",
            raw,
        )
        if not match:
            continue

        url = urljoin(BASE_URL, match.group(0))
        url = url.rstrip(".,;)]}")

        if DETAIL_FRAGMENT in url and url not in seen:
            seen.add(url)
            urls.append(url)

    return urls


def collect_document_titles(page: Page) -> list[str]:
    """
    Tìm các tiêu đề văn bản đang hiển thị.

    Chỉ lấy phần tử sâu nhất có nội dung bắt đầu bằng loại văn bản,
    tránh lấy cả card cha chứa thêm trạng thái và ngày tháng.
    """
    titles = page.evaluate(
        """
        () => {
            const prefix = /^(Hiến pháp|Bộ luật|Luật|Pháp lệnh|Lệnh|Nghị quyết|Nghị định|Quyết định|Thông tư)/i;
            const normalize = value =>
                (value || "").replace(/\\s+/g, " ").trim();

            const isVisible = element => {
                const style = getComputedStyle(element);
                const rect = element.getBoundingClientRect();

                return (
                    style.display !== "none" &&
                    style.visibility !== "hidden" &&
                    rect.width > 5 &&
                    rect.height > 5
                );
            };

            const result = [];

            for (const element of document.querySelectorAll(
                "a,button,h1,h2,h3,h4,h5,p,span,div"
            )) {
                if (!isVisible(element)) {
                    continue;
                }

                const text = normalize(element.innerText);

                if (
                    text.length < 35 ||
                    text.length > 1500 ||
                    !prefix.test(text)
                ) {
                    continue;
                }

                const childHasSameKind = [...element.children].some(child => {
                    const childText = normalize(child.innerText);
                    return (
                        childText.length >= 35 &&
                        prefix.test(childText)
                    );
                });

                if (childHasSameKind) {
                    continue;
                }

                result.push(text);
            }

            return [...new Set(result)];
        }
        """
    )

    cleaned: list[str] = []

    for title in titles:
        if not isinstance(title, str):
            continue

        title = normalize_one_line(title)

        if LEGAL_TITLE_RE.match(title) and title not in cleaned:
            cleaned.append(title)

    return cleaned


def mark_best_title_element(page: Page, title: str) -> bool:
    """
    Gắn data attribute vào phần tử phù hợp nhất với tiêu đề.

    Nếu phần tử tiêu đề không tự có click handler thì sự kiện click vẫn
    bubble lên card cha. Trường hợp cần thiết, chọn ancestor có cursor:pointer,
    role=link/button hoặc onclick.
    """
    return bool(
        page.evaluate(
            """
            title => {
                const normalize = value =>
                    (value || "").replace(/\\s+/g, " ").trim();

                document.querySelectorAll(
                    "[data-vbpl-crawl-target='1']"
                ).forEach(element => {
                    element.removeAttribute("data-vbpl-crawl-target");
                });

                const visible = element => {
                    const style = getComputedStyle(element);
                    const rect = element.getBoundingClientRect();
                    return (
                        style.display !== "none" &&
                        style.visibility !== "hidden" &&
                        rect.width > 5 &&
                        rect.height > 5
                    );
                };

                let candidates = [...document.querySelectorAll(
                    "a,button,h1,h2,h3,h4,h5,p,span,div"
                )].filter(element => {
                    return (
                        visible(element) &&
                        normalize(element.innerText) === title
                    );
                });

                if (!candidates.length) {
                    const prefix = title.slice(0, Math.min(120, title.length));

                    candidates = [...document.querySelectorAll(
                        "a,button,h1,h2,h3,h4,h5,p,span,div"
                    )].filter(element => {
                        const value = normalize(element.innerText);
                        return (
                            visible(element) &&
                            value.startsWith(prefix) &&
                            value.length <= title.length + 120
                        );
                    });
                }

                if (!candidates.length) {
                    return false;
                }

                candidates.sort((a, b) => {
                    const aChildren = a.querySelectorAll("*").length;
                    const bChildren = b.querySelectorAll("*").length;
                    return aChildren - bChildren;
                });

                let target = candidates[0];
                let node = target;

                for (let depth = 0; depth < 6 && node; depth += 1) {
                    const style = getComputedStyle(node);
                    const role = node.getAttribute("role");

                    if (
                        node.tagName === "A" ||
                        node.tagName === "BUTTON" ||
                        node.hasAttribute("onclick") ||
                        role === "link" ||
                        role === "button" ||
                        style.cursor === "pointer"
                    ) {
                        target = node;
                        break;
                    }

                    node = node.parentElement;
                }

                target.setAttribute("data-vbpl-crawl-target", "1");
                return true;
            }
            """,
            title,
        )
    )


def restore_listing_page(
    page: Page,
    page_number: int,
    timeout_ms: int,
) -> bool:
    """
    Mở lại trang danh sách và chuyển đến đúng page_number.

    Cách này chậm hơn nhưng không phụ thuộc việc browser có giữ state
    phân trang khi quay lại từ trang chi tiết hay không.
    """
    page.goto(
        START_URL,
        wait_until="domcontentloaded",
        timeout=timeout_ms,
    )
    wait_page_ready(page, timeout_ms)

    for current in range(1, page_number):
        if not click_next_page(
            page=page,
            current_page_number=current,
            timeout_ms=timeout_ms,
        ):
            return False

    return True


def click_title_get_url(
    page: Page,
    title: str,
    page_number: int,
    timeout_ms: int,
) -> str | None:
    if not mark_best_title_element(page, title):
        return None

    target = page.locator("[data-vbpl-crawl-target='1']").first

    old_pages = list(page.context.pages)
    old_url = page.url

    try:
        target.scroll_into_view_if_needed(timeout=5000)
        target.click(timeout=10_000)
    except Exception:
        try:
            target.click(timeout=5000, force=True)
        except Exception:
            return None

    # Cùng tab: route SPA hoặc navigation thông thường.
    try:
        page.wait_for_url(
            re.compile(r".*/van-ban/chi-tiet/.*"),
            timeout=12_000,
        )
    except PlaywrightTimeoutError:
        pass

    if DETAIL_FRAGMENT in page.url:
        detail_url = page.url

        try:
            page.go_back(
                wait_until="domcontentloaded",
                timeout=timeout_ms,
            )
            wait_page_ready(page, timeout_ms)
        except Exception:
            restore_listing_page(page, page_number, timeout_ms)

        # Một số SPA quay lại trang 1; bảo đảm đúng trang đang crawl.
        if title not in normalize_one_line(
            page.locator("body").inner_text()
        ):
            restore_listing_page(page, page_number, timeout_ms)

        return detail_url

    # Trường hợp website mở tab mới.
    new_pages = [
        candidate
        for candidate in page.context.pages
        if candidate not in old_pages
    ]

    for new_page in new_pages:
        try:
            wait_page_ready(new_page, timeout_ms)

            if DETAIL_FRAGMENT in new_page.url:
                detail_url = new_page.url
                new_page.close()
                return detail_url
        except Exception:
            try:
                new_page.close()
            except Exception:
                pass

    # Không điều hướng: cố gắng trở lại trạng thái ban đầu.
    if page.url != old_url:
        restore_listing_page(page, page_number, timeout_ms)

    return None


def page_signature(page: Page) -> list[str]:
    return collect_document_titles(page)[:5]


def click_next_page(
    page: Page,
    current_page_number: int,
    timeout_ms: int,
) -> bool:
    """
    Click pagination bằng JavaScript vì URL không đổi.

    Ưu tiên nút số trang kế tiếp. Nếu không thấy, tìm nút có aria-label,
    title hoặc class biểu thị next.
    """
    before = page_signature(page)
    next_number = str(current_page_number + 1)

    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    page.wait_for_timeout(1000)

    clicked = page.evaluate(
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

            const clickableAncestor = element => {
                let node = element;

                for (let depth = 0; depth < 5 && node; depth += 1) {
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

            let numeric = [...document.querySelectorAll(
                "button,a,[role='button'],li,span"
            )].filter(element => {
                return (
                    visible(element) &&
                    normalize(element.innerText) === nextNumber
                );
            });

            numeric.sort((a, b) => {
                return (
                    b.getBoundingClientRect().top -
                    a.getBoundingClientRect().top
                );
            });

            if (numeric.length) {
                clickableAncestor(numeric[0]).click();
                return true;
            }

            const selectors = [
                "[aria-label*='next' i]",
                "[title*='next' i]",
                "[aria-label*='sau' i]",
                "[title*='sau' i]",
                ".ant-pagination-next",
                ".p-paginator-next",
                ".pagination-next",
                ".next"
            ];

            for (const selector of selectors) {
                const candidate = [...document.querySelectorAll(selector)]
                    .find(visible);

                if (candidate) {
                    clickableAncestor(candidate).click();
                    return true;
                }
            }

            return false;
        }
        """,
        next_number,
    )

    if not clicked:
        return False

    deadline = time.time() + timeout_ms / 1000

    while time.time() < deadline:
        page.wait_for_timeout(500)
        after = page_signature(page)

        if after and after != before:
            return True

    return False


def collect_urls_on_page(
    page: Page,
    page_number: int,
    timeout_ms: int,
    debug_dir: Path,
) -> list[str]:
    # Cách nhanh nhất: URL đã tồn tại trong DOM.
    direct_urls = extract_urls_from_dom(page)

    if direct_urls:
        print(
            f"Trang {page_number}: tìm trực tiếp "
            f"{len(direct_urls)} URL trong DOM."
        )
        return direct_urls

    # Fallback dành cho SPA điều hướng bằng click handler.
    titles = collect_document_titles(page)

    if not titles:
        save_debug(
            page,
            debug_dir,
            f"no_titles_page_{page_number}",
        )
        print(
            f"Trang {page_number}: không tìm thấy tiêu đề. "
            f"Đã lưu debug vào {debug_dir}."
        )
        return []

    print(
        f"Trang {page_number}: không có href, "
        f"sẽ click {len(titles)} tiêu đề để lấy URL."
    )

    urls: list[str] = []
    seen: set[str] = set()

    for index, title in enumerate(titles, start=1):
        # Sau mỗi lần quay lại, đảm bảo tiêu đề vẫn tồn tại.
        body_text = normalize_one_line(page.locator("body").inner_text())

        if title not in body_text:
            if not restore_listing_page(
                page,
                page_number,
                timeout_ms,
            ):
                break

        url = click_title_get_url(
            page=page,
            title=title,
            page_number=page_number,
            timeout_ms=timeout_ms,
        )

        if url and url not in seen:
            seen.add(url)
            urls.append(url)
            print(f"  [{index}/{len(titles)}] URL OK")
        else:
            print(
                f"  [{index}/{len(titles)}] Không lấy được URL: "
                f"{title[:90]}"
            )

    return urls


def collect_all_detail_urls(
    page: Page,
    max_pages: int,
    timeout_ms: int,
    debug_dir: Path,
) -> list[str]:
    if not restore_listing_page(page, 1, timeout_ms):
        raise RuntimeError("Không mở được trang danh sách VBPL.")

    all_urls: list[str] = []
    seen: set[str] = set()
    page_number = 1

    while True:
        urls = collect_urls_on_page(
            page=page,
            page_number=page_number,
            timeout_ms=timeout_ms,
            debug_dir=debug_dir,
        )

        for url in urls:
            if url not in seen:
                seen.add(url)
                all_urls.append(url)

        print(
            f"Trang {page_number}: lấy {len(urls)} URL, "
            f"tổng cộng {len(all_urls)}."
        )

        if max_pages > 0 and page_number >= max_pages:
            break

        # Phục hồi đúng trang hiện tại trước khi bấm next.
        if not restore_listing_page(
            page,
            page_number,
            timeout_ms,
        ):
            break

        if not click_next_page(
            page,
            current_page_number=page_number,
            timeout_ms=timeout_ms,
        ):
            save_debug(
                page,
                debug_dir,
                f"cannot_next_after_page_{page_number}",
            )
            print("Không bấm được trang tiếp theo.")
            break

        page_number += 1

    return all_urls


def extract_detail_text(page: Page) -> str:
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
                ".vanban-chitiet"
            ];

            let root = null;

            for (const selector of selectors) {
                const candidate = document.querySelector(selector);

                if (
                    candidate &&
                    candidate.innerText &&
                    candidate.innerText.trim().length > 300
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

    return normalize_space(text)


def crawl_one_detail(
    context: BrowserContext,
    url: str,
    timeout_ms: int,
) -> str:
    page = context.new_page()

    try:
        page.goto(
            url,
            wait_until="domcontentloaded",
            timeout=timeout_ms,
        )
        wait_page_ready(page, timeout_ms)

        text = extract_detail_text(page)

        if len(text) < 300:
            page.wait_for_timeout(3000)
            text = extract_detail_text(page)

        return text
    finally:
        page.close()


def crawl(args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    error_path = Path(args.error_output)
    debug_dir = Path(args.debug_dir)

    existing_ids = load_existing_ids(output_path)
    print(f"Đã có {len(existing_ids)} bản ghi trong file output.")

    timeout_ms = int(args.timeout * 1000)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=not args.headed,
            channel="chromium",
        )

        context = browser.new_context(
            locale="vi-VN",
            viewport={"width": 1440, "height": 1000},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )

        listing_page = context.new_page()

        try:
            detail_urls = collect_all_detail_urls(
                page=listing_page,
                max_pages=args.max_pages,
                timeout_ms=timeout_ms,
                debug_dir=debug_dir,
            )
        finally:
            listing_page.close()

        if args.max_docs > 0:
            detail_urls = detail_urls[: args.max_docs]

        print(f"Sẽ crawl nội dung của {len(detail_urls)} văn bản.")

        saved = 0
        skipped = 0
        failed = 0

        for index, url in enumerate(detail_urls, start=1):
            document_id = make_document_id(url)

            if document_id in existing_ids:
                skipped += 1
                print(
                    f"[{index}/{len(detail_urls)}] "
                    f"SKIP {document_id}"
                )
                continue

            try:
                text = crawl_one_detail(
                    context=context,
                    url=url,
                    timeout_ms=timeout_ms,
                )

                if len(text) < args.min_chars:
                    raise ValueError(
                        f"Nội dung quá ngắn: {len(text)} ký tự"
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
                    f"OK {document_id} | {len(text)} ký tự"
                )

            except Exception as error:
                failed += 1

                append_jsonl(
                    error_path,
                    {
                        "id": document_id,
                        "url": url,
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
                    args.delay + random.uniform(0, args.jitter),
                )
            )

        context.close()
        browser.close()

    print("\nHoàn tất")
    print(f"- Lưu mới: {saved}")
    print(f"- Bỏ qua: {skipped}")
    print(f"- Lỗi: {failed}")
    print(f"- Output: {output_path}")
    print(f"- Debug: {debug_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        default="data/raw/legal.jsonl",
    )
    parser.add_argument(
        "--error-output",
        default="data/raw/legal_errors.jsonl",
    )
    parser.add_argument(
        "--debug-dir",
        default="data/debug/vbpl",
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
        default=5,
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
    )
    parser.add_argument(
        "--headed",
        action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":
    crawl(parse_args())