from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urljoin, urlparse, urlunparse

from playwright.sync_api import (
    Browser,
    BrowserContext,
    Error as PlaywrightError,
    Page,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)

VERSION = "2.2.0-rendered-dom"
BASE_URL = "https://ssc.gov.vn"
DOMAIN = "finance"


@dataclass(frozen=True)
class Section:
    key: str
    name: str
    start_url: str
    id_prefix: str


SECTIONS: dict[str, Section] = {
    "knowledge": Section(
        key="knowledge",
        name="Kiến thức chứng khoán",
        start_url=(
            "https://ssc.gov.vn/webcenter/portal/ubck/"
            "pages_r/m/oto/kinthcchngkhon"
        ),
        id_prefix="finance_ssc_knowledge",
    ),
    "legal": Section(
        key="legal",
        name="Hỏi đáp pháp luật chứng khoán",
        start_url=(
            "https://ssc.gov.vn/webcenter/portal/ubck/"
            "pages_otherpages/giipphplut"
        ),
        id_prefix="finance_ssc_legal",
    ),
}

DETAIL_MARKERS = (
    "/pages_r/l/chitit",
    "/pages_r/m/chitit",
    "/chitit?",
)

WHITESPACE_RE = re.compile(r"[ \t\u00a0]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
DATE_RE = re.compile(r"^\s*\d{1,2}[/-]\d{1,2}[/-]\d{4}\s*$")
ARTICLE_SIGNALS = (
    "trả lời:",
    "hỏi/đáp",
    "đề nghị cho biết",
    "xin cho biết",
    "chứng khoán",
    "cổ phiếu",
    "trái phiếu",
)

NOISE_LINE_RE = re.compile(
    r"^(?:"
    r"chi tiết|"
    r"cỡ chữ(?::.*)?|"
    r"a-\s*a\+|"
    r"đọc báo|"
    r"tương phản(?::.*)?|"
    r"giảm\s+tăng|"
    r"facebook|twitter|email|in bài|chia sẻ|"
    r"trang chủ|giới thiệu|liên hệ|"
    r"sơ đồ cổng thông tin|"
    r"bản quyền thuộc.*|"
    r"địa chỉ:.*|điện thoại:.*|fax:.*"
    r")$",
    re.IGNORECASE,
)

STOP_RE = re.compile(
    r"^(?:các tin khác|tin liên quan|bài viết liên quan|xem thêm)\s*$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crawl SSC knowledge and legal Q&A sections."
    )
    parser.add_argument(
        "--section",
        choices=("all", "knowledge", "legal"),
        default="all",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/finance_ssc.jsonl"),
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Maximum list pages per section; 0 means all pages.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="Maximum new records for the whole run; 0 means unlimited.",
    )
    parser.add_argument("--min-chars", type=int, default=180)
    parser.add_argument("--delay", type=float, default=1.2)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument(
        "--detail-wait",
        type=int,
        default=35,
        help="Seconds to wait for JavaScript-rendered article text.",
    )
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug-dir", type=Path, default=None)
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def normalize_line(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip(" \t\r\n|•·")


def canonicalize_url(url: str) -> str:
    absolute = urljoin(BASE_URL, url.strip())
    parsed = urlparse(absolute)
    query = parse_qs(parsed.query)
    ddoc = query.get("dDocName") or query.get("ddocname")
    normalized_query = f"dDocName={ddoc[0]}" if ddoc else parsed.query
    return urlunparse(
        (
            parsed.scheme or "https",
            parsed.netloc.lower(),
            re.sub(r"/{2,}", "/", parsed.path),
            "",
            normalized_query,
            "",
        )
    )


def is_detail_url(url: str) -> bool:
    lowered = url.lower()
    return (
        "ssc.gov.vn" in lowered
        and "ddocname=" in lowered
        and any(marker in lowered for marker in DETAIL_MARKERS)
    )


def extract_ddoc_name(url: str) -> str | None:
    query = parse_qs(urlparse(url).query)
    values = query.get("dDocName") or query.get("ddocname")
    if not values:
        return None
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "", values[0])
    return cleaned or None


def make_record_id(section: Section, url: str) -> str:
    ddoc = extract_ddoc_name(url)
    if ddoc:
        return f"{section.id_prefix}_{ddoc.lower()}"
    digest = hashlib.sha256(canonicalize_url(url).encode("utf-8")).hexdigest()[:24]
    return f"{section.id_prefix}_{digest}"


def load_existing_ids(path: Path) -> set[str]:
    result: set[str] = set()
    if not path.exists():
        return result
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                logging.warning("Invalid JSON ignored at %s:%d", path, line_number)
                continue
            value = item.get("id")
            if isinstance(value, str):
                result.add(value)
    return result


def jitter_sleep(base: float) -> None:
    if base > 0:
        time.sleep(base + random.uniform(0.1, max(0.25, base * 0.35)))


def safe_goto(page: Page, url: str, timeout_ms: int, retries: int = 3) -> None:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            try:
                page.wait_for_load_state("networkidle", timeout=8_000)
            except PlaywrightTimeoutError:
                pass
            page.wait_for_timeout(1_000)
            return
        except (PlaywrightTimeoutError, PlaywrightError) as error:
            last_error = error
            logging.warning(
                "Navigation attempt %d/%d failed: %s", attempt, retries, url
            )
            if attempt < retries:
                page.wait_for_timeout(1_500 * attempt)
    raise RuntimeError(f"Cannot load {url}: {last_error}")


def auto_scroll(page: Page, rounds: int = 5) -> None:
    previous = -1
    for _ in range(rounds):
        try:
            height = int(page.evaluate("document.body.scrollHeight"))
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(700)
            if height == previous:
                break
            previous = height
        except PlaywrightError:
            break
    try:
        page.evaluate("window.scrollTo(0, 0)")
    except PlaywrightError:
        pass


def collect_detail_links(page: Page) -> list[str]:
    auto_scroll(page)
    hrefs = page.locator("a[href]").evaluate_all(
        "els => els.map(a => a.href || a.getAttribute('href') || '')"
    )
    output: list[str] = []
    seen: set[str] = set()
    for href in hrefs:
        if not isinstance(href, str) or not is_detail_url(href):
            continue
        normalized = canonicalize_url(href)
        if normalized not in seen:
            seen.add(normalized)
            output.append(normalized)
    return output


def pagination_signature(page: Page, links: Iterable[str]) -> str:
    value = page.url + "\n" + "\n".join(sorted(links))
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def next_locators(page: Page):
    selectors = (
        "a[rel='next']",
        "button[aria-label*='next' i]",
        "a[aria-label*='next' i]",
        "button[title*='next' i]",
        "a[title*='next' i]",
        "button[aria-label*='tiếp' i]",
        "a[aria-label*='tiếp' i]",
        "button[title*='tiếp' i]",
        "a[title*='tiếp' i]",
        ".p_AFNext",
        "[class*='next' i]",
    )
    for selector in selectors:
        yield page.locator(selector)

    for pattern in (
        re.compile(r"^tiếp(?: theo)?$", re.I),
        re.compile(r"^trang sau$", re.I),
        re.compile(r"^(?:>|›|»|→)$"),
    ):
        yield page.get_by_role("link", name=pattern)
        yield page.get_by_role("button", name=pattern)


def click_next_page(page: Page, current_signature: str, timeout_ms: int) -> bool:
    for locator in next_locators(page):
        try:
            count = min(locator.count(), 10)
        except PlaywrightError:
            continue
        for index in range(count):
            candidate = locator.nth(index)
            try:
                if not candidate.is_visible() or not candidate.is_enabled():
                    continue
                aria_disabled = candidate.get_attribute("aria-disabled")
                classes = candidate.get_attribute("class") or ""
                if aria_disabled == "true" or re.search(
                    r"disabled|p_AFDisabled", classes, re.I
                ):
                    continue
                candidate.scroll_into_view_if_needed(timeout=2_000)
                candidate.click(timeout=5_000)
                try:
                    page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
                except PlaywrightTimeoutError:
                    pass
                page.wait_for_timeout(1_500)
                links = collect_detail_links(page)
                if links and pagination_signature(page, links) != current_signature:
                    return True
            except (PlaywrightError, PlaywrightTimeoutError):
                continue
    return False


def clean_rendered_article(raw_text: str) -> str:
    """Extract the visible article region from an SSC rendered text block."""
    raw_text = raw_text.replace("\r", "\n")
    lines = [normalize_line(line) for line in raw_text.split("\n")]

    # If the page contains a visible detail heading, discard everything before it.
    detail_index: int | None = None
    for index, line in enumerate(lines):
        if line.casefold() in {"chi tiết", "chi tiết tin"}:
            detail_index = index
            break
    if detail_index is not None:
        lines = lines[detail_index + 1 :]

    output: list[str] = []
    previous = ""
    for line in lines:
        if not line:
            if output and output[-1] != "":
                output.append("")
            continue
        if STOP_RE.match(line):
            break
        if DATE_RE.match(line):
            continue
        if NOISE_LINE_RE.match(line):
            continue
        if len(line) <= 2 and not line.isdigit():
            continue
        if line.casefold() == previous.casefold():
            continue
        output.append(line)
        previous = line

    text = "\n".join(output).strip()
    text = MULTI_NEWLINE_RE.sub("\n\n", text)

    # Remove an occasional footer that can remain in the selected container.
    footer_markers = (
        "Bản quyền thuộc",
        "Sơ đồ cổng thông tin",
        "Địa chỉ:",
    )
    cut_positions = [text.find(marker) for marker in footer_markers if marker in text]
    if cut_positions:
        text = text[: min(cut_positions)].strip()
    return text


def score_article(text: str) -> float:
    if not text:
        return -1.0
    lowered = text.casefold()
    score = float(len(text))
    score += 1_500 if "trả lời:" in lowered else 0
    score += 800 if "hỏi/đáp" in lowered else 0
    score += sum(220 for signal in ARTICLE_SIGNALS if signal in lowered)
    score -= 1_000 if "requires a javascript enabled browser" in lowered else 0
    score -= 700 if "bản quyền thuộc" in lowered else 0
    if len(text) > 50_000:
        score *= 0.25
    return score


def frame_text_candidates(page: Page) -> list[tuple[float, str, str]]:
    """Read visible innerText from top page and every child frame."""
    candidates: list[tuple[float, str, str]] = []
    selectors = (
        "body",
        "main",
        "article",
        "[role='main']",
        "[class*='detail' i]",
        "[id*='detail' i]",
        "[class*='content' i]",
        "[id*='content' i]",
        "section",
        "td",
    )

    for frame_index, frame in enumerate(page.frames):
        for selector in selectors:
            try:
                limit = 1 if selector == "body" else 100
                values = frame.locator(selector).evaluate_all(
                    """(els, limit) => els
                        .filter(el => {
                            const s = getComputedStyle(el);
                            const r = el.getBoundingClientRect();
                            return s.display !== 'none' && s.visibility !== 'hidden' &&
                                   (r.width > 0 || r.height > 0);
                        })
                        .slice(0, limit)
                        .map(el => el.innerText || '')""",
                    limit,
                )
            except PlaywrightError:
                continue

            for element_index, value in enumerate(values):
                if not isinstance(value, str):
                    continue
                cleaned = clean_rendered_article(value)
                if len(cleaned) < 40:
                    continue
                score = score_article(cleaned)
                if selector == "body":
                    score *= 0.8
                elif selector in {"main", "article", "[role='main']"}:
                    score += 500
                elif "detail" in selector:
                    score += 400
                source = (
                    f"frame={frame_index}, selector={selector}, "
                    f"element={element_index}"
                )
                candidates.append((score, cleaned, source))
    return candidates


def wait_and_extract_article(
    page: Page,
    min_chars: int,
    wait_seconds: int,
) -> tuple[str, str]:
    """Wait until the *cleaned* rendered article is long enough."""
    deadline = time.monotonic() + max(5, wait_seconds)
    best_text = ""
    best_source = "none"

    while time.monotonic() < deadline:
        candidates = frame_text_candidates(page)
        if candidates:
            _, candidate, source = max(candidates, key=lambda item: item[0])
            if score_article(candidate) > score_article(best_text):
                best_text = candidate
                best_source = source

            lowered = candidate.casefold()
            has_strong_signal = (
                "trả lời:" in lowered
                or "hỏi/đáp" in lowered
                or "các tin khác" in lowered
            )
            if len(candidate) >= min_chars and (
                has_strong_signal or len(candidate) >= max(700, min_chars * 2)
            ):
                return candidate, source
        page.wait_for_timeout(750)

    return best_text, best_source


def save_debug(page: Page, debug_dir: Path, name: str) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:120]

    try:
        page.screenshot(
            path=str(debug_dir / f"{safe_name}.png"),
            full_page=True,
        )
    except Exception:
        pass

    try:
        (debug_dir / f"{safe_name}.html").write_text(
            page.content(), encoding="utf-8"
        )
    except Exception:
        pass

    try:
        chunks: list[str] = []
        for frame_index, frame in enumerate(page.frames):
            try:
                text = frame.locator("body").inner_text(timeout=2_000)
            except PlaywrightError as error:
                text = f"<cannot read frame: {error}>"
            chunks.append(
                f"===== FRAME {frame_index}: {frame.url} =====\n{text}\n"
            )
        (debug_dir / f"{safe_name}.txt").write_text(
            "\n".join(chunks), encoding="utf-8"
        )
    except Exception:
        pass


def crawl_article(
    detail_page: Page,
    section: Section,
    url: str,
    min_chars: int,
    timeout_ms: int,
    detail_wait: int,
    debug_dir: Path | None,
) -> dict[str, str] | None:
    record_id = make_record_id(section, url)
    try:
        safe_goto(detail_page, url, timeout_ms=timeout_ms)
        context, source = wait_and_extract_article(
            detail_page,
            min_chars=min_chars,
            wait_seconds=detail_wait,
        )
        logging.debug(
            "Rendered extraction source: %s | %d chars | %s",
            source,
            len(context),
            url,
        )

        if len(context) < min_chars:
            logging.warning("Too short (%d chars), skipped: %s", len(context), url)
            if debug_dir:
                save_debug(detail_page, debug_dir, f"short_{record_id}")
            return None

        lowered = context.casefold()
        if "requires a javascript enabled browser" in lowered:
            logging.warning("JavaScript placeholder page, skipped: %s", url)
            if debug_dir:
                save_debug(detail_page, debug_dir, f"placeholder_{record_id}")
            return None

        return {"id": record_id, "domain": DOMAIN, "context": context}
    except Exception as error:
        logging.exception("Failed article %s: %s", url, error)
        if debug_dir:
            save_debug(detail_page, debug_dir, f"error_{record_id}")
        return None


def create_context(browser: Browser, timeout_ms: int) -> BrowserContext:
    # Deliberately do not block image/font requests. Some Oracle/ADF pages have
    # fragile client-side initialization and are safer when loaded normally.
    context = browser.new_context(
        locale="vi-VN",
        timezone_id="Asia/Ho_Chi_Minh",
        viewport={"width": 1440, "height": 1100},
        extra_http_headers={
            "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.7",
        },
    )
    context.set_default_timeout(timeout_ms)
    return context


def selected_sections(value: str) -> list[Section]:
    if value == "all":
        return [SECTIONS["knowledge"], SECTIONS["legal"]]
    return [SECTIONS[value]]


def crawl_section(
    list_page: Page,
    detail_page: Page,
    section: Section,
    output_handle,
    existing_ids: set[str],
    args: argparse.Namespace,
    total_new: int,
) -> int:
    logging.info("=== %s ===", section.name)
    safe_goto(list_page, section.start_url, timeout_ms=args.timeout * 1000)

    page_number = 1
    seen_urls: set[str] = set()
    previous_signatures: set[str] = set()

    while True:
        links = collect_detail_links(list_page)
        signature = pagination_signature(list_page, links)
        if signature in previous_signatures:
            logging.info("Repeated list page detected; stop section.")
            break
        previous_signatures.add(signature)

        fresh_links = [url for url in links if url not in seen_urls]
        logging.info(
            "List page %d: %d detail links (%d unseen in section)",
            page_number,
            len(links),
            len(fresh_links),
        )

        if not links:
            logging.warning("No detail link found on list page %d", page_number)
            if args.debug_dir:
                save_debug(
                    list_page,
                    args.debug_dir,
                    f"empty_list_{section.key}_{page_number}",
                )
            break

        for index, url in enumerate(fresh_links, 1):
            seen_urls.add(url)
            record_id = make_record_id(section, url)
            if record_id in existing_ids:
                logging.info(
                    "[%d/%d] Existing, skipped: %s",
                    index,
                    len(fresh_links),
                    record_id,
                )
                continue
            if args.max_items and total_new >= args.max_items:
                return total_new

            logging.info("[%d/%d] Crawling %s", index, len(fresh_links), url)
            record = crawl_article(
                detail_page=detail_page,
                section=section,
                url=url,
                min_chars=args.min_chars,
                timeout_ms=args.timeout * 1000,
                detail_wait=args.detail_wait,
                debug_dir=args.debug_dir,
            )
            if record:
                output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_handle.flush()
                existing_ids.add(record["id"])
                total_new += 1
                logging.info(
                    "Saved %s (%d chars)",
                    record["id"],
                    len(record["context"]),
                )
            jitter_sleep(args.delay)

        if args.max_pages and page_number >= args.max_pages:
            logging.info("Reached --max-pages=%d", args.max_pages)
            break

        logging.info("Moving to next list page...")
        if not click_next_page(
            list_page,
            current_signature=signature,
            timeout_ms=args.timeout * 1000,
        ):
            logging.info("No usable next page; section completed.")
            break

        page_number += 1
        jitter_sleep(args.delay)

    return total_new


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)
    logging.info("SSC crawler version: %s", VERSION)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = load_existing_ids(args.output)
    logging.info("Loaded %d existing IDs from %s", len(existing_ids), args.output)

    total_new = 0
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=not args.headed,
            channel="chromium",
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
            ],
        )
        context = create_context(browser, timeout_ms=args.timeout * 1000)
        list_page = context.new_page()
        detail_page = context.new_page()

        try:
            with args.output.open("a", encoding="utf-8") as output_handle:
                for section in selected_sections(args.section):
                    if args.max_items and total_new >= args.max_items:
                        break
                    total_new = crawl_section(
                        list_page=list_page,
                        detail_page=detail_page,
                        section=section,
                        output_handle=output_handle,
                        existing_ids=existing_ids,
                        args=args,
                        total_new=total_new,
                    )
        finally:
            context.close()
            browser.close()

    logging.info("Finished. New records: %d | Output: %s", total_new, args.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        logging.warning("Stopped by user. Existing JSONL data has been preserved.")
        raise SystemExit(130)