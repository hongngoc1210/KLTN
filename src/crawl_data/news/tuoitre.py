import requests
from bs4 import BeautifulSoup
import json
import os
import random
import time
from playwright.sync_api import sync_playwright

from .configs.tuoitre import CATEGORIES, LIMIT_PER_CATEGORY

DOMAIN = "https://tuoitre.vn"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": DOMAIN + "/",
}

# Nghỉ giữa mỗi bài (giây) - random trong khoảng này
DELAY_MIN = 0.3
DELAY_MAX = 1.0

SMALL_BATCH_SIZE = 40
SMALL_BATCH_PAUSE_MIN = 1
SMALL_BATCH_PAUSE_MAX = 5

BATCH_SIZE = 150
BATCH_PAUSE_MIN = 5
BATCH_PAUSE_MAX = 10

OUTPUT_DIR = "output_tuoitre"
FINAL_FILE = os.path.join(OUTPUT_DIR, "tuoitre.json")

session = requests.Session()
session.headers.update(HEADERS)


def links_file(cat_name):
    return os.path.join(OUTPUT_DIR, f"links_{cat_name}.json")


def partial_file(cat_name):
    return os.path.join(OUTPUT_DIR, f"partial_{cat_name}.json")


# ---------------------- HTTP helper ----------------------

def get_html(url, max_retries=5):
    for attempt in range(max_retries):
        try:
            r = session.get(url, timeout=30)
            if r.status_code in (428, 429):
                wait = (2 ** attempt) + random.uniform(1, 3)
                print(f"Bị chặn (status {r.status_code}), chờ {wait:.1f}s rồi thử lại...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.text
        except requests.exceptions.RequestException as e:
            wait = (2 ** attempt) + random.uniform(1, 3)
            print(f"Lỗi request: {e}, chờ {wait:.1f}s rồi thử lại...")
            time.sleep(wait)

    raise Exception(f"Không lấy được {url} sau {max_retries} lần thử")


# ---------------------- Playwright helper ----------------------

def safe_goto(page, url, retries=3):
    for attempt in range(retries):
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            return
        except Exception as e:
            print(f"Goto lỗi (lần {attempt + 1}): {e}")
            time.sleep(5 * (attempt + 1))
    raise Exception(f"Không load được trang sau {retries} lần thử: {url}")


def click_view_more_if_present(page):
    """
    Tuổi Trẻ:
    - Ban đầu auto load khi scroll.
    - Hết auto load mới hiện nút 'Xem thêm'.
    """

    while True:
        # số bài hiện tại
        old_count = page.locator("article.box-category-item").count()

        # scroll xuống cuối
        page.mouse.wheel(0, 6000)
        page.wait_for_timeout(2500)

        # nếu auto load thêm bài thì chưa cần click
        new_count = page.locator("article.box-category-item").count()

        if new_count > old_count:
            print(f"Auto load: {old_count} -> {new_count}")
            return True

        # kiểm tra nút xem thêm
        btn = page.locator("a.view-more")

        if btn.count() > 0 and btn.last.is_visible():
            print("Click Xem thêm...")

            btn.last.click(force=True)

            try:
                page.wait_for_function(
                    f"document.querySelectorAll('article.box-category-item').length > {new_count}",
                    timeout=10000,
                )
            except:
                pass

            page.wait_for_timeout(3000)
            return True

        # không auto load nữa và cũng không có nút
        print("Không còn bài mới.")
        return False

# ---------------------- Step 1: collect article links for one category ----------------------

def get_article_links(category_url, limit):
    links = []
    seen = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=HEADERS["User-Agent"])

        safe_goto(page, category_url)

        try:
            page.wait_for_selector("article.box-category-item", timeout=15000)
        except Exception:
            print("Không tìm thấy danh sách bài viết.")
            browser.close()
            return []

        while len(links) < limit:

            soup = BeautifulSoup(page.content(), "html.parser")

            for article in soup.select("article.box-category-item"):
                a = (
                    article.select_one("a.box-category-link-title")
                    or article.select_one("a.box-category-link-with-avatar")
                )

                if not a or not a.get("href"):
                    continue

                href = a["href"]
                if href.startswith("/"):
                    href = DOMAIN + href

                if href not in seen:
                    seen.add(href)
                    links.append(href)

                # các bài liên quan
                for rel_a in article.select("a.box-category-related-link-title"):
                    rel_href = rel_a.get("href")
                    if not rel_href:
                        continue

                    if rel_href.startswith("/"):
                        rel_href = DOMAIN + rel_href

                    if rel_href not in seen:
                        seen.add(rel_href)
                        links.append(rel_href)

            print(f"Collected: {len(links)}/{limit}")

            if len(links) >= limit:
                break

            # thử scroll trước
            old_count = page.locator("article.box-category-item").count()

            page.mouse.wheel(0, 6000)
            page.wait_for_timeout(2500)

            new_count = page.locator("article.box-category-item").count()

            if new_count > old_count:
                print(f"Auto load: {old_count} -> {new_count}")
                continue

            # hết auto load thì thử click Xem thêm
            btn = page.locator("a.view-more")

            if btn.count() > 0 and btn.last.is_visible():
                print("Click Xem thêm...")

                btn.last.click(force=True)

                try:
                    page.wait_for_function(
                        f"document.querySelectorAll('article.box-category-item').length > {new_count}",
                        timeout=10000,
                    )
                except:
                    pass

                page.wait_for_timeout(2500)
                continue

            print("Không còn bài mới.")
            break

        browser.close()

    return links[:limit]


# ---------------------- Step 2: crawl one article ----------------------

def crawl_article(url, category, article_id=None):
    html = get_html(url)
    soup = BeautifulSoup(html, "html.parser")

    content = []

    title = soup.select_one("h1.detail-title") or soup.select_one("h1")
    if title:
        content.append(title.get_text(" ", strip=True))

    sapo = soup.select_one("p.detail-sapo") \
        or soup.select_one("[data-role='sapo']") \
        or soup.select_one(".sapo")
    if sapo:
        content.append(sapo.get_text(" ", strip=True))

    body = soup.select_one("div.detail-content") or soup.select_one("div[data-role='content']")
    if body:
        for p in body.find_all("p"):
            text = p.get_text(" ", strip=True)
            if text:
                content.append(text)

    return {
        "id": article_id,
        "url": url,
        "content": "\n\n".join(content),
        "domain": category,
    }


# ---------------------- Per-category pipeline ----------------------

def crawl_category(cat_name, category_url, limit, global_counters):
    print(f"\n=== Category: {cat_name} ({category_url}) ===")

    lf = links_file(cat_name)
    if os.path.exists(lf):
        with open(lf, "r", encoding="utf-8") as f:
            links = json.load(f)
        print(f"Đã có sẵn {len(links)} link từ {lf}, bỏ qua bước lấy link.")
    else:
        links = get_article_links(category_url, limit)
        with open(lf, "w", encoding="utf-8") as f:
            json.dump(links, f, ensure_ascii=False, indent=4)
        print(f"Đã lưu {len(links)} link vào {lf}")

    results = []
    done_urls = set()

    pf = partial_file(cat_name)
    if os.path.exists(pf):
        with open(pf, "r", encoding="utf-8") as f:
            try:
                prev = json.load(f)
                for item in prev:
                    if item["url"] not in done_urls:
                        results.append(item)
                        done_urls.add(item["url"])
            except json.JSONDecodeError:
                pass

    if done_urls:
        print(f"Resume: đã crawl trước đó {len(done_urls)} bài của category '{cat_name}'.")

    remaining = [link for link in links if link not in done_urls]

    for link in remaining:
        try:
            global_counters["done"] += 1
            article_id = global_counters["done"]
            item = crawl_article(link, cat_name, article_id=article_id)
            results.append(item)
            print(f"[{cat_name}] {len(results)}/{len(links)} (tổng: {global_counters['done']})")
        except Exception as e:
            global_counters["done"] -= 1
            print(f"Lỗi crawl {link}: {e}")

        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

        if len(results) % 20 == 0:
            with open(pf, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

        if global_counters["done"] % BATCH_SIZE == 0:
            pause = random.uniform(BATCH_PAUSE_MIN, BATCH_PAUSE_MAX)
            print(f"Đã crawl tổng {global_counters['done']} bài, nghỉ {pause:.0f}s...")
            time.sleep(pause)
        elif global_counters["done"] % SMALL_BATCH_SIZE == 0:
            pause = random.uniform(SMALL_BATCH_PAUSE_MIN, SMALL_BATCH_PAUSE_MAX)
            print(f"Đã crawl tổng {global_counters['done']} bài, nghỉ {pause:.0f}s...")
            time.sleep(pause)

    with open(pf, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    return results


# ---------------------- Main ----------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []
    global_counters = {"done": 0}

    for cat_name, category_url in CATEGORIES.items():
        cat_results = crawl_category(cat_name, category_url, LIMIT_PER_CATEGORY, global_counters)
        all_results.extend(cat_results)

        with open(FINAL_FILE, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)

    with open(FINAL_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    print(f"\nHoàn tất: {len(all_results)} bài đã lưu vào {FINAL_FILE}")


if __name__ == "__main__":
    main()