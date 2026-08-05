import requests
from bs4 import BeautifulSoup
import json
import os
import random
import time
from playwright.sync_api import sync_playwright

from .configs.thanhnien import CATEGORIES, LIMIT_PER_CATEGORY

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://thanhnien.vn/",
}

# Nghỉ giữa mỗi bài (giây) - random trong khoảng này
DELAY_MIN = 0.3
DELAY_MAX = 1.0

# Cứ crawl xong SMALL_BATCH_SIZE bài (tính chung toàn bộ, không phải theo từng category)
# thì nghỉ ngắn
SMALL_BATCH_SIZE = 40
SMALL_BATCH_PAUSE_MIN = 1
SMALL_BATCH_PAUSE_MAX = 5

# Cứ crawl xong BATCH_SIZE bài (tính chung toàn bộ, không phải theo từng category)
# thì nghỉ dài hơn để tránh bị chặn
BATCH_SIZE = 150
BATCH_PAUSE_MIN = 5
BATCH_PAUSE_MAX = 10

OUTPUT_DIR = "output"
FINAL_FILE = os.path.join(OUTPUT_DIR, "thanhnien.json")

# Dùng chung một session cho toàn bộ chương trình
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
    Trang thanhnien.vn dùng nút 'Xem thêm' (class list__viewmore / view-more)
    thay vì tự cuộn là ra bài mới. Hàm này thử click nút đó, nếu không tồn tại
    hoặc bị ẩn thì trả về False.
    """
    selector = "a.list__viewmore, a.view-more"
    try:
        btn = page.query_selector(selector)
        if btn is None:
            return False
        if not btn.is_visible():
            return False
        btn.scroll_into_view_if_needed()
        btn.click()
        return True
    except Exception as e:
        print(f"Không click được nút 'Xem thêm': {e}")
        return False


# ---------------------- Step 1: collect article links for one category ----------------------

def get_article_links(category_url, limit):
    links = []
    seen = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent="Mozilla/5.0")

        safe_goto(page, category_url)

        try:
            page.wait_for_selector("div.box-category-item", timeout=15000)
        except Exception:
            print("Cảnh báo: không thấy div.box-category-item sau 15s, vẫn tiếp tục thử parse.")

        last_count = 0
        no_change = 0

        while len(links) < limit:
            soup = BeautifulSoup(page.content(), "html.parser")

            for item in soup.select("div.box-category-item"):
                a = item.select_one("a.box-category-link-with-avatar") \
                    or item.select_one("a.box-category-link-title")
                if not a or not a.get("href"):
                    continue

                href = a["href"]
                if href.startswith("/"):
                    href = "https://thanhnien.vn" + href

                if href not in seen:
                    seen.add(href)
                    links.append(href)

                for rel_a in item.select("a.box-category-related-link-title"):
                    rel_href = rel_a.get("href")
                    if not rel_href:
                        continue
                    if rel_href.startswith("/"):
                        rel_href = "https://thanhnien.vn" + rel_href
                    if rel_href not in seen:
                        seen.add(rel_href)
                        links.append(rel_href)

            print(f"  Collected: {len(links)}/{limit}")

            if len(links) >= limit:
                break

            clicked = click_view_more_if_present(page)
            if not clicked:
                page.evaluate("""
                    window.scrollTo({
                        top: document.body.scrollHeight,
                        behavior: 'smooth'
                    });
                """)

            page.wait_for_timeout(2000)

            if len(links) == last_count:
                no_change += 1
            else:
                no_change = 0

            last_count = len(links)

            if no_change >= 5:
                print("  Không còn bài mới.")
                break

        browser.close()

    return links[:limit]


# ---------------------- Step 2: crawl one article ----------------------

def crawl_article(url, category, article_id=None):
    html = get_html(url)
    soup = BeautifulSoup(html, "html.parser")

    content = []

    # LƯU Ý: nếu content rỗng, mở DevTools trên 1 bài viết thật của thanhnien.vn
    # để lấy đúng class cho title/sapo/nội dung rồi sửa lại 3 dòng bên dưới.

    title = soup.select_one("h1.detail-title") or soup.select_one("h1")
    if title:
        content.append(title.get_text(" ", strip=True))

    sapo = soup.select_one("h2.detail-sapo") or soup.select_one(".sapo")
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

    # Bước 1: lấy link (cache riêng theo category)
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

    # Bước 2: resume theo category
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

        # Ghi lại file tổng hợp sau mỗi category để không mất dữ liệu nếu bị dừng
        with open(FINAL_FILE, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)

    with open(FINAL_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    print(f"\nHoàn tất: {len(all_results)} bài đã lưu vào {FINAL_FILE}")


if __name__ == "__main__":
    main()