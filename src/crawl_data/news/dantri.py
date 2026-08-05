import requests
from bs4 import BeautifulSoup
import json
import os
import random
import time

from .configs.dantri import CATEGORIES, LIMIT_PER_CATEGORY

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://dantri.com.vn/",
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

OUTPUT_DIR = "output_dantri"
FINAL_FILE = os.path.join(OUTPUT_DIR, "dantri.json")

# Dùng chung một session cho toàn bộ chương trình
session = requests.Session()
session.headers.update(HEADERS)


def _safe_name(cat_name):
    """Làm sạch tên category để dùng an toàn trong tên file."""
    return cat_name.replace("+", "p")


def links_file(cat_name):
    return os.path.join(OUTPUT_DIR, f"links_{_safe_name(cat_name)}.json")


def partial_file(cat_name):
    return os.path.join(OUTPUT_DIR, f"partial_{_safe_name(cat_name)}.json")


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


# ---------------------- Step 1: collect article links for one category ----------------------

def _page_url(category_url, page_num):
    """
    Trang 1: chính là category_url (vd https://dantri.com.vn/the-gioi.htm)
    Trang N>1: https://dantri.com.vn/the-gioi/trang-N.htm
    """
    if page_num <= 1:
        return category_url

    base = category_url[:-4] if category_url.endswith(".htm") else category_url
    return f"{base}/trang-{page_num}.htm"


def get_article_links(category_url, limit):
    links = []
    seen = set()
    page_num = 1

    while len(links) < limit:
        url = _page_url(category_url, page_num)

        try:
            html = get_html(url)
        except Exception as e:
            print(f"  Lỗi lấy trang {page_num}: {e}")
            break

        soup = BeautifulSoup(html, "html.parser")
        articles = soup.select("article.article-item")

        if not articles:
            print(f"  Hết bài ở trang {page_num}.")
            break

        found_new = False
        for article in articles:
            a = article.select_one("h3.article-title a")
            if not a or not a.get("href"):
                continue

            href = a["href"]
            if href.startswith("/"):
                href = "https://dantri.com.vn" + href

            if href not in seen:
                seen.add(href)
                links.append(href)
                found_new = True

        print(f"  Trang {page_num}: tổng {len(links)}/{limit}")

        if not found_new:
            print("  Không còn bài mới.")
            break

        if len(links) >= limit:
            break

        page_num += 1
        time.sleep(random.uniform(0.5, 1.5))

    return links[:limit]


# ---------------------- Step 2: crawl one article ----------------------

def crawl_article(url, category, article_id=None):
    html = get_html(url)
    soup = BeautifulSoup(html, "html.parser")

    content = []

    title = soup.select_one('h1[data-slot="title"]')
    if title:
        content.append(title.get_text(" ", strip=True))

    sapo = soup.select_one('h2[data-slot="sapo"]')
    if sapo:
        content.append(sapo.get_text(" ", strip=True))

    body = soup.select_one('div[data-slot="content"]')
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