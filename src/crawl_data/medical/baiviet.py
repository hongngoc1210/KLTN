import os
import json
import time
import random
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from .configs.config import CATEGORIES, LIMIT_PER_CATEGORY


BASE = "https://suckhoedoisong.vn"

HEADERS = {
    "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0 Safari/537.36",
    "Referer": BASE,
    "Accept-Language": "vi-VN,vi;q=0.9",
}

DELAY_MIN = 0.3
DELAY_MAX = 1.0

SMALL_BATCH_SIZE = 40
SMALL_BATCH_PAUSE_MIN = 1
SMALL_BATCH_PAUSE_MAX = 5

BATCH_SIZE = 150
BATCH_PAUSE_MIN = 5
BATCH_PAUSE_MAX = 10

OUTPUT_FILE = "output_baiviet/suckhoe.json"

session = requests.Session()
session.headers.update(HEADERS)


# --------------------------------------------------
# HTTP
# --------------------------------------------------

def get_soup(url, retries=5):

    for attempt in range(retries):

        try:

            r = session.get(url, timeout=30)

            if r.status_code in (428, 429):

                wait = (2 ** attempt) + random.uniform(1, 3)

                print(f"429 -> sleep {wait:.1f}s")

                time.sleep(wait)

                continue

            r.raise_for_status()

            return BeautifulSoup(r.text, "lxml")

        except Exception as e:

            wait = (2 ** attempt) + random.uniform(1, 3)

            print(e)

            time.sleep(wait)

    raise Exception(url)


# --------------------------------------------------
# Playwright
# --------------------------------------------------

def safe_goto(page, url):

    for i in range(3):

        try:

            page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=60000
            )

            return

        except Exception:

            time.sleep(5 * (i + 1))

    raise Exception(url)


def click_view_more_if_present(page):

    selectors = [
        "a.list__viewmore",
        "a.view-more"
    ]

    for selector in selectors:

        try:

            btn = page.query_selector(selector)

            if btn and btn.is_visible():

                btn.scroll_into_view_if_needed()

                btn.click()

                return True

        except:
            pass

    return False


# --------------------------------------------------
# lấy link bài
# --------------------------------------------------

def get_article_links(category_url, limit):

    links = []
    seen = set()

    with sync_playwright() as p:

        browser = p.chromium.launch(headless=True)

        page = browser.new_page()

        safe_goto(page, category_url)

        page.wait_for_selector(
            "article.box-category-item",
            timeout=15000
        )

        no_change = 0
        last = 0

        while len(links) < limit:

            soup = BeautifulSoup(
                page.content(),
                "lxml"
            )

            for item in soup.select("article.box-category-item"):

                a = item.select_one("a.box-category-link-with-avatar")

                if not a:
                    continue

                href = a.get("href")

                if not href:
                    continue

                href = urljoin(BASE, href)

                if href not in seen:

                    seen.add(href)

                    links.append(href)

            print(
                f"Collected {len(links)}/{limit}"
            )

            if len(links) >= limit:
                break

            if click_view_more_if_present(page):

                page.wait_for_timeout(1500)

            else:

                page.evaluate(
                    "window.scrollTo(0, document.body.scrollHeight)"
                )

                page.wait_for_timeout(2000)

            if len(links) == last:

                no_change += 1

            else:

                no_change = 0

            last = len(links)

            if no_change >= 5:
                break

        browser.close()

    return links[:limit]


# --------------------------------------------------
# crawl bài
# --------------------------------------------------

def crawl_article(url, domain, idx):

    soup = get_soup(url)

    texts = []

    title = soup.select_one("h1.detail-title") or soup.select_one("h1")

    if title:

        texts.append(
            title.get_text(" ", strip=True)
        )

    sapo = (
        soup.select_one("div.detail-sapo")
        or
        soup.select_one("h2.detail-sapo")
        or
        soup.select_one(".sapo")
    )

    if sapo:

        texts.append(
            sapo.get_text(" ", strip=True)
        )

    body = (
        soup.select_one("div.detail-content")
        or
        soup.select_one("div[data-role='content']")
    )

    if body:

        for p in body.find_all(["p", "h2", "li"]):

            text = p.get_text(" ", strip=True)

            if text:

                texts.append(text)

    return {
        "id": idx,
        "url": url,
        "domain": domain,
        "content": "\n\n".join(texts)
    }


# --------------------------------------------------
# main
# --------------------------------------------------

def main():

    os.makedirs("output_baiviet", exist_ok=True)

    outputs = []

    seen = set()

    idx = 1

    for domain, url in CATEGORIES.items():

        print(f"\n===== {domain} =====")

        links = get_article_links(
            url,
            LIMIT_PER_CATEGORY
        )

        print(f"Found {len(links)} articles")

        for link in links:

            if link in seen:
                continue

            seen.add(link)

            try:

                outputs.append(
                    crawl_article(
                        link,
                        domain,
                        idx
                    )
                )

                print(f"[{idx}] {link}")

                idx += 1

            except Exception as e:

                print(e)

            time.sleep(
                random.uniform(
                    DELAY_MIN,
                    DELAY_MAX
                )
            )

            if idx % SMALL_BATCH_SIZE == 0:

                pause = random.uniform(
                    SMALL_BATCH_PAUSE_MIN,
                    SMALL_BATCH_PAUSE_MAX
                )

                print(f"Sleep {pause:.1f}s")

                time.sleep(pause)

            if idx % BATCH_SIZE == 0:

                pause = random.uniform(
                    BATCH_PAUSE_MIN,
                    BATCH_PAUSE_MAX
                )

                print(f"Long sleep {pause:.1f}s")

                time.sleep(pause)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            outputs,
            f,
            ensure_ascii=False,
            indent=4
        )

    print(f"\nDone: {len(outputs)} articles")


if __name__ == "__main__":
    main()