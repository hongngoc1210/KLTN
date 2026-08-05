import os
import json
from urllib.parse import urljoin
import time
import random

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

BASE = "https://suckhoedoisong.vn"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

LIMIT = 5


# --------------------------------------------------
# requests
# --------------------------------------------------
def get_soup(url):
    
    time.sleep(random.uniform(0.1, 0.5))

    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()

    return BeautifulSoup(r.text, "lxml")


# --------------------------------------------------
# Playwright chỉ dùng cho trang danh sách bệnh
# --------------------------------------------------
def crawl_disease_links():

    with sync_playwright() as p:

        browser = p.chromium.launch(headless=True)

        page = browser.new_page()

        page.goto(
            BASE + "/tra-cuu-benh.htm",
            wait_until="networkidle"
        )

        page.wait_for_selector("#lstDsBenh a.item")

        soup = BeautifulSoup(page.content(), "lxml")

        browser.close()

    links = []

    for a in soup.select("#lstDsBenh a.item"):

        href = a.get("href")

        if href:

            links.append(urljoin(BASE, href))

    return links


# --------------------------------------------------
# lấy <=5 bài
# --------------------------------------------------
def crawl_article_links(disease_url, limit=LIMIT):

    soup = get_soup(disease_url)

    links = []
    seen = set()

    for a in soup.select("div.box-category-item h3 a"):

        href = a.get("href")

        if not href:
            continue

        href = urljoin(BASE, href)

        if href in seen:
            continue

        seen.add(href)

        links.append(href)

        if len(links) >= limit:
            break

    return links


# --------------------------------------------------
# crawl bài
# --------------------------------------------------
def crawl_article(url):

    soup = get_soup(url)

    title = ""

    h1 = soup.select_one("h1.detail-title")
    if h1:
        title = h1.get_text(" ", strip=True)

    sapo = ""

    s = soup.select_one("div.detail-sapo")
    if s:
        sapo = s.get_text(" ", strip=True)

    texts = []

    body = soup.select_one("div.detail-content")

    if body:

        for tag in body.find_all(["h2", "p", "li"]):

            txt = tag.get_text(" ", strip=True)

            if txt:
                texts.append(txt)

    return {
        "url": url,
        "domain": "tra_cuu_benh",
        "content": "\n\n".join(
            x for x in [
                title,
                sapo,
                "\n".join(texts)
            ] if x
        )
    }


# --------------------------------------------------
# main
# --------------------------------------------------
def main():

    diseases = crawl_disease_links()

    print("Diseases:", len(diseases))

    outputs = []
    seen_articles = set()

    crawled_count = 0

    for i, disease in enumerate(diseases, 1):

        print(f"[{i}/{len(diseases)}] {disease}")

        try:
            articles = crawl_article_links(
                disease,
                limit=LIMIT
            )

        except Exception as e:
            print(e)
            continue

        for article in articles:

            # chống trùng URL
            if article in seen_articles:
                continue

            seen_articles.add(article)

            try:

                outputs.append(crawl_article(article))

                crawled_count += 1

                print(f"    [{crawled_count}] {article}")

                # cứ 30 bài nghỉ 1-5 giây
                if crawled_count % 30 == 0:
                    sleep_time = random.uniform(1, 5)
                    print(f"---- Sleep {sleep_time:.2f}s ----")
                    time.sleep(sleep_time)

            except Exception as e:
                print(e)

    os.makedirs("output_yte", exist_ok=True)

    with open(
        "output_yte/tra_cuu_benh.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            outputs,
            f,
            ensure_ascii=False,
            indent=2
        )

if __name__ == "__main__":
    main()