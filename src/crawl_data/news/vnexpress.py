import random
import time

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/138.0.0.0 Safari/537.36",
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
    "Referer": "https://vnexpress.net/",
    "Connection": "keep-alive"
}

session = requests.Session()
session.headers.update(HEADERS)

retry = Retry(
    total=5,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
)

adapter = HTTPAdapter(max_retries=retry)

session.mount("http://", adapter)
session.mount("https://", adapter)


def get_html(url):

    while True:
        try:
            response = session.get(url, timeout=20)

            if response.status_code == 429:
                print("429 -> ngủ 60s")
                time.sleep(60)
                continue

            response.raise_for_status()

            # giả lập người đọc
            time.sleep(random.uniform(1.5, 3.5))

            return response.text

        except Exception as e:
            print(e)
            time.sleep(10)


def get_article_links(category_url):

    html = get_html(category_url)

    soup = BeautifulSoup(html, "html.parser")

    links = []

    for article in soup.find_all("article"):
        a = article.select_one("h3.title-news a[href]")

        if a:
            links.append(a["href"])

    return list(dict.fromkeys(links))


def crawl_article(url, article_id):

    html = get_html(url)

    soup = BeautifulSoup(html, "html.parser")

    title = soup.select_one("h1.title-detail")
    description = soup.select_one("p.description")

    body = "\n".join(
        p.get_text(" ", strip=True)
        for p in soup.select("p.Normal")
    )

    title = title.get_text(strip=True) if title else ""
    description = description.get_text(" ", strip=True) if description else ""

    return {
        "id": article_id,
        "url": url,
        "content": "\n\n".join(
            x for x in [title, description, body] if x
        )
    }