import json
import os

from .configs.vnexpress import CATEGORIES
from .vnexpress import get_article_links, crawl_article

OUTPUT = "output_tintuc.json"

if os.path.exists(OUTPUT):
    with open(OUTPUT, encoding="utf8") as f:
        articles = json.load(f)
else:
    articles = []

article_id = len(articles) + 1

for domain, base_url in CATEGORIES.items():

    print("=" * 40)
    print(domain)

    links = []

    for page in range(1, 11):

        if page == 1:
            url = base_url
        else:
            url = f"{base_url}-p{page}"

        print(url)

        links.extend(get_article_links(url))

    links = list(dict.fromkeys(links))

    print(len(links))

    crawled = {x["url"] for x in articles}

    for link in links:

        if link in crawled:
            continue

        try:

            article = crawl_article(link, article_id)
            article["domain"] = domain

            articles.append(article)

            article_id += 1

            with open(OUTPUT, "w", encoding="utf8") as f:
                json.dump(
                    articles,
                    f,
                    ensure_ascii=False,
                    indent=4
                )

            print(article["id"], article["url"])

        except Exception as e:
            print(link, e)

print("Done")