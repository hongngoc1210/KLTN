from bs4 import BeautifulSoup

from .request import get_html, get_response


def crawl_article(url):

    html = get_html(url)

    soup = BeautifulSoup(html, "html.parser")

    data = {

        "article_url": url,

        "title": None,

        "authors": [],

        "keywords": [],

        "abstract": None,

        "doi": None,

        "publish_date": None,

        "pdf_url": None

    }

    # TODO:
    # lấy title
    # authors
    # abstract
    # keywords
    # doi
    # publish_date
    # pdf_url

    return data