import re
from bs4 import BeautifulSoup
from .request import get_html, get_response


def crawl_issue(start_url):
    """
    Crawl TAT CA bai viet qua cac trang cua 1 section (khong random).
    Duyet theo tung khoi 'div.obj_article_summary' de lay dung 1 title-link
    va 1 pdf-link cho moi bai, tranh bug thieu ten do link an/trung truoc day.
    """
    url = start_url
    papers = []
    seen_ids = set()

    while url:
        print(f"Crawling: {url}")
        html = get_html(url)
        soup = BeautifulSoup(html, "html.parser")

        for article in soup.select("div.obj_article_summary"):
            title_tag = article.select_one("a[id^='article-']")
            if title_tag is None:
                continue

            article_url = title_tag["href"]
            match = re.search(r"/article/view/(\d+)", article_url)
            if not match:
                continue

            paper_id = int(match.group(1))
            if paper_id in seen_ids:
                continue
            seen_ids.add(paper_id)

            pdf_tag = article.select_one("a.obj_galley_link.pdf")
            pdf_url = pdf_tag["href"] if pdf_tag else None

            papers.append({
                "paper_id": paper_id,
                "title": title_tag.get_text(" ", strip=True),
                "article_url": article_url,
                "pdf_url": pdf_url
            })

        next_btn = soup.select_one("a.next")
        url = next_btn["href"] if next_btn else None

    return papers


def crawl_all_sections(section_urls):
    if isinstance(section_urls, str):
        section_urls = [section_urls]

    all_papers = []
    seen_ids = set()

    for section_url in section_urls:
        section_papers = crawl_issue(section_url)
        for p in section_papers:
            if p["paper_id"] in seen_ids:
                continue
            seen_ids.add(p["paper_id"])
            all_papers.append(p)

    return all_papers