import os

from .configs.config_yte import SECTION_URL

from ..utils.issue import crawl_all_sections
from ..utils.article import crawl_article
from ..utils.pdf import extract_text_from_pdf

from ..utils.save import save_json


papers = crawl_all_sections(SECTION_URL)

print(f"Found {len(papers)} papers")

save_json(
    papers,
    "output_yte/metadata/articles.json"
)

os.makedirs("output_yte/articles", exist_ok=True)

for paper in papers:

    print(f"Processing {paper['paper_id']}")

    metadata = crawl_article(
        paper["article_url"]
    )

    if metadata["pdf_url"] is None:
        metadata["pdf_url"] = paper["pdf_url"]

    if metadata["pdf_url"]:
        try:
            metadata["content"] = extract_text_from_pdf(
                metadata["pdf_url"]
            )
        except Exception as e:
            print(f"  Loi trich text PDF: {e}")
            metadata["content"] = None
    else:
        metadata["content"] = None

    save_json(
        metadata,
        f"output_yte/articles/{paper['paper_id']}.json"
    )

print("Finished.")