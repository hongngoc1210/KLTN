import os

from .config import SECTION_URL

from ..utils.issue import crawl_all_sections
from ..utils.article import crawl_article
from ..utils.pdf import extract_text_from_pdf

from ..utils.save import save_json


papers = crawl_all_sections(SECTION_URL)

print(f"Found {len(papers)} papers")

save_json(
    papers,
    "output/metadata/articles.json"
)

os.makedirs("output/articles", exist_ok=True)

success = 0
failed = 0
total = len(papers)

for i, paper in enumerate(papers, 1):

    print(f"[{i}/{total}] Processing {paper['paper_id']}")

    output_path = f"output/articles/{paper['paper_id']}.json"

    # Bỏ qua nếu đã crawl
    if os.path.exists(output_path):
        print("  -> Skip (đã tồn tại)")
        continue

    try:
        metadata = crawl_article(
            paper["article_url"]
        )

        if metadata["pdf_url"] is None:
            metadata["pdf_url"] = paper["pdf_url"]

        if metadata["pdf_url"]:
            try:
                content = extract_text_from_pdf(
                    metadata["pdf_url"]
                )

                if content:
                    content = content.strip()

                metadata["content"] = content if content else None

            except Exception as e:
                print(f"  -> Lỗi trích PDF: {e}")
                metadata["content"] = None
        else:
            metadata["content"] = None

        save_json(
            metadata,
            output_path
        )

        success += 1

    except Exception as e:
        print(f"  -> Lỗi: {e}")
        failed += 1

print("\nFinished.")
print(f"Success : {success}")
print(f"Failed  : {failed}")
print(f"Skipped : {total - success - failed}")