from bs4 import BeautifulSoup
import pymupdf4llm
import io
import fitz

from .request import get_html, get_response


def _resolve_real_download_url(galley_view_url):
    """
    Lấy link download PDF thật.
    """
    html = get_html(galley_view_url)
    soup = BeautifulSoup(html, "html.parser")

    download_tag = soup.select_one("a.download")

    if download_tag is None or not download_tag.get("href"):
        raise ValueError(
            f"Không tìm thấy link download PDF: {galley_view_url}"
        )

    return download_tag["href"]


# def extract_text_from_pdf(galley_view_url):
#     download_url = _resolve_real_download_url(galley_view_url)

#     response = get_response(download_url)

#     content_type = response.headers.get("Content-Type", "")
#     if "pdf" not in content_type.lower():
#         raise ValueError(
#             f"Không phải PDF ({content_type})"
#         )

#     pdf_bytes = response.content

#     markdown = pymupdf4llm.to_markdown(
#         pdf_bytes,
#         write_images=False,
#         image_path=None,
#         page_chunks=False
#     )

#     return markdown.strip()

def extract_text_from_pdf(galley_view_url):
    download_url = _resolve_real_download_url(galley_view_url)

    response = get_response(download_url)

    if response.status_code != 200:
        raise RuntimeError(f"Download PDF failed: {response.status_code}")

    if "pdf" not in response.headers.get("Content-Type", "").lower():
        raise RuntimeError("URL không trả về PDF.")

    pdf_bytes = response.content

    # ---------------------------------------------------
    # Kiểm tra PDF có text hay không
    # ---------------------------------------------------
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    total_chars = 0

    for page in doc:
        total_chars += len(page.get_text())

    doc.close()

    # ---------------------------------------------------
    # PDF có text -> dùng pymupdf4llm
    # ---------------------------------------------------
    if total_chars > 300:

        md = pymupdf4llm.to_markdown(
            io.BytesIO(pdf_bytes),
            write_images=False,
            page_chunks=False,
            ignore_graphics=True,
            show_progress=False,
        )

        return md.strip()

    # ---------------------------------------------------
    # PDF scan
    # ---------------------------------------------------
    raise RuntimeError(
        "PDF gần như không có text (scan PDF). "
        "Nên dùng MinerU hoặc Nougat OCR."
    )
