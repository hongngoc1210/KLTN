
import json
import os

from science import SciPaperChunker
from base_chunker import BaseChunker


class MedicalNewsChunker(BaseChunker):

    def preprocess_paragraphs(self, paragraphs):

        # Bỏ dòng đầu tiên: title
        if len(paragraphs) <= 1:
            return []

        return paragraphs[1:]


class MedicalChunker:
    """
    Dùng chung 2 chunker:

    - medical_vjst_* -> SciPaperChunker
    - còn lại       -> MedicalNewsChunker
    """

    def __init__(
        self,
        min_words=30,
        max_words=128,
        min_paragraph_words=10,
    ):
        self.science_chunker = SciPaperChunker(
            min_words=min_words,
            max_words=max_words,
        )

        self.news_chunker = MedicalNewsChunker(
            min_paragraph_words=min_paragraph_words,
            min_chunk_words=min_words,
            max_chunk_words=max_words,
        )

    def chunk_article(self, article):
        """
        Chọn chunker dựa trên ID document.
        """

        article_id = article["id"]

        if article_id.startswith("medical_vjst_"):
            return self.science_chunker.chunk_document(
                article["content"]
            )

        else:
            return self.news_chunker.chunk_article(
                article
            )

    def chunk_file(self, input_path, output_path):

        with open(
            input_path,
            "r",
            encoding="utf-8"
        ) as f:
            data = json.load(f)

        chunks = []

        for article in data:

            article_chunks = self.chunk_article(
                article
            )

            for i, ch in enumerate(article_chunks):

                chunks.append(
                    {
                        "chunk_id": f"{article['id']}_chunk{i}",
                        "doc_id": article["id"],
                        "domain": article.get("domain"),
                        "chunk_index": i,
                        "n_chunks": len(article_chunks),
                        "word_count": self.word_count(ch),
                        "content": ch,
                    }
                )

        output_dir = os.path.dirname(
            output_path
        )

        if output_dir:
            os.makedirs(
                output_dir,
                exist_ok=True
            )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                chunks,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(
            f"Saved {len(chunks)} chunks -> "
            f"{output_path}"
        )

    def word_count(self, text):
        return len(text.split())


if __name__ == "__main__":

    chunker = MedicalChunker(
        min_words=30,
        max_words=128,
        min_paragraph_words=10,
    )

    chunker.chunk_file(
        "data/medical_clean.json",
        "data/chunking/medical_chunks.json",
    )
