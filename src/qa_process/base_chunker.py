import json
import os

from underthesea import sent_tokenize


class BaseChunker:

    def __init__(
        self,
        min_paragraph_words=10,
        min_chunk_words=30,
        max_chunk_words=128,
    ):
        self.min_paragraph_words = min_paragraph_words
        self.min_chunk_words = min_chunk_words
        self.max_chunk_words = max_chunk_words

    def word_count(self, text):
        return len(text.split())

    def split_paragraphs(self, content):
        paragraphs = []

        for p in content.split("\n"):

            p = p.strip()

            if not p:
                continue

            if (
                self.word_count(p) < self.min_chunk_words
                and paragraphs
            ):
                paragraphs[-1] += " " + p
            else:
                paragraphs.append(p)

        return paragraphs

    def preprocess_paragraphs(self, paragraphs):
        """
        Override ở class con.
        """
        return paragraphs

    def split_long_paragraph(self, paragraph):

        if self.word_count(paragraph) <= self.max_chunk_words:
            return [paragraph]

        chunks = []
        current = ""

        for sent in sent_tokenize(paragraph):

            sent = sent.strip()

            if not sent:
                continue

            if not current:
                current = sent

            elif (
                self.word_count(current)
                + self.word_count(sent)
                <= self.max_chunk_words
            ):
                current += " " + sent

            else:

                if self.word_count(current) >= self.min_chunk_words:
                    chunks.append(current)

                current = sent

        if (
            current
            and self.word_count(current) >= self.min_chunk_words
        ):
            chunks.append(current)

        return chunks

    def chunk_article(self, article):

        paragraphs = self.split_paragraphs(
            article["content"]
        )

        paragraphs = self.preprocess_paragraphs(
            paragraphs
        )

        article_chunks = []

        for p in paragraphs:
            article_chunks.extend(
                self.split_long_paragraph(p)
            )

        return article_chunks

    def chunk_file(self, input_path, output_path):

        with open(
            input_path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        chunks = []

        for article in data:

            article_chunks = self.chunk_article(article)

            # Tổng số chunk của document
            n_chunks = len(article_chunks)

            for idx, chunk in enumerate(article_chunks):

                chunks.append(
                    {
                        "chunk_id": (
                            f"{article['id']}_chunk{idx}"
                        ),

                        "doc_id": article["id"],

                        "domain": article["domain"],

                        "chunk_index": idx,

                        "n_chunks": n_chunks,

                        "word_count": self.word_count(chunk),

                        "content": chunk,
                    }
                )

        output_dir = os.path.dirname(output_path)

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
            f"Saved {len(chunks)} chunks -> {output_path}"
        )