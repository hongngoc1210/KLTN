from base_chunker import BaseChunker


class NewsChunker(BaseChunker):

    def preprocess_paragraphs(self, paragraphs):

        # title + description + author
        if len(paragraphs) <= 3:
            return []

        return paragraphs[2:-1]