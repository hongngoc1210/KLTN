import json
import os
import re

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from underthesea import sent_tokenize


MIN_WORDS = 10
MIN_CHUNK_WORDS = 30
MAX_CHUNK_WORDS = 128


class EducationChunker:

    def __init__(self):

        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        self.semantic_splitter = SemanticChunker(
            embeddings=self.embedding,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
        )

    ############################################################
    # utils
    ############################################################

    @staticmethod
    def word_count(text):
        return len(text.split())

    ############################################################
    # clean
    ############################################################

    def clean_text(self, text):

        text = text.replace("\r", "")

        text = re.sub(r"\n{2,}", "\n", text)

        text = re.sub(r"[ \t]+", " ", text)

        return text.strip()

    ############################################################
    # remove table
    ############################################################

    def remove_tables(self, text):

        lines = text.split("\n")

        new_lines = []

        skip = False

        for line in lines:

            s = line.strip()

            if re.match(r"^STT\b", s):
                skip = True
                continue

            if skip:

                if (
                    s.startswith("PHỤ LỤC")
                    or s.startswith("Điều")
                    or s.startswith("CHƯƠNG")
                    or s.startswith("Chương")
                    or s.startswith("MỤC")
                    or s.startswith("Mục")
                ):
                    skip = False
                    new_lines.append(s)

                continue

            new_lines.append(s)

        return "\n".join(new_lines)

    ############################################################
    # split section
    ############################################################

    def split_sections(self, text):

        patterns = [

            r"^Điều\s+\d+",

            r"^Chương\s+[IVXLC]+",

            r"^CHƯƠNG",

            r"^Mục\s+\d+",

            r"^MỤC",

            r"^PHỤ LỤC",

            r"^Phụ lục",

            r"^QUYẾT ĐỊNH",

            r"^THÔNG BÁO",

            r"^HƯỚNG DẪN",

        ]

        sections = []

        current = []

        for line in text.split("\n"):

            line = line.strip()

            if not line:
                continue

            heading = any(
                re.match(p, line, re.IGNORECASE)
                for p in patterns
            )

            if heading and current:

                sections.append("\n".join(current))
                current = [line]

            else:

                current.append(line)

        if current:
            sections.append("\n".join(current))

        return sections

    ############################################################
    # semantic
    ############################################################

    def semantic_split(self, section):

        docs = [Document(page_content=section)]

        chunks = self.semantic_splitter.split_documents(docs)

        return [doc.page_content for doc in chunks]

    ############################################################
    # merge
    ############################################################

    def merge_small_chunks(self, chunks):

        merged = []

        for chunk in chunks:

            n = self.word_count(chunk)

            if n < MIN_WORDS:
                continue

            if n < MIN_CHUNK_WORDS and merged:

                merged[-1] += "\n" + chunk

            else:

                merged.append(chunk)

        return merged

    ############################################################
    # split long
    ############################################################

    def split_long_chunk(self, chunk):

        if self.word_count(chunk) <= MAX_CHUNK_WORDS:
            return [chunk]

        results = []

        current = ""

        for sent in sent_tokenize(chunk):

            sent = sent.strip()

            if not sent:
                continue

            if not current:

                current = sent

                continue

            if self.word_count(current) + self.word_count(sent) <= MAX_CHUNK_WORDS:

                current += " " + sent

            else:

                if self.word_count(current) >= MIN_CHUNK_WORDS:
                    results.append(current)

                current = sent

        if current:

            if self.word_count(current) >= MIN_CHUNK_WORDS:
                results.append(current)

        return results

    ############################################################
    # process article
    ############################################################

    def process(self, content):

        content = self.clean_text(content)

        content = self.remove_tables(content)

        sections = self.split_sections(content)

        semantic_chunks = []

        for section in sections:

            semantic_chunks.extend(
                self.semantic_split(section)
            )

        semantic_chunks = self.merge_small_chunks(
            semantic_chunks
        )

        final_chunks = []

        for chunk in semantic_chunks:

            final_chunks.extend(
                self.split_long_chunk(chunk)
            )

        return final_chunks


############################################################
# run
############################################################

INPUT = "data/education_clean.json"

OUTPUT = "data/chunking/education_chunks.json"

os.makedirs("data/chunking", exist_ok=True)

with open(INPUT, "r", encoding="utf8") as f:
    data = json.load(f)

chunker = EducationChunker()

outputs = []

for article in data:

    chunks = chunker.process(article["content"])

    for i, chunk in enumerate(chunks):

        outputs.append(
            {
                "id": f"{article['id']}_{i}",
                "parent_id": article["id"],
                "chunk_index": i,
                "domain": article["domain"],
                "content": chunk,
            }
        )

with open(OUTPUT, "w", encoding="utf8") as f:
    json.dump(outputs, f, ensure_ascii=False, indent=2)

print("Total chunks:", len(outputs))
print("Saved:", OUTPUT)