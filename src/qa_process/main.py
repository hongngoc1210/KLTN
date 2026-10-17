from finance import FinanceChunker
from news import NewsChunker


def main():

    NewsChunker().chunk_file(
        "data/news_clean.json",
        "data/chunking/news_chunks.json",
    )

    FinanceChunker().chunk_file(
        "data/finance_clean.json",
        "data/chunking/finance_chunks.json",
    )


if __name__ == "__main__":
    main()