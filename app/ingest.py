from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.database import get_vector_store


def ingest_pdf(file_path: str) -> int:
    reader = PdfReader(file_path)
    pages = [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": file_path, "page": i},
        )
        for i, page in enumerate(reader.pages)
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(pages)

    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    return len(chunks)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: uv run python -m app.ingest <pdf_file_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    print(f"Ingesting: {pdf_path} ...")
    count = ingest_pdf(pdf_path)
    print(f"Done! {count} chunks saved to pgvector.")
