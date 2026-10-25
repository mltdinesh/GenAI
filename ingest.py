"""Ingestion: load policy documents -> chunk -> embed -> store in Chroma.

Run once, and again whenever a document changes:
    python ingest.py
"""
import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

import config


def get_embeddings():
    return OllamaEmbeddings(model=config.EMBED_MODEL, base_url=config.OLLAMA_URL)


def load_and_chunk() -> list[Document]:
    """Split each markdown file on its headings first, then by size.

    Splitting on headings keeps each policy section together, and the heading
    is stored as metadata so answers can cite "file > section".
    """
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "section")])
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )

    chunks = []
    for path in sorted(config.DOCS_DIR.glob("*.md")):
        sections = header_splitter.split_text(path.read_text(encoding="utf-8"))
        for doc in size_splitter.split_documents(sections):
            doc.metadata["source"] = path.name
            doc.metadata.setdefault("section", "General")
            chunks.append(doc)
    return chunks


def build_index(embeddings=None) -> Chroma:
    """Rebuild the vector store from scratch so it never holds stale chunks."""
    shutil.rmtree(config.CHROMA_DIR, ignore_errors=True)
    chunks = load_and_chunk()
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings or get_embeddings(),
        collection_name=config.COLLECTION,
        persist_directory=str(config.CHROMA_DIR),
    )
    print(f"Indexed {len(chunks)} chunks from {config.DOCS_DIR}")
    return store


def load_index(embeddings=None) -> Chroma:
    if not config.CHROMA_DIR.exists():
        raise RuntimeError("No index found. Run `python ingest.py` first.")
    return Chroma(
        collection_name=config.COLLECTION,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )


if __name__ == "__main__":
    build_index()
