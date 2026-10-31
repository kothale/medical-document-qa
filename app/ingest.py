from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import ALLOWED_SUFFIXES, CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR


def iter_document_files(docs_dir: Path | None = None) -> list[Path]:
    root = docs_dir or DOCS_DIR
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES
    )


def load_file(path: Path) -> list[Document]:
    if path.suffix.lower() == ".pdf":
        loaded = PyPDFLoader(str(path)).load()
    else:
        loaded = TextLoader(str(path), encoding="utf-8").load()

    documents: list[Document] = []
    for doc in loaded:
        documents.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    "source": path.name,
                    "path": str(path),
                    "page": doc.metadata.get("page"),
                },
            )
        )
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(documents)
    per_source: dict[str, int] = {}
    prepared: list[Document] = []
    for chunk in chunks:
        source = str(chunk.metadata.get("source", "unknown"))
        index = per_source.get(source, 0)
        per_source[source] = index + 1
        text = chunk.page_content.strip()
        if not text:
            continue
        prepared.append(
            Document(
                page_content=text,
                metadata={
                    "source": source,
                    "path": chunk.metadata.get("path"),
                    "page": chunk.metadata.get("page"),
                    "chunk_index": index,
                },
            )
        )
    return prepared


def load_and_chunk(paths: list[Path] | None = None) -> list[Document]:
    files = paths if paths is not None else iter_document_files()
    documents: list[Document] = []
    for path in files:
        documents.extend(load_file(path))
    return split_documents(documents)
