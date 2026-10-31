import threading
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import EMBEDDING_MODEL, GEMINI_API_KEY, INDEX_DIR
from app.ingest import load_and_chunk


class IndexNotReady(Exception):
    pass


class MissingAPIKey(Exception):
    pass


def _embeddings() -> GoogleGenerativeAIEmbeddings:
    if not GEMINI_API_KEY:
        raise MissingAPIKey(
            "Set GEMINI_API_KEY in a .env file before ingesting documents or asking questions."
        )
    # Leave task_type unset so documents use RETRIEVAL_DOCUMENT and
    # questions use RETRIEVAL_QUERY.
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GEMINI_API_KEY,
    )


def _index_exists(index_dir: Path | None = None) -> bool:
    folder = index_dir or INDEX_DIR
    return (folder / "index.faiss").exists() and (folder / "index.pkl").exists()


def cosine_from_squared_l2(distance: float) -> float:
    """Convert an IndexFlatL2 score to cosine similarity.

    FAISS L2 search returns squared Euclidean distance. Vectors are normalized
    before they are stored and queried, so cosine = 1 - distance / 2.
    """
    return 1.0 - float(distance) / 2.0


class VectorIndex:
    """Gemini embeddings stored in a local FAISS index."""

    def __init__(self, index_dir: Path | None = None) -> None:
        self.index_dir = index_dir or INDEX_DIR
        self._store: FAISS | None = None
        self._lock = threading.Lock()

    def chunk_count(self) -> int | None:
        store = self._load()
        if store is None:
            return None
        return int(store.index.ntotal)

    def rebuild(self, paths: list[Path] | None = None) -> int:
        chunks = load_and_chunk(paths)
        if not chunks:
            raise FileNotFoundError(
                "No .md, .txt, or .pdf documents were found to index."
            )
        store = FAISS.from_documents(chunks, _embeddings(), normalize_L2=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        store.save_local(str(self.index_dir))
        with self._lock:
            self._store = store
        return len(chunks)

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        store = self._load()
        if store is None:
            raise IndexNotReady(
                "The FAISS index is empty. Ingest documents before asking a question."
            )
        matches = store.similarity_search_with_score(query, k=k)
        return [(doc, cosine_from_squared_l2(distance)) for doc, distance in matches]

    def _load(self) -> FAISS | None:
        if self._store is not None:
            return self._store
        if not _index_exists(self.index_dir):
            return None
        store = FAISS.load_local(
            str(self.index_dir),
            _embeddings(),
            allow_dangerous_deserialization=True,
            normalize_L2=True,
        )
        with self._lock:
            self._store = store
        return store
