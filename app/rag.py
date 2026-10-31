from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import CHAT_MODEL, GEMINI_API_KEY, SCORE_THRESHOLD, TOP_K
from app.prompts import PROMPT, format_context, format_history
from app.sanitize import scrub_phi
from app.store import IndexNotReady, MissingAPIKey, VectorIndex

INSUFFICIENT_CONTEXT = (
    "The indexed medical documents do not contain enough relevant information "
    "to answer that. Try rephrasing, or add a document that covers the topic. "
    "This is educational information from the indexed documents, not medical advice."
)


def _chat_model() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=GEMINI_API_KEY,
        timeout=45,
    )


class MedicalRAG:
    def __init__(self, index: VectorIndex | None = None) -> None:
        self.index = index or VectorIndex()

    def ingest(self, rebuild: bool = True) -> dict:
        count = self.index.rebuild()
        return {"chunks_indexed": count, "rebuilt": rebuild}

    def ask(self, question: str, history: list[dict] | None = None) -> dict:
        question = question.strip()
        if not question:
            raise ValueError("Question is empty.")
        if not GEMINI_API_KEY:
            raise MissingAPIKey(
                "Set GEMINI_API_KEY in a .env file before asking a question."
            )

        matches = self.index.search(question, k=TOP_K)
        kept = [
            (doc, score)
            for doc, score in matches
            if score >= SCORE_THRESHOLD
        ]
        sources = [_source_payload(doc, score) for doc, score in kept]
        if not sources:
            return {"answer": INSUFFICIENT_CONTEXT, "sources": [], "used_llm": False}

        chain = PROMPT | _chat_model() | StrOutputParser()
        answer = chain.invoke(
            {
                "context": format_context(sources),
                "history": format_history(history or []),
                "question": question,
            }
        )
        return {"answer": answer.strip(), "sources": sources, "used_llm": True}


def _source_payload(doc: Document, score: float) -> dict:
    text = scrub_phi(doc.page_content.strip())
    return {
        "source": doc.metadata.get("source", "unknown"),
        "page": doc.metadata.get("page"),
        "chunk_index": doc.metadata.get("chunk_index", 0),
        "score": round(float(score), 3),
        "text": text,
    }
