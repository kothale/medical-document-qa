import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from langchain_google_genai.chat_models import GoogleAPIError
from pydantic import BaseModel, Field

from app.config import (
    ALLOWED_SUFFIXES,
    CHAT_MODEL,
    DOCS_DIR,
    EMBEDDING_MODEL,
    GEMINI_API_KEY,
    STATIC_DIR,
)
from app.ingest import iter_document_files
from app.rag import MedicalRAG
from app.store import IndexNotReady, MissingAPIKey, _index_exists

app = FastAPI(title="Medical Document Q&A", version="1.0.0")
rag = MedicalRAG()


class Turn(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[Turn] = Field(default_factory=list)


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def status() -> dict:
    documents = [path.name for path in iter_document_files()]
    chunk_count = None
    if _index_exists():
        try:
            chunk_count = rag.index.chunk_count()
        except MissingAPIKey:
            chunk_count = None
    return {
        "index_ready": _index_exists(),
        "chunk_count": chunk_count,
        "documents": documents,
        "embedding_model": EMBEDDING_MODEL,
        "chat_model": CHAT_MODEL,
        "api_key_configured": bool(GEMINI_API_KEY),
    }


@app.post("/api/ingest")
def ingest() -> dict:
    try:
        return rag.ingest(rebuild=True)
    except MissingAPIKey as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    name = Path(file.filename or "").name
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail="Upload a .pdf, .md, or .txt file.",
        )
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    destination = DOCS_DIR / name
    backup = destination.with_suffix(destination.suffix + ".bak")
    had_backup = False
    if destination.exists():
        destination.replace(backup)
        had_backup = True
    with destination.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    try:
        indexed = rag.ingest(rebuild=True)
    except MissingAPIKey as exc:
        _rollback_upload(destination, backup, had_backup)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        _rollback_upload(destination, backup, had_backup)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        _rollback_upload(destination, backup, had_backup)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if had_backup:
        backup.unlink(missing_ok=True)
    return {"filename": name, **indexed}


def _rollback_upload(destination: Path, backup: Path, had_backup: bool) -> None:
    destination.unlink(missing_ok=True)
    if had_backup and backup.exists():
        backup.replace(destination)


@app.post("/api/ask")
def ask(body: AskRequest) -> dict:
    history = [{"role": turn.role, "content": turn.content} for turn in body.history]
    try:
        return rag.ask(body.question, history)
    except MissingAPIKey as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except IndexNotReady as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except GoogleAPIError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
