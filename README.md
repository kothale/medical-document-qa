# Medical Document Q&A

Retrieval-augmented question answering over medical documents. Python, LangChain, Gemini embeddings and chat, and a local FAISS index.

This follows the normal document RAG path: embed the question, retrieve the closest chunks, put those chunks into a grounded prompt, and generate an answer. It does not call live hospital systems, and it does not give personal medical advice.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# put your Gemini key in .env as GEMINI_API_KEY
python -m app.cli ingest
python -m app.cli ask "What sodium limit does the blood pressure handout suggest?"
uvicorn app.api:app --port 8000
```

Open http://127.0.0.1:8000. Sample handouts already live in `data/documents/`. Upload replaces or adds a `.pdf`, `.md`, or `.txt` file and rebuilds the index.

Get a key from [Google AI Studio](https://aistudio.google.com/apikey). `gemini-embedding-001` and `gemini-3.5-flash-lite` are on the Gemini API free tier, with per-minute and per-day caps. On that free tier, Google can use submitted content to improve its products, so keep real patient records out of this demo. If you already built an index with another embedding model, click **Rebuild index** before asking questions.

## Pipeline

```text
documents (.md .txt .pdf)
        |
        v
load and split into overlapping chunks
        |
        v
Gemini gemini-embedding-001
        |
        v
FAISS L2 index on disk
        |
        v
question --> same embedding model --> top-k search
        |
        v
convert distance to cosine similarity and drop weak chunks
        |
        v
scrub emails, phones, SSNs, and record numbers
        |
        v
prompt: rules + excerpts + last 4 turns + question
        |
        v
gemini-3.5-flash-lite answer, with the excerpts returned beside it
```

If nothing clears the relevance threshold, the app does not call the chat model. It says the documents do not cover the question.

## Layout

| Path | Role |
| --- | --- |
| `app/ingest.py` | Load PDF, Markdown, and text. Split into chunks. |
| `app/store.py` | Embed chunks and save or search the FAISS index. |
| `app/sanitize.py` | Mask obvious identifiers before they enter the prompt. |
| `app/prompts.py` | Grounding rules, source labels, and history formatting. |
| `app/rag.py` | Retrieval gate, then one chat completion. |
| `app/api.py` | Status, rebuild, upload, and ask. |
| `app/cli.py` | `ingest` and `ask` without the browser. |
| `static/index.html` | Chat page, source cards, and index controls. |

The index files under `storage/` are built locally and are not committed. `allow_dangerous_deserialization` is set only because LangChain loads this app's own index pickle from disk.
