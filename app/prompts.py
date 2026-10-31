from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You answer questions about medical documents that were retrieved for this turn.

Rules:
- Use only the context excerpts. Do not add facts from outside those excerpts.
- If the excerpts do not contain the answer, say that the indexed documents do not cover it. Do not guess.
- Do not diagnose a person, prescribe a drug, choose a dose, or give a personalized treatment plan.
- When you use an excerpt, cite its source label in square brackets, for example [hypertension.md].
- Write in plain language. Keep the answer short enough to scan.
- End with this sentence: This is educational information from the indexed documents, not medical advice. A clinician should interpret it for a specific person.
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            """Context excerpts:
{context}

Recent conversation (for wording only — not a source of medical facts):
{history}

Question:
{question}""",
        ),
    ]
)


def format_context(chunks: list[dict]) -> str:
    blocks: list[str] = []
    for item in chunks:
        page = item.get("page")
        page_label = f", page {page + 1}" if isinstance(page, int) else ""
        blocks.append(
            f"[{item['source']} chunk {item['chunk_index']}{page_label}]\n{item['text']}"
        )
    return "\n\n".join(blocks)


def format_history(history: list[dict]) -> str:
    if not history:
        return "(none)"
    lines: list[str] = []
    for turn in history[-4:]:
        role = turn.get("role", "user")
        if role not in {"user", "assistant"}:
            continue
        content = str(turn.get("content", "")).strip()
        if not content:
            continue
        lines.append(f"{role}: {content[:500]}")
    return "\n".join(lines) if lines else "(none)"
