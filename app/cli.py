import argparse
import json
import sys

from app.rag import MedicalRAG
from app.store import IndexNotReady, MissingAPIKey


def main() -> None:
    parser = argparse.ArgumentParser(description="Medical document Q&A")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="Embed data/documents and build the FAISS index")

    ask = sub.add_parser("ask", help="Ask one question against the index")
    ask.add_argument("question")

    args = parser.parse_args()
    rag = MedicalRAG()
    try:
        if args.command == "ingest":
            result = rag.ingest()
            print(f"Indexed {result['chunks_indexed']} chunks.")
            return
        result = rag.ask(args.question)
    except (MissingAPIKey, IndexNotReady, FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc

    print(result["answer"])
    print()
    if result["sources"]:
        print("Sources:")
        for source in result["sources"]:
            print(f"- {source['source']} chunk {source['chunk_index']} (score {source['score']})")
    else:
        print(json.dumps({"sources": []}))


if __name__ == "__main__":
    main()
