"""Ask questions about a book (or any long text) with RAG, and see exactly what the model was given.

The database is the folder made by create_rag_embeddings.py (vectors.npy + chunks.json).
It is called test_anna because the course example is Anna Karenina, but any book works.

What it shows, for every question:
  - the chunks that were retrieved (with similarity scores)
  - the model's answer built only from those chunks

Usage:
    python test_anna.py                          # asks for the folder
    python test_anna.py --data ../data/anna_karenina
    python test_anna.py --data ../data/anna_karenina --top-k 6
    python test_anna.py --data ../data/anna_karenina --compare     # top-1 against top-k, side by side

Type any question you like. Empty line = quit.
Start KoboldCPP first (chat model + embedding model). Needs rag_common.py in the same folder.
"""
import argparse

from rag_common import (DEFAULT_SERVER, ask_for_folder, ask_with_context, embed, load_db_or_exit, server_help)

SYSTEM = ("You are a careful assistant. Answer the question using ONLY the excerpts provided. "
          "If the excerpts do not contain the answer, say that you cannot tell from the excerpts. "
          "Do not guess and do not use outside knowledge. Answer briefly.")

def show_chunk(rank, score, index, text, width=300):
    flat = " ".join(text.split())
    print("  [%d] chunk #%d  similarity %.3f\n      %s%s" % (rank, index, score, flat[:width], "..." if len(flat) > width else ""))


def run(db, args, question, k):
    try:
        hits = db.search(embed(args.server, question), k)
    except Exception as e:
        print(server_help(args.server, e))
        return
    print("\n--- top-%d ---" % k)
    for rank, (score, i) in enumerate(hits, 1):
        show_chunk(rank, score, i, db.chunks[i])
    context = "\n\n---\n\n".join("[%d] %s" % (n, db.chunks[i]) for n, (_, i) in enumerate(hits, 1))
    try:
        answer = ask_with_context(args.server, SYSTEM, "EXCERPTS:\n%s\n\nQUESTION: %s" % (context, question),
                                  max_tokens=300, temperature=0.2, no_think=args.no_think)
    except Exception as e:
        print(server_help(args.server, e))
        return
    print("\nANSWER (top-%d):\n%s" % (k, answer))


def ask(db, args, question):
    print("\nQUESTION: " + question)
    for k in ([1, args.top_k] if args.compare and args.top_k != 1 else [args.top_k]):
        run(db, args, question, k)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="folder with vectors.npy and chunks.json")
    ap.add_argument("--server", default=DEFAULT_SERVER)
    ap.add_argument("--top-k", type=int, default=4, help="how many chunks to give the model (default 4)")
    ap.add_argument("--compare", action="store_true", help="answer each question twice: with 1 chunk and with top-k chunks")
    ap.add_argument("--no-think", action="store_true", help="add /no_think (Qwen3)")
    args = ap.parse_args()

    db = load_db_or_exit(ask_for_folder(args.data, "../data/anna_karenina", "Folder with vectors.npy and chunks.json"))

    print("\nAsk anything about the text.")
    while True:
        try:
            text = input("\nQuestion (empty line to quit): ").strip()
        except EOFError:
            break
        if not text:
            break
        ask(db, args, text)


if __name__ == "__main__":
    main()
