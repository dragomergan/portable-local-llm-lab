"""Compare plain RAG with graph RAG on "The Winter of Lanterns" (the lore example).

One folder holds everything:
    graph.json        made by graph_build.py            (needed for the graph modes)
    aliases.json      nicknames, optional               (helps to find "the Pale Queen")
    vectors.npy + chunks.json   made by create_rag_embeddings.py   (needed for the plain mode)

Modes:
    plain        vector search: the top-k chunks go to the model
    graph        only the facts around the entities in the question go to the model
    graph+text   the facts, plus the source passages they came from

Usage:
    python test_lore.py                              # asks for the folder, then you type questions
    python test_lore.py --data ../data/Lore
    python test_lore.py --data ../data/Lore --run-key              # run the 10 class questions in every mode
    python test_lore.py --data ../data/Lore --run-key --report lore_report.md
    python test_lore.py --data ../data/Lore --mode graph -v        # show the facts that were sent

Start KoboldCPP first (chat model; plus embedding model for the plain mode). Needs rag_common.py in the same folder.
"""
import argparse
import os
import sys

from rag_common import (DEFAULT_SERVER, VectorDB, ask_for_folder, ask_with_context, embed, find_entities,
                        load_json, neighborhood, server_help)

# The ten questions of lore_answer_key.md, with the correct answers.
KEY = [
    ("Who is the Pale Queen?", "Isolde Varn"),
    ("Who killed Orren Thale?", "Dorn Hask (Cassian's captain)"),
    ("Who was the mentor of Tamsin Reed's uncle?", "Seraphine Dusk (the uncle is Orren Thale)"),
    ("Which two characters worked together against Queen Isolde?", "Maelric Voss and Cassian Varn"),
    ("Who carried the Charter out of Ashgate, and to whom?", "Tamsin Reed, to Admiral Halloran"),
    ("What happened to Cassian at the end?", "Exiled to the Ember Isles"),
    ("Why did Seraphine hate Voss?", "He executed her husband, Edmund Dusk"),
    ("How many characters have the surname Varn?", "3 (Aldous, Isolde, Cassian)"),
    ("Who was Isolde's mother?", "The text does not say"),
    ("How old is Tamsin Reed?", "The text does not say"),
]

SYSTEM_PLAIN = ("Answer the question using ONLY the excerpts provided. If they do not contain the answer, "
                "say that you cannot tell from the excerpts. Do not guess and do not use outside knowledge. Answer briefly.")

PROMPT_GRAPH = """Answer the question using ONLY the facts and excerpts below.
If they do not contain the answer, say that you cannot tell from the graph. Do not guess and do not use outside knowledge.

How to read the facts: each line is "subject | relation | object".
Example: "A | child_of | B" means A is the child of B. "A | killed | B" means A killed B.
Relations like sibling_of, spouse_of, ally_of and enemy_of work in both directions.

FACTS:
{facts}
{text}
QUESTION: {question}
"""


class Lore:
    def __init__(self, folder, graph_file):
        self.vectors = None
        self.graph = load_json(graph_file or os.path.join(folder, "graph.json"))
        self.aliases = load_json(os.path.join(folder, "aliases.json"), {})
        if self.graph and not ("nodes" in self.graph and "edges" in self.graph):
            sys.exit("This is not a graph file (it needs 'nodes' and 'edges'). Make one with graph_build.py.")
        try:
            self.vectors = VectorDB(folder)
        except (OSError, ValueError) as e:
            print("Plain mode not available: %s" % e)
        if self.graph:
            print("Graph: %d entities, %d facts." % (len(self.graph["nodes"]), len(self.graph["edges"])))
        else:
            print("Graph modes not available: no graph.json found in %s" % folder)
        if not self.graph and not self.vectors:
            sys.exit("Nothing to test. Put graph.json and/or vectors.npy + chunks.json in the folder.")

    def modes(self):
        return (["plain"] if self.vectors else []) + (["graph", "graph+text"] if self.graph else [])


def answer_plain(lore, args, question):
    hits = lore.vectors.search(embed(args.server, question), args.top_k)
    context = "\n\n---\n\n".join("[%d] %s" % (n, lore.vectors.chunks[i]) for n, (_, i) in enumerate(hits, 1))
    used = "chunks " + ", ".join("#%d (%.2f)" % (i, s) for s, i in hits)
    reply = ask_with_context(args.server, SYSTEM_PLAIN, "EXCERPTS:\n%s\n\nQUESTION: %s" % (context, question),
                             max_tokens=300, temperature=0.2, no_think=args.no_think)
    return reply, used, [c.replace("\n", " ")[:150] for c in (lore.vectors.chunks[i] for _, i in hits)]


def answer_graph(lore, args, question, with_text):
    g = lore.graph
    found = find_entities(question, g["nodes"], lore.aliases)
    if found:
        picked = neighborhood(g["edges"], found, args.hops, args.max_facts)
        used = "entities: %s | %d facts" % (", ".join(sorted(found)), len(picked))
    else:
        picked = g["edges"]
        used = "no entity recognised, whole graph (%d facts)" % len(picked)
    facts = "\n".join("%s | %s | %s" % (e["source"], e["relation"], e["target"]) for e in picked)
    text = ""
    if with_text and g.get("chunks"):
        votes = {}
        for e in picked:
            for c in e["chunks"]:
                votes[c] = votes.get(c, 0) + 1
        top = sorted(sorted(votes, key=lambda c: (-votes[c], c))[:args.passages])
        text = "\nEXCERPTS:\n" + "\n\n".join("[%d] %s" % (c + 1, g["chunks"][c]) for c in top) + "\n"
        used += " | passages " + ", ".join(str(c + 1) for c in top)
    reply = ask_with_context(args.server, "You answer questions about a story from a knowledge graph.",
                             PROMPT_GRAPH.format(facts=facts, text=text, question=question),
                             max_tokens=400, temperature=0.2, no_think=args.no_think)
    shown = ["%s | %s | %s" % (e["source"], e["relation"], e["target"]) for e in picked]
    return reply, used, shown


def run_mode(lore, args, mode, question):
    if mode == "plain":
        return answer_plain(lore, args, question)
    return answer_graph(lore, args, question, with_text=(mode == "graph+text"))


def ask(lore, args, modes, question, expected=None):
    print("\nQUESTION: " + question)
    if expected:
        print("EXPECTED: " + expected)
    rows = []
    for mode in modes:
        try:
            reply, used, shown = run_mode(lore, args, mode, question)
        except Exception as e:
            print(server_help(args.server, e))
            return rows
        print("\n[%s]  %s" % (mode, used))
        if args.verbose:
            for line in shown:
                print("    " + line)
        print("ANSWER: " + " ".join(reply.split()))
        rows.append((mode, used, reply))
    return rows


def write_report(path, results, modes):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Lore test report\n\nRead the answers and mark each one correct (OK) or wrong (X) by hand.\n\n")
        for n, (question, expected, rows) in enumerate(results, 1):
            f.write("## %d. %s\n\n**Expected:** %s\n\n" % (n, question, expected))
            for mode, used, reply in rows:
                f.write("- **%s** [ ]  %s\n  - sent: %s\n" % (mode, " ".join(reply.split()), used))
            f.write("\n")
    print("\nReport saved: " + os.path.abspath(path))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="folder with graph.json, aliases.json, vectors.npy, chunks.json")
    ap.add_argument("--graph", help="graph file, if it is not <data>/graph.json")
    ap.add_argument("--server", default=DEFAULT_SERVER)
    ap.add_argument("--mode", choices=["plain", "graph", "graph+text", "all"], default="all")
    ap.add_argument("--top-k", type=int, default=4, help="chunks for the plain mode (default 4)")
    ap.add_argument("--hops", type=int, default=2, help="graph steps around each entity (default 2)")
    ap.add_argument("--max-facts", type=int, default=40)
    ap.add_argument("--passages", type=int, default=2, help="source passages for graph+text (default 2)")
    ap.add_argument("--run-key", action="store_true", help="run the 10 class questions and save a report")
    ap.add_argument("--report", default="lore_report.md", help="report file for --run-key")
    ap.add_argument("-v", "--verbose", action="store_true", help="show the chunks / facts that were sent")
    ap.add_argument("--no-think", action="store_true", help="add /no_think (Qwen3)")
    args = ap.parse_args()

    lore = Lore(ask_for_folder(args.data, "../data/lore", "Folder with graph.json and vectors.npy"), args.graph)
    modes = lore.modes() if args.mode == "all" else [args.mode]
    missing = [m for m in modes if m not in lore.modes()]
    if missing:
        sys.exit("Mode '%s' is not available with the files in this folder." % missing[0])
    print("Modes: " + ", ".join(modes))

    if args.run_key:
        results = [(q, exp, ask(lore, args, modes, q, exp)) for q, exp in KEY]
        write_report(args.report, results, modes)
        return

    print("\nClass questions (type the number):")
    for n, (q, _) in enumerate(KEY, 1):
        print("  %d. %s" % (n, q))
    while True:
        try:
            text = input("\nQuestion or number (empty line to quit): ").strip()
        except EOFError:
            break
        if not text:
            break
        expected = None
        if text.isdigit() and 1 <= int(text) <= len(KEY):
            text, expected = KEY[int(text) - 1]
        ask(lore, args, modes, text, expected)


if __name__ == "__main__":
    main()
