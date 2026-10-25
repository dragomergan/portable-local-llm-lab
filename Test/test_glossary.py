"""Test a terminology (glossary) RAG database.

The database is the folder made by create_glossary_embeddings.py (vectors.npy + chunks.json).
Each record looks like:
    en: disabled (adjective)
    tr: devre dışı (preferred); gri (admitted)
    Definition: ...

Scenarios (menu):
  1. Look up a term      exact match first (shows every meaning), then the nearest records
  2. Translate a text    finds the glossary terms in your sentence and asks the model to use them
  3. Show a definition   the definition stored with the term

Usage:
    python test_glossary.py                       # asks for the folder
    python test_glossary.py --data ../data/glossary
    python test_glossary.py --data ../data/glossary --server http://localhost:5001

Start KoboldCPP first (chat model + embedding model). Needs rag_common.py in the same folder.
"""
import argparse
import re

from rag_common import (DEFAULT_SERVER, ask_for_folder, ask_with_context, embed, load_db_or_exit, server_help)

LANGUAGES = {"en": "English", "tr": "Turkish", "ar": "Arabic", "de": "German", "fr": "French", "es": "Spanish",
             "it": "Italian", "pt": "Portuguese", "ru": "Russian", "nl": "Dutch", "pl": "Polish", "ja": "Japanese",
             "zh": "Chinese", "ko": "Korean", "el": "Greek", "fa": "Persian", "he": "Hebrew"}


# --------------------------------------------------------------------------- reading records
def parse_record(text):
    """Split a record into its parts: {'langs': [(code, line)], 'definition': str}."""
    langs, definition = [], ""
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z]{2,3}(?:-[A-Za-z0-9]+)?):\s*(.*)$", line)
        if line.startswith("Definition:"):
            definition = line[len("Definition:"):].strip()
        elif m and not line.startswith(("Example", "Definition")):
            langs.append((m.group(1), m.group(2)))
    return {"langs": langs, "definition": definition}


def term_forms(line):
    """'disabled (preferred); inactive (adjective)' -> ['disabled', 'inactive']."""
    forms = []
    for part in line.split(";"):
        part = part.strip()
        while True:                                   # drop trailing "(...)" labels such as status or part of speech
            stripped = re.sub(r"\s*\([^()]*\)\s*$", "", part)
            if stripped == part:
                break
            part = stripped
        if part:
            forms.append(part)
    return forms


def norm(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s'-]", " ", text.lower())).strip()


def build_index(db):
    """normalised source term -> list of record numbers; also the language codes of the database."""
    index, source, target = {}, None, None
    for i, text in enumerate(db.chunks):
        rec = parse_record(text)
        if not rec["langs"]:
            continue
        if source is None:
            source = rec["langs"][0][0]
            target = rec["langs"][1][0] if len(rec["langs"]) > 1 else None
        for form in term_forms(rec["langs"][0][1]):
            index.setdefault(norm(form), []).append(i)
    return index, source, target


def exact_matches(index, sentence, longest=5):
    """Glossary terms that appear in the sentence (longest phrases first). Returns [(phrase, [record numbers])]."""
    words = norm(sentence).split()
    taken, found = [False] * len(words), []
    for size in range(min(longest, len(words)), 0, -1):
        for start in range(len(words) - size + 1):
            if any(taken[start:start + size]):
                continue
            phrase = " ".join(words[start:start + size])
            if phrase in index:
                found.append((phrase, index[phrase]))
                for j in range(start, start + size):
                    taken[j] = True
    return found


def similar(db, server, text, k):
    return db.search(embed(server, text), k)


def show_record(number, text, score=None):
    head = "#%d" % number + ("  similarity %.3f" % score if score is not None else "")
    print("  " + head)
    for line in text.splitlines():
        print("    " + line)


# --------------------------------------------------------------------------- scenarios
def scenario_lookup(ctx):
    term = input("\nTerm to look up (for example: disabled): ").strip()
    if not term:
        return
    hits = ctx["index"].get(norm(term), [])
    if hits:
        print("\nEXACT MATCH: %d record(s). Read them all: one word can have several meanings." % len(hits))
        for i in hits:
            show_record(i, ctx["db"].chunks[i])
    else:
        print("\nNo exact match for '%s'." % term)
    print("\nNEAREST RECORDS (by meaning):")
    for score, i in similar(ctx["db"], ctx["server"], term, ctx["k"]):
        if i not in hits:
            show_record(i, ctx["db"].chunks[i], score)


def scenario_translate(ctx):
    db, index = ctx["db"], ctx["index"]
    text = input("\nSentence to translate: ").strip()
    if not text:
        return
    exact = exact_matches(index, text)
    chosen = []                                       # record numbers sent to the model, in order
    print("\nGLOSSARY TERMS FOUND IN YOUR SENTENCE:")
    if not exact:
        print("  (none)")
    for phrase, numbers in exact:
        print("  '%s' -> %d record(s)" % (phrase, len(numbers)))
        for n in numbers:
            if n not in chosen:
                chosen.append(n)
    extra = []
    if ctx["similar"]:
        print("\nSIMILAR RECORDS (found by meaning, may be unrelated):")
        for score, i in similar(db, ctx["server"], text, ctx["similar"]):
            if i not in chosen:
                extra.append(i)
                show_record(i, db.chunks[i], score)
    entries = "\n\n".join(db.chunks[i] for i in chosen + extra)
    if not entries:
        print("\nNothing to send to the model.")
        return
    target = LANGUAGES.get((ctx["target"] or "").split("-")[0], ctx["target"] or "the target language")
    source = LANGUAGES.get((ctx["source"] or "").split("-")[0], ctx["source"] or "the source language")
    prompt = ("Translate the %s text into %s.\n"
              "Use the glossary entries below for the terms they cover. Prefer the term marked (preferred); "
              "use an (admitted) term only if it fits the context better. Never use a term marked (deprecated) or (superseded). If a word has several entries, "
              "choose the one whose definition fits the sentence. Translate everything else normally.\n"
              "Reply with the translation only.\n\nGLOSSARY:\n%s\n\nTEXT: %s" % (source, target, entries, text))
    print("\nAsking the model...")
    try:
        answer = ask_with_context(ctx["server"], "You are a careful technical translator.", prompt,
                                  max_tokens=300, temperature=0.2, no_think=ctx["no_think"])
    except Exception as e:
        print(server_help(ctx["server"], e))
        return
    print("\nTRANSLATION:\n" + answer)


def scenario_definition(ctx):
    term = input("\nTerm (for example: firewall): ").strip()
    if not term:
        return
    db = ctx["db"]
    numbers = ctx["index"].get(norm(term), [])
    if not numbers:
        score, i = similar(db, ctx["server"], term, 1)[0]
        print("No exact match. Nearest record (similarity %.3f):" % score)
        numbers = [i]
    for i in numbers:
        rec = parse_record(db.chunks[i])
        print("\n" + " | ".join("%s: %s" % (c, t) for c, t in rec["langs"]))
        print("  " + (rec["definition"] or "(this record has no definition)"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="folder with vectors.npy and chunks.json")
    ap.add_argument("--server", default=DEFAULT_SERVER)
    ap.add_argument("--top-k", type=int, default=5, help="nearest records to show in scenario 1 (default 5)")
    ap.add_argument("--similar", type=int, default=3,
                    help="extra records found by meaning and sent with the exact matches in scenario 2 (0 = exact matches only)")
    ap.add_argument("--no-think", action="store_true", help="add /no_think (Qwen3)")
    args = ap.parse_args()

    db = load_db_or_exit(ask_for_folder(args.data, "../data/glossary", "Folder with vectors.npy and chunks.json"))
    index, source, target = build_index(db)
    print("Languages: %s -> %s. %d searchable source terms." % (source, target, len(index)))
    ctx = {"db": db, "index": index, "source": source, "target": target, "server": args.server,
           "k": args.top_k, "similar": args.similar, "no_think": args.no_think}

    menu = {"1": scenario_lookup, "2": scenario_translate, "3": scenario_definition}
    while True:
        print("\n=== GLOSSARY TEST ===\n1. Look up a term\n2. Translate a sentence with the glossary\n3. Show a definition\n4. Quit")
        choice = input("Choice (1-4): ").strip()
        if choice == "4" or choice.lower() in ("q", "quit"):
            break
        if choice in menu:
            try:
                menu[choice](ctx)
            except Exception as e:
                print("\nError: %s\n(If this is a connection error, check that KoboldCPP is running at %s.)" % (e, args.server))
        else:
            print("Please type 1, 2, 3 or 4.")


if __name__ == "__main__":
    main()
