"""Build a small knowledge graph from a short text with a local LLM.

Standard library only.

Usage:
    python graph_build.py                       # lore.txt -> graph.json
    python graph_build.py story.txt out.json
    python graph_build.py lore.txt graph.json --aliases aliases.json
    python graph_build.py lore.txt graph.json --aliases none   # see raw, unmerged names
    python graph_build.py lore.txt graph.json --no-think       # for Qwen3

Start KoboldCPP first (chat model, port 5001).

Pipeline:
  1. Split the text into small chunks (a few paragraphs each).
  2. Ask the model for facts as (subject, relation, object) triples, as JSON.
  3. Clean up: merge names that mean the same entity, unify relation labels,
     remove duplicates, remember which chunk each fact came from.
  4. Save graph.json and print every edge so you can check it by eye.
"""
import argparse
import json
import re
import sys
import urllib.request
from collections import Counter, defaultdict

MAX_WORDS = 170  # chunk size for extraction (small chunks work best for small models)

ENTITY_TYPES = ["Person", "Place", "Group", "Item", "Event"]

PREFERRED_RELATIONS = [
    "ruler_of", "child_of", "sibling_of", "spouse_of", "relative_of",
    "ally_of", "enemy_of", "mentor_of", "member_of", "leader_of", "serves",
    "holds", "guards", "located_in", "killed", "betrayed", "exiled_to",
    "carried_to", "opposed", "proposed",
]

# Relations that read the same in both directions.
SYMMETRIC = {"sibling_of", "spouse_of", "ally_of", "enemy_of"}

# Label clean-up: wrong or inverse wording -> (canonical label, swap subject/object?)
RELATION_MAP = {
    "parent_of": ("child_of", True), "father_of": ("child_of", True), "mother_of": ("child_of", True),
    "son_of": ("child_of", False), "daughter_of": ("child_of", False),
    "brother_of": ("sibling_of", False), "sister_of": ("sibling_of", False),
    "husband_of": ("spouse_of", False), "wife_of": ("spouse_of", False),
    "married_to": ("spouse_of", False), "betrothed_to": ("spouse_of", False),
    "uncle_of": ("relative_of", False), "aunt_of": ("relative_of", False),
    "niece_of": ("relative_of", False), "nephew_of": ("relative_of", False),
    "allied_with": ("ally_of", False), "enemies_with": ("enemy_of", False),
    "rules": ("ruler_of", False), "ruled_by": ("ruler_of", True), "rules_over": ("ruler_of", False),
    "leads": ("leader_of", False), "led_by": ("leader_of", True), "head_of": ("leader_of", False),
    "killed_by": ("killed", True), "murdered": ("killed", False), "executed": ("killed", False),
    "betrayed_by": ("betrayed", True), "mentor": ("mentor_of", False), "taught": ("mentor_of", False),
    "taught_by": ("mentor_of", True), "mentored_by": ("mentor_of", True),
    "serves_in": ("serves", False), "belongs_to": ("member_of", False),
    "possesses": ("holds", False), "keeps": ("holds", False), "owns": ("holds", False), "carries": ("holds", False),
    "protects": ("guards", False), "lives_in": ("located_in", False),
}

TITLES = {"the", "king", "queen", "lord", "lady", "admiral", "chancellor", "captain", "sir", "dame", "prince", "princess"}

SYSTEM_PROMPT = """You extract facts from a story to build a knowledge graph.

Return ONLY a JSON array. Each item has exactly these keys:
  "subject", "subject_type", "relation", "object", "object_type"

Rules:
- Use a full name, never a pronoun (he, she, it, they, her brother).
- Only facts the text states clearly. Do not guess and do not add outside knowledge.
- subject_type and object_type must be one of: {types}.
- relation is a short snake_case label. Prefer one of: {relations}.
  If none fits, use a short snake_case verb phrase of your own.
- One fact per item. The subject performs or owns the relation.

Example input: "Mira, the miller's daughter, guards the Old Bridge. Her rival, Tobin, leads the Wolf Guild."
Example output:
[{{"subject": "Mira", "subject_type": "Person", "relation": "guards", "object": "Old Bridge", "object_type": "Place"}},
 {{"subject": "Tobin", "subject_type": "Person", "relation": "leader_of", "object": "Wolf Guild", "object_type": "Group"}},
 {{"subject": "Mira", "subject_type": "Person", "relation": "enemy_of", "object": "Tobin", "object_type": "Person"}}]

TEXT:
{text}
"""


# ---------------------------------------------------------------- chunking
def split_chunks(text, max_words=MAX_WORDS):
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur, count = [], [], 0
    for p in paragraphs:
        w = len(p.split())
        if cur and count + w > max_words:
            chunks.append("\n\n".join(cur))
            cur, count = [], 0
        cur.append(p)
        count += w
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


# ---------------------------------------------------------------- LLM call
def ask_llm(url, text, no_think):
    prompt = SYSTEM_PROMPT.format(types=", ".join(ENTITY_TYPES), relations=", ".join(PREFERRED_RELATIONS), text=text)
    if no_think:
        prompt += "\n/no_think"
    payload = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 1500, "temperature": 0}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))["choices"][0]["message"]["content"]


def parse_triples(reply):
    """Pull a list of fact dicts out of a model reply, even if it adds prose or code fences."""
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.S)
    reply = re.sub(r"^.*</think>", "", reply, flags=re.S)
    items = None
    m = re.search(r"\[.*\]", reply, re.S)
    if m:
        try:
            items = json.loads(m.group(0))
        except json.JSONDecodeError:
            items = None
    if items is None:  # fall back: collect individual {...} objects
        items = []
        for obj in re.findall(r"\{[^{}]*\}", reply, re.S):
            try:
                items.append(json.loads(obj))
            except json.JSONDecodeError:
                pass
    good = []
    for it in items if isinstance(items, list) else []:
        if isinstance(it, dict) and all(isinstance(it.get(k), str) and it.get(k).strip() for k in ("subject", "relation", "object")):
            good.append(it)
    return good


# ---------------------------------------------------------------- cleaning
def key(name):
    """Comparison key: lowercase, no possessive, no leading titles."""
    name = re.sub(r"['’]s\b", "", name.lower())
    words = re.sub(r"[^\w\s-]", " ", name).split()
    while words and words[0] in TITLES:
        words = words[1:]
    return " ".join(words)


def tidy(name):
    name = " ".join(name.split()).strip(" .,;:\"'")
    return re.sub(r"^(the|a|an)\s+", "", name, flags=re.I)


def load_aliases(path):
    amap = {}
    if path and path.lower() != "none":
        with open(path, encoding="utf-8") as f:
            for canon, alts in json.load(f).items():
                amap[key(canon)] = canon
                for a in alts:
                    amap[key(a)] = canon
    return amap


def clean_relation(rel, subj, obj):
    rel = re.sub(r"[^a-z0-9]+", "_", rel.lower()).strip("_")
    if rel in RELATION_MAP:
        rel, swap = RELATION_MAP[rel]
        if swap:
            subj, obj = obj, subj
    return rel, subj, obj


def build(chunk_triples, amap):
    """chunk_triples: list of (chunk_index, triple_dict). Returns nodes, edges, merge log."""
    surface = defaultdict(set)         # canonical -> surface forms seen
    types = defaultdict(Counter)       # canonical -> type votes
    raw = []                           # (chunk, subj, rel, obj)

    def resolve(name):
        name = tidy(name)
        canon = amap.get(key(name), name)
        if canon != name:
            surface[canon].add(name)
        return canon

    for ci, t in chunk_triples:
        s, o = resolve(t["subject"]), resolve(t["object"])
        rel, s, o = clean_relation(t["relation"], s, o)
        for n, ty in ((t["subject"], t.get("subject_type")), (t["object"], t.get("object_type"))):
            c = resolve(n)
            types[c][ty if ty in ENTITY_TYPES else "Other"] += 1
        raw.append((ci, s, rel, o))

    # Automatic merge: a one-word name that is part of exactly one longer name.
    names = {n for _, s, _, o in raw for n in (s, o)}
    merges = {}
    for n in sorted(names):
        k = key(n)
        if " " in k or k in amap:
            continue
        cands = [m for m in names if m != n and k in key(m).split()]
        if len(cands) == 1:
            merges[n] = cands[0]
    if merges:
        raw = [(ci, merges.get(s, s), r, merges.get(o, o)) for ci, s, r, o in raw]
        for short, full in merges.items():
            surface[full].add(short)
            types[full].update(types.get(short, {}))

    edges = {}
    for ci, s, rel, o in raw:
        if s == o:
            continue
        if rel in SYMMETRIC and s > o:
            s, o = o, s
        e = edges.setdefault((s, rel, o), {"source": s, "relation": rel, "target": o, "chunks": set()})
        e["chunks"].add(ci)

    edge_list = sorted(({**e, "chunks": sorted(e["chunks"])} for e in edges.values()),
                       key=lambda e: (e["source"], e["relation"], e["target"]))
    mention = Counter()
    for e in edge_list:
        mention[e["source"]] += 1
        mention[e["target"]] += 1
    nodes = []
    for n in sorted(mention):
        ty = types[n].most_common(1)[0][0] if types[n] else "Other"
        nodes.append({"id": n, "type": ty, "aliases": sorted(surface[n] - {n}), "edges": mention[n]})
    return nodes, edge_list, merges


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", default="lore.txt")
    ap.add_argument("out", nargs="?", default="graph.json")
    ap.add_argument("--aliases", default="aliases.json", help="alias file, or 'none'")
    ap.add_argument("--url", default="http://localhost:5001/v1/chat/completions")
    ap.add_argument("--no-think", action="store_true", help="add /no_think (Qwen3)")
    a = ap.parse_args()

    try:
        text = open(a.text, encoding="utf-8-sig").read()
    except OSError as e:
        sys.exit("Cannot read " + a.text + ": " + str(e))
    try:
        amap = load_aliases(a.aliases)
    except (OSError, json.JSONDecodeError) as e:
        print("Alias file not used (" + str(e) + "). Continuing without it.")
        amap = {}

    chunks = split_chunks(text)
    print(len(chunks), "chunks. Asking the model for facts (this can take a few minutes)...")
    found = []
    for i, chunk in enumerate(chunks):
        triples = []
        for attempt in (1, 2):
            try:
                triples = parse_triples(ask_llm(a.url, chunk, a.no_think))
            except Exception as e:
                print("  chunk", i + 1, "request failed:", e)
                triples = []
            if triples:
                break
        print("  chunk %d/%d: %d facts" % (i + 1, len(chunks), len(triples)))
        found += [(i, t) for t in triples]

    if not found:
        sys.exit("No facts extracted. Is the server running? Try --no-think for Qwen3.")

    nodes, edges, merges = build(found, amap)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"nodes": nodes, "edges": edges, "chunks": chunks}, f, ensure_ascii=False, indent=2)

    print("\nSaved %s: %d nodes, %d edges\n" % (a.out, len(nodes), len(edges)))
    if merges:
        print("Merged automatically:", ", ".join(s + " -> " + f for s, f in merges.items()))
    print("EDGES (check these by eye)")
    for e in edges:
        print("  %s --%s--> %s   [chunks %s]" % (e["source"], e["relation"], e["target"], ",".join(str(c + 1) for c in e["chunks"])))
    once = [n["id"] for n in nodes if n["edges"] == 1]
    if once:
        print("\nOnly one edge (possible alias or extraction mistake):", ", ".join(once))
    print("\nNode types:", dict(Counter(n["type"] for n in nodes)))


if __name__ == "__main__":
    main()
