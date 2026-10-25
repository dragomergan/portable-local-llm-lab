"""Shared helpers for the test scripts (check_server, test_glossary, test_anna, test_lore).

Standard library only. If numpy happens to be installed it is used to make the search
faster, but nothing requires it. Keep this file in the same folder as the test scripts.
"""
import array
import json
import math
import operator
import os
import re
import struct
import sys
import urllib.request

try:                       # optional speed-up
    import numpy as np
except ImportError:
    np = None

DEFAULT_SERVER = "http://localhost:5001"


# --------------------------------------------------------------------------- server calls
def post_json(url, payload, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def embed(server, text):
    """Return the embedding (list of floats) of one text."""
    return post_json(server.rstrip("/") + "/v1/embeddings", {"input": text})["data"][0]["embedding"]


def strip_think(text):
    """Remove the <think> ... </think> block that Qwen3 may write."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return re.sub(r"^.*</think>", "", text, flags=re.S).strip()


def chat(server, messages, max_tokens=500, temperature=0.2, no_think=False):
    """Send a chat request and return the answer text."""
    if no_think:                                  # Qwen3: switch the thinking mode off
        messages = messages[:-1] + [dict(messages[-1], content=messages[-1]["content"] + "\n/no_think")]
    data = post_json(server.rstrip("/") + "/v1/chat/completions",
                     {"messages": messages, "max_tokens": max_tokens, "temperature": temperature})
    return strip_think(data["choices"][0]["message"]["content"] or "")


def ask_with_context(server, system, user, **kw):
    return chat(server, [{"role": "system", "content": system}, {"role": "user", "content": user}], **kw)


def server_help(server, error):
    return ("Could not reach the model: %s\nIs KoboldCPP running at %s? "
            "(Start it with the start_model script.)" % (error, server))


# --------------------------------------------------------------------------- vector database
def read_npy(path):
    """Read a 2-D float32/float64 .npy file. Returns (rows, cols, array('f'))."""
    with open(path, "rb") as f:
        if f.read(6) != b"\x93NUMPY":
            raise ValueError(path + " is not a .npy file")
        major = f.read(2)[0]
        hlen = struct.unpack("<H", f.read(2))[0] if major == 1 else struct.unpack("<I", f.read(4))[0]
        header = f.read(hlen).decode("latin1")
        descr = re.search(r"'descr':\s*'([^']+)'", header).group(1)
        shape = [int(x) for x in re.search(r"'shape':\s*\(([^)]*)\)", header).group(1).split(",") if x.strip()]
        if "True" in re.search(r"'fortran_order':\s*(\w+)", header).group(1) or len(shape) != 2:
            raise ValueError("Expected a normal (C-order) 2-D array of vectors.")
        raw = f.read()
    rows, cols = shape
    if descr in ("<f4", "|f4"):
        data = array.array("f")
        data.frombytes(raw[:rows * cols * 4])
    elif descr == "<f8":
        big = array.array("d")
        big.frombytes(raw[:rows * cols * 8])
        data = array.array("f", big)
    else:
        raise ValueError("Unsupported dtype %s (use float32 or float64)." % descr)
    if sys.byteorder == "big":
        data.byteswap()
    return rows, cols, data


def pick_text(item):
    """Text of one chunk: a string, or the first useful string field of an object."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for k in ("text", "chunk", "content", "record", "term", "source"):
            if isinstance(item.get(k), str):
                return item[k]
        for v in item.values():
            if isinstance(v, str):
                return v
    return str(item)


class VectorDB:
    """vectors.npy + chunks.json from one folder."""

    def __init__(self, folder):
        vec_path = os.path.join(folder, "vectors.npy")
        chunk_path = os.path.join(folder, "chunks.json")
        for p in (vec_path, chunk_path):
            if not os.path.exists(p):
                raise FileNotFoundError("Not found: " + p)
        self.folder = folder
        with open(chunk_path, encoding="utf-8") as f:
            data = json.load(f)
        self.chunks = [pick_text(c) for c in (data if isinstance(data, list) else data.get("chunks", []))]
        self.n, self.d, flat = read_npy(vec_path)
        if self.n != len(self.chunks):
            raise ValueError("vectors.npy has %d rows but chunks.json has %d entries. "
                             "They must come from the same run." % (self.n, len(self.chunks)))
        if np is not None:
            m = np.frombuffer(flat, dtype=np.float32).reshape(self.n, self.d)
            norms = np.linalg.norm(m, axis=1, keepdims=True)
            norms[norms == 0] = 1
            self.matrix = m / norms
        else:
            self.rows = []
            for i in range(self.n):
                row = flat[i * self.d:(i + 1) * self.d]
                s = math.sqrt(sum(x * x for x in row)) or 1.0
                self.rows.append(array.array("f", (x / s for x in row)))

    def search(self, query_vec, k=4):
        """Return [(score, index), ...], best first."""
        if len(query_vec) != self.d:
            raise ValueError("The database has %d dimensions but the server's embedding model gives %d. "
                             "Use the same embedding model that built the database." % (self.d, len(query_vec)))
        if np is not None:
            q = np.asarray(query_vec, dtype=np.float32)
            q = q / (np.linalg.norm(q) or 1.0)
            scores = self.matrix @ q
            top = np.argsort(-scores)[:k]
            return [(float(scores[i]), int(i)) for i in top]
        s = math.sqrt(sum(x * x for x in query_vec)) or 1.0
        q = [x / s for x in query_vec]
        scored = [(sum(map(operator.mul, row, q)), i) for i, row in enumerate(self.rows)]
        scored.sort(reverse=True)
        return scored[:k]


def ask_for_folder(arg_value, default, what):
    """Folder from the command line, or ask in the terminal (Enter = default)."""
    if arg_value:
        return arg_value
    answer = input("%s [%s]: " % (what, default)).strip().strip('"')
    return answer or default


def load_db_or_exit(folder):
    try:
        db = VectorDB(folder)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
        sys.exit("Cannot load the database from '%s':\n  %s\nBuild it first with the create_*_embeddings script." % (folder, e))
    print("Loaded %d chunks, %d dimensions (%s)." % (db.n, db.d, os.path.abspath(folder)))
    return db


# --------------------------------------------------------------------------- knowledge graph
TITLES = {"the", "king", "queen", "lord", "lady", "admiral", "chancellor", "captain", "sir", "dame", "prince", "princess"}


def name_key(name):
    name = re.sub(r"['’]s\b", "", name.lower())
    words = re.sub(r"[^\w\s-]", " ", name).split()
    while words and words[0] in TITLES:
        words = words[1:]
    return " ".join(words)


def load_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def find_entities(question, nodes, aliases=None):
    """Node ids mentioned in the question: full names, aliases, or a word that belongs to one entity only."""
    q = " " + name_key(question) + " "
    names = {n["id"]: {n["id"], *n.get("aliases", [])} for n in nodes}
    for canon, alts in (aliases or {}).items():
        if canon in names:
            names[canon] |= set(alts)
    found = set()
    for nid, forms in names.items():
        for form in forms:
            k = name_key(form)
            if k and (" " + k + " ") in q:
                found.add(nid)
    owner = {}
    for nid, forms in names.items():
        for form in forms:
            for w in name_key(form).split():
                if len(w) > 2:
                    owner.setdefault(w, set()).add(nid)
    for w in set(q.split()):
        if len(owner.get(w, ())) == 1:
            found |= owner[w]
    return found


def neighborhood(edges, start, hops=2, max_facts=40):
    """Edges within `hops` steps of the start entities, nearest first."""
    seen, picked, frontier = set(start), {}, set(start)
    for h in range(1, hops + 1):
        nxt = set()
        for i, e in enumerate(edges):
            if (e["source"] in frontier or e["target"] in frontier) and i not in picked:
                picked[i] = h
                nxt |= {e["source"], e["target"]}
        frontier = nxt - seen
        seen |= nxt
    order = sorted(picked, key=lambda i: (picked[i], edges[i]["source"], edges[i]["relation"]))
    return [edges[i] for i in order[:max_facts]]
