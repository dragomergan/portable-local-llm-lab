#!/usr/bin/env python3
"""Build a RAG index (vectors + text chunks) from plain-text files.

Standard library only: no numpy, no pip install. The vectors are written in the
standard NumPy .npy format, so numpy.load() can read them, but numpy is not needed
to create them.

What it does
  1. Reads one .txt file, or every .txt file in a folder.
  2. Removes the Project Gutenberg header and license text, if present.
  3. Cuts the text into chunks of about CHUNK_WORDS words, always at paragraph
     boundaries (very long paragraphs are cut at sentence boundaries).
     Each chunk repeats the last paragraph(s) of the previous one (OVERLAP words).
  4. Sends the chunks to an embedding server in batches and saves the result.

Output (written to the output folder)
  vectors.npy   float32 array, one row per chunk
  chunks.json   list of strings, same order as the rows of vectors.npy
  meta.json     settings and sizes used for this run (handy to check that two
                indexes were built the same way)

Usage
  python create_rag_embeddings.py                        # opens folder pickers
  python create_rag_embeddings.py -i book.txt -o out/
  python create_rag_embeddings.py -i texts_folder/ -o out/ --chunk-words 200 --overlap 40
  python create_rag_embeddings.py -i book.txt -o out/ --force   # overwrite existing files

Before you run it, start an embedding server that offers the OpenAI-style
/v1/embeddings endpoint. With KoboldCPP, add --embeddingsmodel <file.gguf> to
your start script (default address: http://localhost:5001).

Important: build every index you want to compare or search together with the SAME
embedding model. Vectors from different models cannot be mixed.
"""
import argparse
import json
import os
import re
import struct
import sys
import time
import urllib.request
from array import array
from datetime import datetime, timezone

DEFAULT_URL = "http://localhost:5001/v1/embeddings"
DEFAULT_CHUNK_WORDS = 300   # target chunk size, counted in words (not tokens)
DEFAULT_OVERLAP = 50        # words carried over from one chunk to the next
DEFAULT_BATCH = 16          # chunks sent to the server in one request
RETRIES = 3
MIN_CHUNK_CHARS = 50        # shorter leftovers are dropped


# --------------------------------------------------------------------------- reading text
def read_text_file(path):
    """Read a text file, trying UTF-8 first and Windows-1252 as a fallback."""
    with open(path, "rb") as f:
        raw = f.read()
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def strip_gutenberg(text):
    """Remove the Project Gutenberg header and footer, if they are present."""
    start = re.search(r"\*\*\* ?START OF.*?\*\*\*", text, re.S)
    if start:
        text = text[start.end():]
    end = re.search(r"\*\*\* ?END OF", text)
    if end:
        text = text[:end.start()]
    return text


def find_input_files(path):
    """Return the list of .txt files for a file or a folder path."""
    if os.path.isdir(path):
        files = sorted(
            os.path.join(path, name)
            for name in os.listdir(path)
            if name.lower().endswith(".txt") and os.path.isfile(os.path.join(path, name))
        )
        return files
    return [path] if os.path.isfile(path) else []


# --------------------------------------------------------------------------- chunking
def split_paragraphs(text):
    """Split on blank lines; line breaks inside a paragraph become spaces."""
    text = text.replace("\r\n", "\n")
    paragraphs = []
    for block in re.split(r"\n\s*\n", text):
        block = " ".join(block.split())
        if block:
            paragraphs.append(block)
    return paragraphs


def split_long_paragraph(paragraph, size):
    """Cut a paragraph longer than `size` words at sentence boundaries."""
    if len(paragraph.split()) <= size:
        return [paragraph]
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    pieces, current, count = [], [], 0
    for sentence in sentences:
        words = len(sentence.split())
        if current and count + words > size:
            pieces.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += words
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_text(text, chunk_words=DEFAULT_CHUNK_WORDS, overlap=DEFAULT_OVERLAP):
    """Group paragraphs into chunks of about `chunk_words` words.

    A new chunk starts with the last paragraph(s) of the previous chunk, up to
    `overlap` words. If the last paragraph alone is longer than `overlap`, nothing
    is carried over for that chunk.
    """
    units = []
    for paragraph in split_paragraphs(text):
        units.extend(split_long_paragraph(paragraph, chunk_words))

    chunks, current, count = [], [], 0
    for unit in units:
        words = len(unit.split())
        if current and count + words > chunk_words:
            chunks.append("\n\n".join(current))
            carried, carried_words = [], 0
            for previous in reversed(current):
                previous_words = len(previous.split())
                if carried_words + previous_words > overlap:
                    break
                carried.insert(0, previous)
                carried_words += previous_words
            current, count = carried, carried_words
        current.append(unit)
        count += words
    if current:
        chunks.append("\n\n".join(current))
    return [c for c in chunks if len(c.strip()) > MIN_CHUNK_CHARS]


# --------------------------------------------------------------------------- embeddings
def get_embeddings(url, texts):
    """Return one vector per text, or None if the server keeps failing."""
    payload = json.dumps({"input": texts}).encode()
    for attempt in range(1, RETRIES + 1):
        try:
            request = urllib.request.Request(
                url, data=payload, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(request, timeout=300) as response:
                data = json.loads(response.read().decode())["data"]
            data.sort(key=lambda item: item["index"])
            if len(data) != len(texts):
                raise ValueError("sent %d texts, got %d vectors" % (len(texts), len(data)))
            return [item["embedding"] for item in data]
        except Exception as error:
            print("\n  Embedding request failed (attempt %d/%d): %s" % (attempt, RETRIES, error))
            time.sleep(2)
    return None


# --------------------------------------------------------------------------- saving
def write_npy(path, rows):
    """Write a 2D float32 array in the .npy format (version 1.0) without numpy."""
    n_rows, n_cols = len(rows), len(rows[0])
    flat = array("f")
    for row in rows:
        if len(row) != n_cols:
            raise ValueError("vectors have different lengths")
        flat.extend(row)
    if sys.byteorder == "big":
        flat.byteswap()  # .npy '<f4' is little-endian
    header = "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }" % (n_rows, n_cols)
    padding = (-(10 + len(header) + 1)) % 64  # magic(6) + version(2) + length(2) + header + newline
    header_bytes = (header + " " * padding + "\n").encode("latin1")
    with open(path, "wb") as f:
        f.write(b"\x93NUMPY" + bytes([1, 0]) + struct.pack("<H", len(header_bytes)) + header_bytes)
        flat.tofile(f)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_atomically(write_function, path, data):
    """Write to a temporary file first, so an interrupted run never leaves a half-written file."""
    temporary = path + ".tmp"
    write_function(temporary, data)
    os.replace(temporary, path)


# --------------------------------------------------------------------------- choosing folders
def pick_with_dialogs():
    """Ask for the input and output locations with folder pickers (needs tkinter)."""
    import tkinter
    from tkinter import filedialog

    root = tkinter.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    chosen_input = filedialog.askopenfilename(
        title="Select a .txt file (press Cancel to select a folder of .txt files instead)",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
    )
    if not chosen_input:
        chosen_input = filedialog.askdirectory(title="Select the folder that contains your .txt files")
    chosen_output = filedialog.askdirectory(
        title="Select the output folder (vectors.npy and chunks.json will be saved here)",
        mustexist=False,
    )
    root.destroy()
    return chosen_input, chosen_output


def pick_with_prompts():
    """Fallback when no window system is available: ask in the terminal."""
    chosen_input = input("Path to a .txt file or a folder of .txt files: ").strip().strip('"')
    chosen_output = input("Output folder (will be created if missing): ").strip().strip('"')
    return chosen_input, chosen_output


def choose_locations(input_path, output_path):
    if input_path and output_path:
        return input_path, output_path
    try:
        picked_input, picked_output = pick_with_dialogs()
    except Exception:  # tkinter missing, or no display
        print("No file dialog available here, so I will ask in the terminal.")
        picked_input, picked_output = pick_with_prompts()
    return input_path or picked_input, output_path or picked_output


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(
        description="Build vectors.npy + chunks.json from plain-text files.",
        epilog="Run without -i and -o to choose the folders in a dialog window.",
    )
    parser.add_argument("-i", "--input", help="a .txt file, or a folder with .txt files")
    parser.add_argument("-o", "--output", help="folder for vectors.npy, chunks.json and meta.json")
    parser.add_argument("--url", default=DEFAULT_URL, help="embeddings endpoint (default: %(default)s)")
    parser.add_argument("--chunk-words", type=int, default=DEFAULT_CHUNK_WORDS, help="chunk size in words (default: %(default)s)")
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP, help="overlap in words (default: %(default)s)")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="chunks per request (default: %(default)s)")
    parser.add_argument("--keep-gutenberg", action="store_true", help="do not remove the Project Gutenberg header and footer")
    parser.add_argument("--force", action="store_true", help="overwrite existing output files without asking")
    args = parser.parse_args()

    if args.chunk_words < 20 or args.overlap < 0 or args.overlap >= args.chunk_words or args.batch < 1:
        sys.exit("Invalid settings: use --chunk-words >= 20, 0 <= --overlap < --chunk-words, --batch >= 1.")

    input_path, output_dir = choose_locations(args.input, args.output)
    if not input_path or not output_dir:
        sys.exit("Input and output locations are both required.")

    files = find_input_files(input_path)
    if not files:
        sys.exit("No .txt file found at: %s" % input_path)

    # Do not overwrite earlier work by accident.
    targets = [os.path.join(output_dir, name) for name in ("vectors.npy", "chunks.json", "meta.json")]
    existing = [t for t in targets if os.path.exists(t)]
    if existing and not args.force:
        if sys.stdin.isatty():
            answer = input("Files already exist in %s. Overwrite? [y/N] " % output_dir).strip().lower()
            if answer != "y":
                sys.exit("Stopped. Nothing was changed.")
        else:
            sys.exit("Files already exist in %s. Use --force to overwrite." % output_dir)

    print("Reading %d file(s)..." % len(files))
    texts = []
    for path in files:
        text = read_text_file(path)
        texts.append(text if args.keep_gutenberg else strip_gutenberg(text))
    chunks = chunk_text("\n\n".join(texts), args.chunk_words, args.overlap)
    if not chunks:
        sys.exit("The text is empty or too short to make chunks.")
    sizes = [len(c.split()) for c in chunks]
    print("%d chunks (average %d words, longest %d)." % (len(chunks), sum(sizes) // len(sizes), max(sizes)))

    # A quick test request, so a wrong address or a missing embeddings model is reported at once.
    print("Testing the embedding server at %s ..." % args.url)
    probe = get_embeddings(args.url, ["test"])
    if probe is None:
        sys.exit(
            "Cannot get embeddings. Is the server running, and was it started with an embeddings model\n"
            "(for KoboldCPP: --embeddingsmodel <file.gguf>)?"
        )
    dimension = len(probe[0])
    print("Server works. Vector size: %d." % dimension)

    vectors, kept_chunks, skipped = [], [], 0
    started = time.time()
    for start in range(0, len(chunks), args.batch):
        batch = chunks[start:start + args.batch]
        result = get_embeddings(args.url, batch)
        if result is None:
            skipped += len(batch)
            print("  Skipped chunks %d-%d." % (start + 1, start + len(batch)))
            continue
        vectors.extend(result)
        kept_chunks.extend(batch)
        done = min(start + args.batch, len(chunks))
        print("Embedded %d / %d  (%d s)" % (done, len(chunks), time.time() - started), end="\r")
    print()

    if not vectors:
        sys.exit("No vectors were created.")

    os.makedirs(output_dir, exist_ok=True)
    save_atomically(write_npy, os.path.join(output_dir, "vectors.npy"), vectors)
    save_atomically(write_json, os.path.join(output_dir, "chunks.json"), kept_chunks)
    meta = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_files": [os.path.basename(p) for p in files],
        "chunks": len(kept_chunks),
        "skipped_chunks": skipped,
        "vector_size": dimension,
        "chunk_words": args.chunk_words,
        "overlap_words": args.overlap,
        "gutenberg_stripped": not args.keep_gutenberg,
        "embedding_url": args.url,
    }
    save_atomically(write_json, os.path.join(output_dir, "meta.json"), meta)

    print("Done. %d chunks saved to %s" % (len(kept_chunks), os.path.abspath(output_dir)))
    if skipped:
        print("Warning: %d chunks were skipped because the server failed. Run again to build a complete index." % skipped)


if __name__ == "__main__":
    main()