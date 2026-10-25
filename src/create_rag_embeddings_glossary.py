#!/usr/bin/env python3
"""Build a RAG index (vectors + text records) from a terminology JSON file.

Works with the JSON files made by tbx_to_json.py: a list of entries, each with a
"languages" object that holds "terms", "definitions" and, if present, "examples" and
"term_details" (status, part of speech) for every language.

Standard library only: no numpy, no pip install. The vectors are saved in the
standard NumPy .npy format, so numpy.load() can read them, but numpy is not needed
to create them.

What it does
  1. Asks for the terminology JSON file and for the output folder (dialog windows,
     or -i / -o on the command line).
  2. Detects the languages in the file and asks which one is the SOURCE and which
     ones are the TARGET languages (defaults are suggested; press Enter to accept).
  3. Makes one record per entry, so the meaning (definition) stays attached to its
     translations. A record looks like this:

         en: disabled (adjective)
         tr: devre dışı (preferred); gri (admitted)
         Definition: Not available for use; turned off so that it does not respond.
         Example (en): The Save button is disabled until all fields are filled in.

  4. Embeds only the SOURCE term(s) (not the whole record), so a search for a term
     is not diluted by definitions or boilerplate text. The full record is what the
     language model later receives. Use --embed term+definition to embed the
     definition as well.
  5. Saves vectors.npy, chunks.json (the records, same order) and meta.json.

A long run can be stopped and continued: progress is saved after every batch.
Run the same command again and it resumes where it stopped.

Usage
  python create_glossary_embeddings.py                       # dialogs
  python create_glossary_embeddings.py -i terms.json -o out/
  python create_glossary_embeddings.py -i terms.json -o out/ --source en --target tr
  python create_glossary_embeddings.py -i terms.json -o out/ --limit 300   # quick test run

Start an embedding server first. With KoboldCPP, add --embeddingsmodel <file.gguf>
to your start script (default address: http://localhost:5001).

Important: build every index you want to search together with the SAME embedding
model. Vectors from different models cannot be mixed.
"""
import argparse
import hashlib
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
DEFAULT_BATCH = 32
RETRIES = 3
PARTIAL_DATA = "vectors.partial.f32"
PARTIAL_META = "partial.json"
IGNORED_POS = {"", "notselected", "other", "unknown", "none"}


# --------------------------------------------------------------------------- reading the JSON
def load_entries(path):
    """Read the terminology JSON and return the list of entries that have languages."""
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("entries", "terms", "value"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list of entries (as made by tbx_to_json.py).")
    return [e for e in data if isinstance(e, dict) and isinstance(e.get("languages"), dict)]


def language_matches(code, wanted):
    """'en' matches 'en', 'en-US' and 'EN'; 'en-US' matches only 'en-US'."""
    code, wanted = code.lower(), wanted.lower()
    return code == wanted or code.startswith(wanted + "-")


def count_languages(entries):
    counts = {}
    for entry in entries:
        for code, block in entry["languages"].items():
            if isinstance(block, dict) and block.get("terms"):
                counts[code] = counts.get(code, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))


def pick_block(languages, wanted):
    """Return (code, block) of the first language in the entry that matches `wanted`."""
    for code, block in languages.items():
        if isinstance(block, dict) and block.get("terms") and language_matches(code, wanted):
            return code, block
    return None, None


# --------------------------------------------------------------------------- making records
def status_label(raw):
    """Short readable label for a TBX administrative status such as 'preferredTerm-admn-sts'."""
    if not raw:
        return None
    lowered = raw.lower()
    for label in ("preferred", "admitted", "deprecated", "superseded"):
        if label in lowered:
            return label
    return raw


def part_of_speech(block):
    """Part of speech of the first term, e.g. 'noun' or 'proper noun'. None if unknown."""
    for detail in block.get("term_details", []):
        for key, value in detail.get("attributes", {}).items():
            if key.lower() in ("partofspeech", "pos"):
                value = value if isinstance(value, str) else (value[0] if value else "")
                value = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value).lower().strip()
                return None if value in IGNORED_POS else value
        break
    return None


def format_terms(block, hide_preferred=False):
    """'term (preferred); other term (admitted)'. A status is shown only if the file has one.
    With hide_preferred=True the label 'preferred' is left out (used for the source terms)."""
    statuses = {d.get("term"): status_label(d.get("status")) for d in block.get("term_details", [])}
    parts = []
    for term in block["terms"]:
        label = statuses.get(term)
        if hide_preferred and label == "preferred":
            label = None
        parts.append("%s (%s)" % (term, label) if label else term)
    return "; ".join(parts)


def first_text(block, key):
    values = block.get(key) or []
    return values[0] if values else None


def build_record(entry, source, targets):
    """Return (record_text, embed_terms, definition) for one entry, or None to skip it."""
    languages = entry["languages"]
    source_code, source_block = pick_block(languages, source)
    if not source_block:
        return None
    target_blocks = []
    for wanted in targets:
        code, block = pick_block(languages, wanted)
        if block and code != source_code and all(code != c for c, _ in target_blocks):
            target_blocks.append((code, block))
    if not target_blocks:
        return None

    pos = part_of_speech(source_block)
    source_line = "%s: %s" % (source_code, format_terms(source_block, hide_preferred=True))
    if pos:
        source_line += " (%s)" % pos
    lines = [source_line]
    for code, block in target_blocks:
        lines.append("%s: %s" % (code, format_terms(block)))

    definition = first_text(source_block, "definitions")
    if not definition:
        for _, block in target_blocks:
            definition = first_text(block, "definitions")
            if definition:
                break
    if definition:
        lines.append("Definition: " + definition)
    for code, block in [(source_code, source_block)] + target_blocks:
        example = first_text(block, "examples")
        if example:
            lines.append("Example (%s): %s" % (code, example))
    return "\n".join(lines), "; ".join(source_block["terms"]), definition


def make_records(entries, source, targets, embed_mode, limit):
    """Build the record list. Returns (records, skipped, duplicates)."""
    records, seen, skipped, duplicates = [], set(), 0, 0
    for entry in entries:
        built = build_record(entry, source, targets)
        if not built:
            skipped += 1
            continue
        text, terms, definition = built
        if text in seen:
            duplicates += 1
            continue
        seen.add(text)
        embed_text = "%s: %s" % (terms, definition) if embed_mode == "term+definition" and definition else terms
        records.append((text, embed_text))
        if limit and len(records) >= limit:
            break
    return records, skipped, duplicates


# --------------------------------------------------------------------------- embeddings
def get_embeddings(url, texts):
    """Return one vector per text, or None if the server keeps failing."""
    payload = json.dumps({"input": texts}).encode()
    for attempt in range(1, RETRIES + 1):
        try:
            request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
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
def npy_header(n_rows, n_cols):
    """Header bytes of a .npy file (version 1.0) for a float32 matrix."""
    header = "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }" % (n_rows, n_cols)
    padding = (-(10 + len(header) + 1)) % 64  # magic(6) + version(2) + length(2) + header + newline
    header_bytes = (header + " " * padding + "\n").encode("latin1")
    return b"\x93NUMPY" + bytes([1, 0]) + struct.pack("<H", len(header_bytes)) + header_bytes


def write_json(path, data):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def finalize_npy(partial_path, out_path, n_rows, n_cols):
    """Turn the raw partial file into vectors.npy (header + the same bytes)."""
    temporary = out_path + ".tmp"
    with open(temporary, "wb") as out, open(partial_path, "rb") as raw:
        out.write(npy_header(n_rows, n_cols))
        while True:
            block = raw.read(1 << 20)
            if not block:
                break
            out.write(block)
    os.replace(temporary, out_path)


def signature_of(records):
    """Fingerprint of what is being embedded; a saved partial run is reused only if it matches."""
    digest = hashlib.sha1()
    for text, embed_text in records:
        digest.update(embed_text.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


# --------------------------------------------------------------------------- choosing files
def pick_with_dialogs():
    import tkinter
    from tkinter import filedialog

    root = tkinter.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    chosen_input = filedialog.askopenfilename(
        title="Select the terminology JSON file",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
    )
    chosen_output = filedialog.askdirectory(
        title="Select the output folder (vectors.npy and chunks.json will be saved here)",
        mustexist=False,
    ) if chosen_input else ""
    root.destroy()
    return chosen_input, chosen_output


def pick_with_prompts():
    chosen_input = input("Path to the terminology JSON file: ").strip().strip('"')
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


def choose_languages(counts, source, targets, interactive):
    """Return (source, targets), asking the user when running in a terminal."""
    codes = list(counts)
    if source is None:
        source = next((c for c in codes if c.lower().startswith("en")), codes[0])
    if not targets:
        targets = [c for c in codes if c != source]
    if interactive:
        print("Languages found:")
        for code, count in counts.items():
            print("   %-10s in %d entries" % (code, count))
        answer = input("Source language [%s]: " % source).strip()
        if answer:
            source = answer
        default_targets = ",".join(c for c in codes if not language_matches(c, source))
        answer = input("Target language(s), comma-separated [%s]: " % default_targets).strip()
        targets = [x.strip() for x in (answer or default_targets).split(",") if x.strip()]
    return source, targets


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(
        description="Build vectors.npy + chunks.json from a terminology JSON file.",
        epilog="Run without -i and -o to choose the files in dialog windows.",
    )
    parser.add_argument("-i", "--input", help="terminology JSON file (made by tbx_to_json.py)")
    parser.add_argument("-o", "--output", help="output folder")
    parser.add_argument("--source", help="source language code, e.g. en (default: asked, or English if present)")
    parser.add_argument("--target", help="target language code(s), comma-separated, e.g. tr")
    parser.add_argument("--embed", choices=["term", "term+definition"], default="term",
                        help="what to embed (default: %(default)s)")
    parser.add_argument("--url", default=DEFAULT_URL, help="embeddings endpoint (default: %(default)s)")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="texts per request (default: %(default)s)")
    parser.add_argument("--limit", type=int, default=0, help="only use the first N records (for a quick test)")
    parser.add_argument("--force", action="store_true", help="overwrite existing output files without asking")
    args = parser.parse_args()

    if args.batch < 1 or args.limit < 0:
        sys.exit("Invalid settings: --batch must be at least 1 and --limit cannot be negative.")

    input_path, output_dir = choose_locations(args.input, args.output)
    if not input_path or not output_dir:
        sys.exit("Input file and output folder are both required.")
    if not os.path.isfile(input_path):
        sys.exit("File not found: %s" % input_path)

    print("Reading %s ..." % input_path)
    try:
        entries = load_entries(input_path)
    except (ValueError, json.JSONDecodeError) as error:
        sys.exit("Cannot use this file: %s" % error)
    counts = count_languages(entries)
    if len(counts) < 2:
        sys.exit("The file needs at least two languages (a source and a target). Found: %s" % (", ".join(counts) or "none"))

    explicit_targets = [x.strip() for x in args.target.split(",") if x.strip()] if args.target else None
    interactive = sys.stdin.isatty() and not (args.source and explicit_targets)
    source, targets = choose_languages(counts, args.source, explicit_targets, interactive)
    for wanted in [source] + targets:
        if not any(language_matches(code, wanted) for code in counts):
            sys.exit("Language not found in the file: %s (available: %s)" % (wanted, ", ".join(counts)))
    if any(language_matches(source, t) or language_matches(t, source) for t in targets):
        sys.exit("The source language cannot also be a target language.")
    print("Source: %s  ->  Target: %s" % (source, ", ".join(targets)))

    records, skipped, duplicates = make_records(entries, source, targets, args.embed, args.limit)
    if not records:
        sys.exit("No usable records: no entry has both a source term and a target term.")
    print("%d records (%d skipped: missing source or target; %d exact duplicates removed)." % (len(records), skipped, duplicates))
    print("Example record:\n  " + records[0][0].replace("\n", "\n  "))

    final_paths = [os.path.join(output_dir, n) for n in ("vectors.npy", "chunks.json", "meta.json")]
    if any(os.path.exists(p) for p in final_paths) and not args.force:
        if sys.stdin.isatty():
            if input("Files already exist in %s. Overwrite? [y/N] " % output_dir).strip().lower() != "y":
                sys.exit("Stopped. Nothing was changed.")
        else:
            sys.exit("Files already exist in %s. Use --force to overwrite." % output_dir)

    print("Testing the embedding server at %s ..." % args.url)
    probe = get_embeddings(args.url, ["test"])
    if probe is None:
        sys.exit(
            "Cannot get embeddings. Is the server running, and was it started with an embeddings model\n"
            "(for KoboldCPP: --embeddingsmodel <file.gguf>)?"
        )
    dimension = len(probe[0])
    print("Server works. Vector size: %d." % dimension)

    os.makedirs(output_dir, exist_ok=True)
    partial_data = os.path.join(output_dir, PARTIAL_DATA)
    partial_meta = os.path.join(output_dir, PARTIAL_META)
    signature = signature_of(records)
    done = 0
    if os.path.exists(partial_data) and os.path.exists(partial_meta):
        try:
            with open(partial_meta, encoding="utf-8") as f:
                saved = json.load(f)
            if saved.get("signature") == signature and saved.get("vector_size") == dimension:
                rows = os.path.getsize(partial_data) // (4 * dimension)
                with open(partial_data, "r+b") as f:  # drop a half-written last row, if any
                    f.truncate(rows * 4 * dimension)
                done = min(rows, len(records))
                print("Found an unfinished run for this data: continuing from record %d." % (done + 1))
            else:
                print("An unfinished run for different data or another model was found: starting over.")
        except (OSError, ValueError):
            print("The saved progress could not be read: starting over.")
    if done == 0:
        with open(partial_data, "wb"):
            pass
        write_json(partial_meta, {"signature": signature, "vector_size": dimension})

    started, start_done = time.time(), done
    try:
        with open(partial_data, "ab") as raw:
            for start in range(done, len(records), args.batch):
                batch = records[start:start + args.batch]
                result = get_embeddings(args.url, [embed_text for _, embed_text in batch])
                if result is None:
                    sys.exit("\nThe server stopped answering. Progress is saved: start the server and run the same command again.")
                values = array("f")
                for vector in result:
                    if len(vector) != dimension:
                        sys.exit("\nThe server returned a vector of a different size. Was the embeddings model changed?")
                    values.extend(vector)
                if sys.byteorder == "big":
                    values.byteswap()  # keep the file little-endian
                values.tofile(raw)
                raw.flush()
                done = min(start + args.batch, len(records))
                speed = (done - start_done) / max(time.time() - started, 0.001)
                remaining = (len(records) - done) / speed if speed else 0
                print("Embedded %d / %d (%.1f%%)  %.0f per second, about %d min left   " % (
                    done, len(records), 100.0 * done / len(records), speed, remaining / 60), end="\r")
    except KeyboardInterrupt:
        sys.exit("\nInterrupted. Progress is saved: run the same command again to continue.")
    print()

    finalize_npy(partial_data, os.path.join(output_dir, "vectors.npy"), len(records), dimension)
    write_json(os.path.join(output_dir, "chunks.json"), [text for text, _ in records])
    write_json(os.path.join(output_dir, "meta.json"), {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_file": os.path.basename(input_path),
        "source_language": source,
        "target_languages": targets,
        "embedded_text": args.embed,
        "records": len(records),
        "skipped_entries": skipped,
        "duplicates_removed": duplicates,
        "vector_size": dimension,
        "embedding_url": args.url,
    })
    os.remove(partial_data)
    os.remove(partial_meta)
    print("Done. %d records saved to %s" % (len(records), os.path.abspath(output_dir)))


if __name__ == "__main__":
    main()
