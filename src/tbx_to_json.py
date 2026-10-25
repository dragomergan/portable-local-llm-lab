#!/usr/bin/env python3
"""Convert a TBX terminology file (.tbx / .xml) into JSON for use in RAG and other tools.

Standard library only. Works with TBX 2008 / MARTIF files (termEntry, langSet, tig,
ntig, termGrp) and with TBX 2019 / ISO 30042 files (conceptEntry, langSec, termSec),
including TBX-Core and TBX-Basic.

How it works
  1. You select a TBX file (a file dialog opens; or pass -i).
  2. The file is analyzed first: dialect, number of entries, languages, and which
     kinds of information it contains (definitions, examples, notes, part of speech,
     approval status, ...). The report is printed.
  3. You choose the name and location of the JSON file (a save dialog opens; or pass -o).
  4. The JSON is written with exactly the information found in your file:
     fields that do not exist in the TBX do not appear in the JSON.

The file can be monolingual, bilingual, or multilingual. Every language found in an
entry is kept, unless you restrict them (the program asks when there are more than
two languages, or use --languages).

Usage
  python tbx_to_json.py                                  # dialogs for input and output
  python tbx_to_json.py -i terms.tbx -o terms.json
  python tbx_to_json.py -i terms.tbx -o terms.json --languages en,tr
  python tbx_to_json.py -i terms.tbx -o terms.json --compact   # no per-term details
  python tbx_to_json.py -i terms.tbx --analyze-only            # only print the report

Output format: a JSON list, one entry (concept) per line:
  {
    "id": "c001",
    "subject_field": "information technology",         (only if present)
    "languages": {
      "en": {
        "terms": ["firewall"],
        "definitions": ["A security system that ..."],
        "examples": ["..."],                              (only if present)
        "notes": ["..."],                                 (only if present)
        "term_details": [                                 (omitted with --compact)
          {"term": "firewall", "id": "t1", "status": "preferredTerm-admn-sts",
           "attributes": {"partOfSpeech": "noun"}, "examples": ["..."]}
        ]
      },
      "tr": { ... }
    }
  }
Concept-level definitions (TBX 2019) are copied into each language that has none.
"""
import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter

XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"

ENTRY_TAGS = {"termEntry", "conceptEntry"}
LANG_TAGS = {"langSet", "langSec"}
TERM_BLOCK_TAGS = {"tig", "ntig", "termSec"}
EXAMPLE_TYPES = {"example", "context", "usageexample", "contextsentence"}
STATUS_TYPES = {"administrativestatus", "normativeauthorization", "approvalstatus", "termstatus", "status"}
SUBJECT_TYPES = {"subjectfield", "subject", "domain"}


# --------------------------------------------------------------------------- XML helpers
def local(tag):
    """Tag name without its namespace."""
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def text_of(element):
    """All text inside an element (including inline markup), with whitespace tidied."""
    return " ".join("".join(element.itertext()).split())


def lang_of(element):
    value = element.get(XML_LANG) or element.get("lang")
    if not value:
        for key, val in element.attrib.items():
            if key.endswith("lang"):
                return val
    return value


def add_unique(target, value):
    if value and value not in target:
        target.append(value)


def single_or_list(values):
    return values[0] if len(values) == 1 else values


# --------------------------------------------------------------------------- reading one entry
def read_meta(container):
    """Read definitions, examples, notes and other typed fields that sit directly
    under an entry or a language block (also inside descripGrp / adminGrp / noteGrp)."""
    meta = {"definitions": [], "examples": [], "notes": [], "other": {}}
    for child in container:
        name = local(child.tag)
        items = list(child) if name in ("descripGrp", "adminGrp", "noteGrp") else [child]
        if name in ("descripGrp", "adminGrp", "noteGrp"):
            items = [i for i in items if local(i.tag) in ("descrip", "admin", "note")]
        for item in items:
            kind, type_ = local(item.tag), item.get("type")
            text = text_of(item)
            if not text:
                continue
            if kind == "descrip":
                if (type_ or "").lower() == "definition":
                    add_unique(meta["definitions"], text)
                elif (type_ or "").lower() in EXAMPLE_TYPES:
                    add_unique(meta["examples"], text)
                else:
                    meta["other"].setdefault(type_ or "descrip", []).append(text)
            elif kind == "note":
                add_unique(meta["notes"], text)
            elif kind == "admin":
                meta["other"].setdefault(type_ or "admin", []).append(text)
    return meta


def read_term_block(block):
    """Read one term with its attributes. Returns None if the block has no term text."""
    term_element = next((e for e in block.iter() if local(e.tag) == "term"), None)
    if term_element is None:
        return None
    term = text_of(term_element)
    if not term:
        return None

    detail = {"term": term}
    term_id = term_element.get("id") or term_element.get(XML_ID)
    if term_id:
        detail["id"] = term_id

    attributes, examples, definitions, notes, history = {}, [], [], [], []
    for element in block.iter():
        kind, type_ = local(element.tag), element.get("type")
        if element is term_element:
            continue
        if kind in ("termNote", "admin"):
            text = text_of(element)
            if text:
                attributes.setdefault(type_ or kind, []).append(text)
        elif kind == "descrip":
            text = text_of(element)
            if not text:
                continue
            if (type_ or "").lower() == "definition":
                add_unique(definitions, text)
            elif (type_ or "").lower() in EXAMPLE_TYPES:
                add_unique(examples, text)
            else:
                attributes.setdefault(type_ or "descrip", []).append(text)
        elif kind == "note":
            add_unique(notes, text_of(element))
        elif kind == "transacGrp":
            record = {}
            for part in element:
                part_kind = local(part.tag)
                if part_kind == "transac":
                    record["action"] = text_of(part)
                elif part_kind == "transacNote":
                    record[part.get("type") or "note"] = text_of(part)
                elif part_kind == "date":
                    record["date"] = text_of(part)
            if record:
                history.append(record)

    for key, values in list(attributes.items()):
        attributes[key] = single_or_list(values)
    status_key = next((k for k in attributes if k.lower() in STATUS_TYPES), None)
    if status_key:
        value = attributes.pop(status_key)  # shown once, as "status"
        detail["status"] = value if isinstance(value, str) else value[0]
    if attributes:
        detail["attributes"] = attributes
    if definitions:
        detail["definitions"] = definitions
    if examples:
        detail["examples"] = examples
    if notes:
        detail["notes"] = notes
    if history:
        detail["history"] = history
    return detail


def read_language(language_element, compact):
    meta = read_meta(language_element)
    details = []
    for child in language_element:
        if local(child.tag) in TERM_BLOCK_TAGS:
            detail = read_term_block(child)
            if detail:
                details.append(detail)
    if not details:
        return None

    result = {"terms": [], "definitions": list(meta["definitions"])}
    examples, notes = list(meta["examples"]), list(meta["notes"])
    for detail in details:
        add_unique(result["terms"], detail["term"])
        for definition in detail.get("definitions", []):
            add_unique(result["definitions"], definition)
        for example in detail.get("examples", []):
            add_unique(examples, example)
        for note in detail.get("notes", []):
            add_unique(notes, note)
    if examples:
        result["examples"] = examples
    if notes:
        result["notes"] = notes
    if meta["other"]:
        result["attributes"] = {k: single_or_list(v) for k, v in meta["other"].items()}
    if not compact:
        result["term_details"] = details
    return result


def language_matches(code, wanted):
    """'en' matches 'en', 'en-US' and 'EN'; 'en-US' matches only 'en-US'."""
    code, wanted = code.lower(), wanted.lower()
    return code == wanted or code.startswith(wanted + "-")


def read_entry(entry_element, keep_languages=None, compact=False):
    """Convert one termEntry / conceptEntry into a dictionary (or None if empty)."""
    entry = {}
    entry_id = entry_element.get("id") or entry_element.get(XML_ID)
    if entry_id:
        entry["id"] = entry_id

    meta = read_meta(entry_element)
    other = dict(meta["other"])
    for key in list(other):
        if key.lower() in SUBJECT_TYPES:
            entry["subject_field"] = single_or_list(other.pop(key))
    if meta["definitions"]:
        entry["definitions"] = meta["definitions"]
    if meta["examples"]:
        entry["examples"] = meta["examples"]
    if meta["notes"]:
        entry["notes"] = meta["notes"]
    if other:
        entry["attributes"] = {k: single_or_list(v) for k, v in other.items()}

    languages = {}
    for child in entry_element:
        if local(child.tag) not in LANG_TAGS:
            continue
        code = lang_of(child) or "unknown"
        if keep_languages and not any(language_matches(code, w) for w in keep_languages):
            continue
        data = read_language(child, compact)
        if not data:
            continue
        if code in languages:  # the same language listed twice in one entry: merge
            for key in ("terms", "definitions", "examples", "notes"):
                for value in data.get(key, []):
                    add_unique(languages[code].setdefault(key, []), value)
            languages[code].setdefault("term_details", []).extend(data.get("term_details", []))
        else:
            languages[code] = data
    if not languages:
        return None

    if entry.get("definitions"):  # concept-level definition: give it to languages that have none
        for data in languages.values():
            if not data["definitions"]:
                data["definitions"] = list(entry["definitions"])
    entry["languages"] = languages
    return entry


def iter_entries(path):
    """Yield every entry element, freeing memory as it goes (works for very large files)."""
    for _, element in ET.iterparse(path, events=("end",)):
        if local(element.tag) in ENTRY_TAGS:
            yield element
            element.clear()


def read_header(path):
    """Return basic facts about the file from its header."""
    info = {"root": None, "dialect": None, "type": None, "language": None, "title": None}
    for event, element in ET.iterparse(path, events=("start", "end")):
        name = local(element.tag)
        if event == "start" and info["root"] is None:
            info["root"] = name
            info["type"] = element.get("type")
            info["dialect"] = element.get("style")
            info["language"] = lang_of(element)
        if event == "end" and name == "title" and not info["title"]:
            info["title"] = text_of(element)
        if event == "start" and name in ENTRY_TAGS:
            break
    return info


# --------------------------------------------------------------------------- analysis
def analyze(path):
    """First pass: read everything and collect statistics about the file."""
    stats = {
        "entries": 0, "empty": 0,
        "languages": Counter(),          # language -> number of terms
        "entries_with": Counter(),       # what kinds of data exist, counted per entry
        "attribute_values": {},          # attribute type -> Counter of values
        "terms_per_language_entry": Counter(),
        "language_sets": Counter(),      # which language combinations occur
    }
    for element in iter_entries(path):
        stats["entries"] += 1
        entry = read_entry(element)
        if not entry:
            stats["empty"] += 1
            continue
        stats["language_sets"][tuple(sorted(entry["languages"]))] += 1
        if "subject_field" in entry:
            stats["entries_with"]["subject field"] += 1
        for code, data in entry["languages"].items():
            stats["languages"][code] += len(data["terms"])
            stats["terms_per_language_entry"][len(data["terms"])] += 1
            if data["definitions"]:
                stats["entries_with"]["definitions"] += 1
            if data.get("examples"):
                stats["entries_with"]["examples"] += 1
            if data.get("notes"):
                stats["entries_with"]["notes"] += 1
            for detail in data["term_details"]:
                if "status" in detail:
                    stats["entries_with"]["approval/status"] += 1
                    stats["attribute_values"].setdefault("status", Counter())[detail["status"]] += 1
                if "history" in detail:
                    stats["entries_with"]["change history"] += 1
                for key, value in detail.get("attributes", {}).items():
                    values = value if isinstance(value, list) else [value]
                    counter = stats["attribute_values"].setdefault(key, Counter())
                    counter.update(values[:1])
    return stats


def print_report(path, header, stats):
    print("\n=== TBX analysis: %s ===" % os.path.basename(path))
    dialect = " / ".join(x for x in (header["root"], header["type"], header["dialect"]) if x)
    print("Format           :", dialect or "unknown")
    if header["title"]:
        print("Title            :", header["title"])
    if header["language"]:
        print("Header language  :", header["language"], "(usually the source language)")
    print("Entries          : %d (%d without usable terms)" % (stats["entries"], stats["empty"]))
    languages = stats["languages"]
    print("Languages        : %d" % len(languages))
    for code, count in languages.most_common():
        print("   %-10s %d terms" % (code, count))
    if len(languages) == 1:
        print("Type             : monolingual")
    elif len(languages) == 2:
        print("Type             : bilingual")
    else:
        print("Type             : multilingual (%d languages)" % len(languages))
    multi = sum(count for n, count in stats["terms_per_language_entry"].items() if n > 1)
    if multi:
        print("Synonyms         : %d language blocks have more than one term" % multi)
    print("Information found:")
    labels = ["definitions", "examples", "notes", "subject field", "approval/status", "change history"]
    for label in labels:
        count = stats["entries_with"].get(label, 0)
        print("   %-16s %s" % (label, ("found (%d)" % count) if count else "none"))
    for key, counter in sorted(stats["attribute_values"].items()):
        shown = ", ".join("%s (%d)" % (v, c) for v, c in counter.most_common(6))
        more = "" if len(counter) <= 6 else ", ... (%d values)" % len(counter)
        print("   term attribute %-22s %s%s" % (key, shown, more))
    print()


# --------------------------------------------------------------------------- choosing files
def ask_input_file():
    try:
        import tkinter
        from tkinter import filedialog

        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select a TBX file",
            filetypes=[("TBX files", "*.tbx *.xml"), ("All files", "*.*")],
        )
        root.destroy()
        return path
    except Exception:
        return input("Path of the TBX file: ").strip().strip('"')


def ask_output_file(default_name):
    try:
        import tkinter
        from tkinter import filedialog

        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.asksaveasfilename(
            title="Choose a name and location for the JSON file",
            defaultextension=".json",
            initialfile=default_name,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        root.destroy()
        return path
    except Exception:
        answer = input("Output JSON file (name and path) [%s]: " % default_name).strip().strip('"')
        return answer or default_name


def ask_languages(found):
    print("This file has more than two languages: " + ", ".join(found))
    answer = input("Languages to keep, separated by commas (press Enter to keep all): ").strip()
    return [x.strip() for x in answer.split(",") if x.strip()] or None


# --------------------------------------------------------------------------- conversion
def convert(path, out_path, keep_languages, compact, min_languages):
    temporary = out_path + ".tmp"
    written, total_terms = 0, 0
    with open(temporary, "w", encoding="utf-8") as f:
        f.write("[\n")
        for element in iter_entries(path):
            entry = read_entry(element, keep_languages, compact)
            if not entry or len(entry["languages"]) < min_languages:
                continue
            if written:
                f.write(",\n")
            f.write(json.dumps(entry, ensure_ascii=False))
            written += 1
            total_terms += sum(len(d["terms"]) for d in entry["languages"].values())
            if written % 5000 == 0:
                print("  %d entries written..." % written, end="\r")
        f.write("\n]\n")
    os.replace(temporary, out_path)
    return written, total_terms


def main():
    parser = argparse.ArgumentParser(
        description="Convert a TBX terminology file to JSON.",
        epilog="Run without -i and -o to choose the files in dialog windows.",
    )
    parser.add_argument("-i", "--input", help="the TBX file")
    parser.add_argument("-o", "--output", help="the JSON file to create")
    parser.add_argument("--languages", help="keep only these languages, e.g. en,tr (default: all)")
    parser.add_argument("--compact", action="store_true", help="leave out per-term details (part of speech, status, ...)")
    parser.add_argument("--min-languages", type=int, default=1, help="skip entries that have fewer languages than this (e.g. 2 for source+target only)")
    parser.add_argument("--analyze-only", action="store_true", help="print the analysis and stop")
    parser.add_argument("--force", action="store_true", help="overwrite the output file without asking")
    args = parser.parse_args()

    input_path = args.input or ask_input_file()
    if not input_path:
        sys.exit("No file selected.")
    if not os.path.isfile(input_path):
        sys.exit("File not found: %s" % input_path)

    print("Analyzing %s ..." % input_path)
    try:
        header = read_header(input_path)
        stats = analyze(input_path)
    except ET.ParseError as error:
        sys.exit("This is not a valid XML/TBX file: %s" % error)
    if not stats["languages"]:
        sys.exit("No terms were found. Is this really a TBX file?")
    print_report(input_path, header, stats)
    if args.analyze_only:
        return

    keep = [x.strip() for x in args.languages.split(",") if x.strip()] if args.languages else None
    if keep is None and len(stats["languages"]) > 2 and sys.stdin.isatty():
        keep = ask_languages(list(stats["languages"]))
    if keep:
        unknown = [w for w in keep if not any(language_matches(c, w) for c in stats["languages"])]
        if unknown:
            sys.exit("Language(s) not found in the file: %s" % ", ".join(unknown))

    default_name = os.path.splitext(os.path.basename(input_path))[0] + ".json"
    output_path = args.output or ask_output_file(default_name)
    if not output_path:
        sys.exit("No output file chosen.")
    if os.path.exists(output_path) and not args.force and args.output:
        if sys.stdin.isatty():
            if input("%s exists. Overwrite? [y/N] " % output_path).strip().lower() != "y":
                sys.exit("Stopped. Nothing was changed.")
        else:
            sys.exit("%s exists. Use --force to overwrite." % output_path)
    folder = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(folder, exist_ok=True)

    print("Converting...")
    written, total_terms = convert(input_path, output_path, keep, args.compact, args.min_languages)
    size_mb = os.path.getsize(output_path) / 1_000_000
    print("\nDone: %d entries, %d terms, %.1f MB -> %s" % (written, total_terms, size_mb, os.path.abspath(output_path)))


if __name__ == "__main__":
    main()
