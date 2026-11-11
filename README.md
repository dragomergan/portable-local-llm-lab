# Portable Local LLM Lab

Run small open language models (Gemma and Qwen) **offline, from a USB stick or a plain folder**, on Windows, macOS or Linux. No installer, no account, no cloud, no `pip install`.

The kit was built for a university course in translation studies, but nothing in it is specific to that course. Use it to learn or teach:

- how to talk to a local model (prompts, system messages, chat in the browser)
- **RAG**: ask questions about a book, or look up terminology in a glossary
- **Graph RAG**: build a knowledge graph from a story and compare it with plain RAG
- how embeddings look (an interactive map of your vectors and graph)
- skills and agents in an editor (VS Code + Cline), and "vibe coding" small HTML apps

Everything runs on one small program ([KoboldCPP](https://github.com/LostRuins/koboldcpp)) that serves a chat model and an embedding model on `http://localhost:5001`.

> **Disclaimer:** Designed and architected for university translation technology labs using AI-assisted development; written purely against the Python standard library so every line remains inspectable for teaching and research.

> **Not included:** the model files, the KoboldCPP program, and any copyrighted text or glossary. You download them yourself (links below). See [Licenses](#licenses).

---

## Contents

1. [Quick start](#1-quick-start)
2. [What you need](#2-what-you-need)
3. [Folder layout](#3-folder-layout)
4. [Setup step by step](#4-setup-step-by-step)
5. [Start the models and check that they work](#5-start-the-models-and-check-that-they-work)
6. [Use the model](#6-use-the-model)
7. [RAG over a book](#7-rag-over-a-book)
8. [RAG over a glossary (terminology)](#8-rag-over-a-glossary-terminology)
9. [Graph RAG over a story](#9-graph-rag-over-a-story)
10. [See your vectors and graphs](#10-see-your-vectors-and-graphs)
11. [Skills, agents and vibe coding](#11-skills-agents-and-vibe-coding)
12. [Script reference](#12-script-reference)
13. [Troubleshooting](#13-troubleshooting)
14. [How it works, and its limits](#14-how-it-works-and-its-limits)
15. [Collaboration: join in](#15-collaboration-join-in)
16. [Licenses](#16-licenses)

---

## 1. Quick start

If you only want a chat with a local model:

1. Download KoboldCPP for your system from the [releases page](https://github.com/LostRuins/koboldcpp/releases) into `win/`, `mac/` or `linux/`.
2. Download [`gemma-3-4b-it-Q4_K_M.gguf`](https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF) into `models/`.
3. Run the start script for your system (`start_model.bat`, `start_model.command` or `start_model.sh`).
4. Open <http://localhost:5001> in your browser.

The rest of this file covers the full kit (RAG, graphs, tools).

---

## 2. What you need

| | Minimum | Comfortable |
| --- | --- | --- |
| RAM | 8 GB | 16 GB |
| Disk | about 6 GB for the three models | 8 GB or more |
| CPU | any 64-bit CPU from the last several years | 8 cores or more |
| GPU | not needed | not needed |
| Python | optional: only for the scripts in `src/` and `Test/` (Python 3.8 or newer) | |

Notes:

- The scripts use only the Python standard library. If `numpy` happens to be installed, searches are faster; nothing depends on it.
- Run **one chat model at a time**. A 4B model plus the small embedding model fits in 8 GB but not much else; close other heavy programs.
- Speed depends on your CPU. A 4B model on a CPU answers slowly (a few words per second is normal on a laptop). Be patient with long answers and keep questions short.
- For a USB stick, format it **exFAT** so that it works on all three systems. Use a USB 3 stick: the models are large and load faster from a fast stick.

---

## 3. Folder layout

Create this structure (a USB stick or any folder; the name `LLM_LAB` is just an example):

```text
LLM_LAB/
├── models/      the .gguf model files (shared by all systems)
├── data/        your texts, and the databases built from them
│   ├── anna_karenina/       the novel + its database (vectors.npy, chunks.json, meta.json)
│   ├── anna_karenina_v2/    a modified copy of the novel (Stepan renamed Sami) + its database
│   ├── Lore/                lore.txt, aliases.json, graph.json
│   └── terminology/         sample glossary (SampleGlossary.tbx and .json); you build its database
├── skills/      instruction files (.md) for the skill exercises
├── src/         scripts that BUILD things (embeddings, graph)
├── Test/        scripts that TEST things (server check, book, glossary and lore tests)
├── work/        pages you USE (chat page, map page)
├── win/         koboldcpp.exe + start_model.bat
├── mac/         koboldcpp-mac-arm64 + start_model.command
└── linux/       koboldcpp-linux-x64 + start_model.sh
```

Where each file goes:

| Folder | Files |
| --- | --- |
| `src/` | `create_rag_embeddings_novel.py`, `create_rag_embeddings_glossary.py`, `tbx_to_json.py`, `graph_build.py` |
| `Test/` | `check_server.py`, `rag_common.py`, `test_anna.py`, `test_glossary.py`, `test_lore.py` |
| `work/` | `rag_chat.html`, `vector_map.html` |
| `win/`, `mac/`, `linux/` | the start script for that system + the KoboldCPP program you download |

`rag_common.py` must stay in the same folder as the three `test_*.py` scripts (`Test/`).

---

## 4. Setup step by step

### 4.1 Download KoboldCPP

Open <https://github.com/LostRuins/koboldcpp/releases> and download the file for your system from the latest release's **Assets**:

| System | File | Put it in |
| --- | --- | --- |
| Windows | `koboldcpp.exe`, or the smaller `koboldcpp_nocuda.exe` (CPU only) | `win/` |
| macOS (Apple Silicon) | `koboldcpp-mac-arm64` | `mac/` |
| Linux | `koboldcpp-linux-x64` (a `nocuda` or `oldpc` build if listed) | `linux/` |

Asset names change between releases. If the exact name is missing, pick the file that matches your system and does not need CUDA. If you downloaded `koboldcpp_nocuda.exe`, either rename it to `koboldcpp.exe` or edit the `KOBOLD=` line at the top of `start_model.bat`.

Intel Macs: the Apple Silicon build does not run on them; see the KoboldCPP releases page for what is currently offered.

### 4.2 Download the models

Download each file into `models/`. A model page may ask you to log in and accept a license first.

| Model | Repository | File | Size |
| --- | --- | --- | --- |
| Gemma 3 4B Instruct (chat) | <https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF> | `gemma-3-4b-it-Q4_K_M.gguf` | about 2.5 GB |
| Qwen3 4B (chat) | <https://huggingface.co/Qwen/Qwen3-4B-GGUF> | `Qwen3-4B-Q4_K_M.gguf` | about 2.5 GB |
| EmbeddingGemma 300M (embeddings, needed for RAG) | <https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF> | `embeddinggemma-300M-Q8_0.gguf` | about 0.3 GB |

Do **not** download the `mmproj` or `f16` files in the Gemma repository (image input, not needed).

**Check the downloads.** An interrupted download is the most common problem. Compare the file size with the repository page.

You can use any other `.gguf` chat model too. The start script lists every `.gguf` file in `models/`, except files with `embed` or `mmproj` in the name.

> **Note:** the models above are deliberately small, so that they run on ordinary computers (a lab PC or a laptop without a GPU). Larger models that support the text modality can be used in the same way if your hardware allows it: put the `.gguf` file in `models/` and pick it in the start menu. Larger models are generally more accurate, but they need more RAM and are slower on a CPU. A model must be supported by your KoboldCPP version, and very large models (hundreds of GB) are out of reach for normal computers. Models that need extra files, such as image input (`mmproj`), are not set up by these scripts. The embedding model must stay the same as the one that built your databases.

### 4.3 First start on each system

- **Windows:** double-click `start_model.bat`. If SmartScreen warns about an unknown program, choose *More info → Run anyway* (only if you downloaded it from the official releases page).
- **macOS:** open Terminal and run `chmod +x mac/koboldcpp-mac-arm64 mac/start_model.command` once, then double-click `start_model.command`. MacOS blocks koboldcpp-mac-arm64, open *System Settings → Privacy & Security* and allow it, or right-click the file and choose *Open*. Then rerun the command to run the localhost. 
- **Linux:** `chmod +x linux/koboldcpp-linux-x64 linux/start_model.sh`, then run `./start_model.sh`.

**exFAT sticks lose the "executable" permission** on macOS and Linux. If the script says *permission denied*, run it with `bash start_model.sh`. If the KoboldCPP program itself cannot start, copy the whole `LLM_LAB` folder to your computer's disk and run `chmod +x` there.

---

## 5. Start the models and check that they work

### 5.1 Start

Run the start script for your system. It shows a numbered list of the chat models it finds:

```text
Available models:
  1) Qwen3-4B-Q4_K_M.gguf
  2) gemma-3-4b-it-Q4_K_M.gguf

Type the number of the model and press Enter:
```

Then it starts KoboldCPP with your chosen chat model **and** the embedding model. Wait until the window says the server is ready (the first load takes a while).

Shortcuts and settings:

- Skip the menu: `./start_model.sh Qwen3-4B-Q4_K_M.gguf` (Windows: `start_model.bat Qwen3-4B-Q4_K_M.gguf`).
- If only one chat model is in `models/`, it starts without asking.
- Port, threads and context size are variables at the top of each script (`5001`, `8`, `8192`). Lower `THREADS` on a weak machine.
- To switch models, close the window (Ctrl+C) and start the script again.

### 5.2 Check

Open <http://localhost:5001> to see KoboldCPP's own chat page. Then, from the `Test/` folder:

```text
python check_server.py
```

It tests the server, the chat model and the embedding model, and prints OK or FAIL for each. Everything in the next sections needs these three to be OK.

(On macOS and Linux the command is usually `python3`.)

---

## 6. Use the model

**In the browser.** KoboldCPP's page at <http://localhost:5001> is a normal chat. Try the same question with Gemma and with Qwen; they behave differently.

**Qwen3 "thinking".** Qwen3 may print a long reasoning block (`<think>...</think>`) before the answer. Add `/no_think` at the end of your message to switch it off. The scripts here hide the block automatically, and the test scripts have a `--no-think` option.

**From your own code.** KoboldCPP offers an OpenAI-compatible API:

```python
import json, urllib.request

req = urllib.request.Request(
    "http://localhost:5001/v1/chat/completions",
    data=json.dumps({
        "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
        "max_tokens": 60,
    }).encode(),
    headers={"Content-Type": "application/json"},
)
print(json.load(urllib.request.urlopen(req))["choices"][0]["message"]["content"])
```

Embeddings: `POST /v1/embeddings` with `{"input": "some text"}` (the embedding model must be loaded, which the start script does).

With `curl` on Windows PowerShell, type `curl.exe`: plain `curl` is an alias of another command there.

---

## 7. RAG over a book

RAG (retrieval-augmented generation) means: split a text into chunks, turn each chunk into a vector (an embedding), and for each question find the chunks whose vectors are closest to the question, then give only those chunks to the model.

### 7.1 Get a text

Any plain `.txt` file works. A good example is *Anna Karenina* from Project Gutenberg: <https://www.gutenberg.org/ebooks/1399> (choose "Plain Text UTF-8"). Put it in `data/anna_karenina/`. Check that the text is public domain in your country before sharing it.

### 7.2 Build the database

Ready-made databases for the examples are already in `data/`, so you can skip this step and go straight to 7.3. To build your own, start the server and run:

```text
cd src
python create_rag_embeddings_novel.py
```

A window (or the terminal) asks for the input text file or folder and the output folder. Or give them directly:

```text
python create_rag_embeddings_novel.py -i ../data/anna_karenina/anna_karenina.txt -o ../data/anna_karenina
```

It writes three files: `vectors.npy` (the numbers), `chunks.json` (the text pieces, same order) and `meta.json`.

| Option | Meaning |
| --- | --- |
| `-i`, `-o` | input file or folder of `.txt` files, and output folder |
| `--chunk-words 300` | words per chunk |
| `--overlap 50` | words shared between neighbouring chunks |
| `--batch 16` | texts per request to the server |
| `--keep-gutenberg` | do not strip the Project Gutenberg header and footer |
| `--force` | overwrite an existing database |

### 7.3 Ask questions

**Command line**, from `Test/`:

```text
python test_anna.py --data ../data/anna_karenina
```

Type any question. For every question you see the retrieved chunks (with similarity scores) and the model's answer. Useful options:

- `--top-k 6`: give the model more chunks (default 4)
- `--compare`: answer each question twice, with 1 chunk and with top-k chunks
- `--no-think`: for Qwen3

**Browser:** open `work/rag_chat.html` (double-click it), click **Load files**, and select `vectors.npy` and `chunks.json` together (Ctrl-click). Ask a question; open **Sources used** under the answer to see exactly what the model was given. Untick **Use RAG** to compare with the model's own memory.

### 7.4 What to expect (and try)

Plain RAG is good at *finding a passage* and weak at *counting or collecting facts spread across the book*. Try:

- "What happens when Levin goes to the mowing fields?" (one scene: usually works)
- "How many characters are called Anna?" or "What are the names of Anna's children?" (needs many chunks: often fails or invents)
- A question the book cannot answer (a good system says it cannot tell)

Then change `--top-k`, the chunk size, or the prompt, and compare. That comparison is the point of the exercise. Small models can hallucinate even with good chunks.

---

## 8. RAG over a glossary (terminology)

A terminology database is different from a book: each record is one *concept* (a source term, its target terms with status, a definition). One word can have several meanings ("disabled"), so every record is kept and the model is shown all of them.

### 8.1 Get a glossary in TBX format

TBX (TermBase eXchange) is the standard terminology file format. This repo handles TBX 2008 (MARTIF) and TBX 2019.

- A small sample is included: `data/terminology/SampleGlossary.tbx` and its JSON `SampleGlossary.json` (original English→Turkish entries, CC0).
- The large Microsoft terminology files are **not** included. If you want them, download them from the [Microsoft Language Portal](https://www.microsoft.com/en-us/language/terminology) and read Microsoft's license terms first. Do not redistribute them.

### 8.2 Convert TBX to JSON

```text
cd src
python tbx_to_json.py -i ../data/terminology/SampleGlossary.tbx -o ../data/terminology/SampleGlossary.json
```

Without `-i`/`-o` it opens file dialogs. The script analyses the file first and reports what it found (languages, definitions, examples, term status, notes), then writes JSON that keeps all of it.

| Option | Meaning |
| --- | --- |
| `--languages en,tr` | keep only these languages |
| `--min-languages 2` | skip entries with fewer languages |
| `--analyze-only` | print the report, write nothing |
| `--compact` | smaller JSON (no indentation) |
| `--force` | overwrite an existing file |

### 8.3 Build the glossary database

The glossary database (`vectors.npy`, `chunks.json`, `meta.json`) is **not** included in the repository. You build it once from the JSON file, with the server running (it needs the embedding model). The test script in 8.4 needs it.

```text
python create_rag_embeddings_glossary.py -i ../data/terminology/SampleGlossary.json -o ../data/terminology --source en --target tr
```

Each record looks like this:

```text
en: disabled (adjective)
tr: devre dışı (preferred); gri (admitted)
Definition: ...
```

By default only the source term is embedded (so a short query like "disabled" matches well). Use `--embed term+definition` to embed the definition too. Large glossaries take a while; the script can **resume** if you stop it (run it again with the same output folder).

| Option | Meaning |
| --- | --- |
| `--source en`, `--target tr` | languages (asked if you leave them out) |
| `--embed term` or `term+definition` | what gets embedded |
| `--batch 32` | texts per request |
| `--limit 500` | only the first N records (quick test) |
| `--force` | start over |

### 8.4 Test it

From `Test/`:

```text
python test_glossary.py --data ../data/terminology
```

Menu:

1. **Look up a term:** exact match first (shows every meaning), then the nearest records by meaning.
2. **Translate a sentence with the glossary:** finds the glossary terms in your sentence, sends them to the model, and asks it to use them (preferred term first, never a deprecated one).
3. **Show a definition.**

Exact matching is plain word matching: it does not know plurals ("button" matches, "buttons" does not).

---

## 9. Graph RAG over a story

Plain RAG retrieves passages. **Graph RAG** first extracts facts as (subject, relation, object) triples with the model, merges the names of the same entity, and then answers from the facts around the entities in your question. It is better for questions about relations and for counting, and worse when the extraction is wrong.

### 9.1 The example text

`data/Lore/lore.txt` is "The Winter of Lanterns", a short story written for this kit (about 600 words, no real-world source, so the model cannot answer from memory). `Test/lore_answer_key.md` lists the entities, the expected facts, and ten test questions with their correct answers. Some characters have nicknames that share no word with their name (the Pale Queen, the Iron Chancellor): an automatic merge cannot catch those, and that is part of the exercise.

### 9.2 Build the graph

With the chat model running (Qwen3: add `--no-think`):

```text
cd src
python graph_build.py ../data/Lore/lore.txt ../data/Lore/graph.json --aliases none
```

The script prints every edge it found so you can check them by eye. Then write your own `aliases.json` (canonical name → other names) and build again:

```text
python graph_build.py ../data/Lore/lore.txt ../data/Lore/graph.json --aliases ../data/Lore/aliases.json
```

An alias file looks like this:

```json
{
  "Isolde Varn": ["Pale Queen", "Queen Isolde"],
  "Maelric Voss": ["Iron Chancellor", "Voss"]
}
```

Build the plain vector database of the same text too (section 7.2), into `data/Lore/`, so you can compare.

### 9.3 Compare plain RAG and graph RAG

From `Test/`:

```text
python test_lore.py --data ../data/Lore
```

Modes: `plain` (vector search), `graph` (facts only), `graph+text` (facts plus the source passages). Type your own question, or a number for one of the ten class questions. To run all ten questions in every mode and save a report to mark by hand:

```text
python test_lore.py --data ../data/Lore --run-key --report lore_report.md
```

That is 30 model calls, so it takes a while on a CPU. Use `-v` to print the chunks or facts that were sent.

In `rag_chat.html`: **Load files**, select `graph.json` (and `aliases.json`, and the vectors + chunks if you want both). Choose the graph in the **Database** list, or **All** to combine vectors and graph. New controls appear: **Hops** (how far to walk from the entities), **Passages** (source passages added to the facts) and **Whole graph** (send every fact, good for counting questions).

---

## 10. See your vectors and graphs

Open `work/vector_map.html`, click **Load files**, and select what you have. The page recognises each file by its content, in any order:

| You load | You get |
| --- | --- |
| `vectors.npy` + `chunks.json` (or one `index.json`) | **Vectors** tab: every chunk is a point, squeezed to 2D with PCA |
| `graph.json` (+ `aliases.json`) | **Graph** tab: entities as circles, facts as arrows |

**Vectors tab.** Hover a point to read its text. Colors show the position in the text. Type a question or any phrase and press **Place on map** (needs the server): it appears as a star with lines to its nearest chunks. Points that sit close together have similar meaning. The map is a rough 2D picture of hundreds of dimensions; the real ranking uses all of them.

**Graph tab.** Drag a circle to pull the graph apart. Hover to see an entity's facts and the passages they came from. Type a question and press **Highlight in graph**: the entities it mentions and the facts around them light up. This works without the server, and shows what graph RAG would send to the model.

Mouse: scroll to zoom, drag the background to move, double-click to reset the view.

Works for any database of this kind: a book, a glossary (tens of thousands of records are fine), or a graph.

---

## 11. Skills, agents and vibe coding

These exercises use an editor with an AI extension pointed at your local model.

1. Install [VS Code](https://code.visualstudio.com/) and the [Cline](https://cline.bot/) extension.
2. In Cline's settings choose an **OpenAI-compatible** provider and set the base URL to `http://localhost:5001/v1`. The API key can be any text. Menu names change between versions; check Cline's documentation if yours differ.
3. **Skills:** a skill is a folder with a `SKILL.md` file. The folder name must equal the `name:` in the file. Cline's own skill folder is `.cline/skills/<skill-name>/SKILL.md`; you invoke a skill by typing `/skill-name`. Example skill ideas: term extraction from a text, a post-editing checklist. Put your own in `skills/` and copy them where your editor expects them.
4. **Agents:** chain two models. For post-editing, one model translates and the other reviews the translation; run one model at a time and switch with the start script, or use the browser chat for each step.
5. **Vibe coding:** ask the model to write a single self-contained HTML page (for example a small tool that calls `http://localhost:5001`), save it, and open it in the browser.

Small 4B models make plenty of mistakes in code and in agent loops. Keep tasks small, check the output, and give precise instructions.

---

## 12. Script reference

Run scripts from the folder they live in (the default `../data/...` paths are relative to where you run them).

| Script | Where | What it does |
| --- | --- | --- |
| `start_model.bat` / `.command` / `.sh` | `win/` `mac/` `linux/` | choose a chat model and start KoboldCPP with the embedding model |
| `create_rag_embeddings_novel.py` | `src/` | text files → `vectors.npy` + `chunks.json` (book RAG) |
| `tbx_to_json.py` | `src/` | TBX terminology file → analysed JSON |
| `create_rag_embeddings_glossary.py` | `src/` | terminology JSON → glossary database (resumable) |
| `graph_build.py` | `src/` | text → `graph.json` (entities and facts, with alias merging) |
| `check_server.py` | `Test/` | tests the server, the chat model and the embeddings |
| `test_anna.py` | `Test/` | ask any question about a book database; compare top-1 and top-k |
| `test_glossary.py` | `Test/` | look up terms, translate with the glossary, show definitions |
| `test_lore.py` | `Test/` | plain RAG vs graph RAG, with the ten class questions |
| `rag_common.py` | `Test/` | helpers shared by the test scripts |
| `rag_chat.html` | `work/` | chat page: vector databases and graphs, shows sources |
| `vector_map.html` | `work/` | map of vectors (PCA) and of graphs |

All scripts print `--help`.

---

## 13. Troubleshooting

| Problem | What to try |
| --- | --- |
| The model does not load, or the program closes | The file may be incomplete: compare its size with the repository page and download it again. Close other heavy programs. |
| `check_server.py` says FAIL for the server | KoboldCPP is not running, or it uses another port. Start it with the start script and wait until it is ready. |
| Embeddings FAIL, chat OK | The embedding model is missing from `models/` (its name must contain `embed`), or you started KoboldCPP another way. Use the start script. |
| "The database has N dimensions but the embedding model gives M" | The database was built with a different embedding model. Use the same one for building and asking. |
| "vectors.npy has N rows but chunks.json has M entries" | The two files come from different runs. Build them again together. |
| `permission denied` (macOS/Linux) | exFAT drives lose the executable flag. Run `chmod +x`, or `bash start_model.sh`, or copy the folder to your disk. |
| macOS or Windows blocks the program | Allow it in system settings (see 4.3) if you got it from the official releases page. |
| Browser page cannot reach the server | Check that the **Server** field in the page says `http://localhost:5001` and that KoboldCPP is running. |
| Answers are very slow | Normal on a CPU. Shorter questions, a smaller `--top-k`, and `CONTEXT=4096` in the start script help. |
| Qwen3 prints a long `<think>` block | Add `/no_think` to your message, or use the `--no-think` option of the scripts. |
| The model invents answers | Small models do this. Open the sources shown under the answer and check them, try a different `--top-k`, and tell the model to say when it cannot tell. Graph RAG helps with relation and counting questions, not with everything. |
| Python not found (Windows) | Install Python 3 from <https://www.python.org/downloads/> and tick "Add python.exe to PATH". |

---

## 14. How it works, and its limits

```text
                         KoboldCPP (localhost:5001)
 your text ──chunks──▶  embedding model ──▶ vectors.npy + chunks.json
 your question ───────▶ embedding model ──▶ nearest chunks ──┐
                                                             ▼
                          chat model  ◀── question + chunks (+ graph facts)
                               │
                               ▼
                            answer
```

- One server, two models: the chat model you pick, and EmbeddingGemma for vectors. Everything stays on your machine.
- Databases are plain files (`vectors.npy`, `chunks.json`, `graph.json`), so you can open, inspect, version and share them.
- Search is a full comparison of the question vector with every chunk vector. That is fine for tens of thousands of records and simple to understand; it is not a production vector database.
- Models are small (4B parameters, 4-bit). Expect mistakes. The kit is for learning how these systems behave, not for trusting their answers blindly.
- The chat and map pages and the Python scripts were tested with a mock server and sample data. Test with your own models, and please report what breaks.

---

## 15. Collaboration: join in

This kit grew out of university teaching, and it is meant to be improved by other teachers, researchers and students. It will be refined after its first classroom use, and **you are warmly invited to help**: academics in translation studies, linguistics, computational linguistics, digital humanities, education and computer science, and anyone who teaches or learns with local models.

Ways to take part:

- **Use it in your own course** and tell us what worked and what did not. Classroom reports are the most valuable feedback.
- **Test on your hardware** (older PCs, Intel Macs, Linux distributions, machines with a GPU) and report speeds and problems.
- **Try more models**: other Gemma, Qwen and open models, translation-specialised models, larger models on stronger machines. Share which ones work well for which task.
- **Add languages and materials**: texts and exercises in other languages, other freely licensed glossaries and TBX files, other short stories for graph RAG.
- **Improve the methods**: better chunking, hybrid keyword + vector search, reranking, better alias merging for graphs, more robust fact extraction with small models.
- **Evaluate**: test sets and scoring for terminology use, translation and post-editing quality, and RAG answers, so results can be compared across models.
- **Improve the tools**: bug fixes, tests, cleaner code, accessibility and translations of this README.
- **Write about it**: papers, teaching notes and worksheets built on the kit.

How to start: open an **Issue** on GitHub to ask a question, report a problem or propose an idea, or send a **Pull Request** with a change. Small, focused changes are easiest to review. Please do not upload copyrighted texts, glossaries or model files; link to them instead. If you use the kit in teaching or research, a mention or a link back is appreciated.

---

## 16. Licenses

- **This repository's code and the example texts it contains:** [License](https://github.com/dragomergan/portable-local-llm-lab/blob/main/LICENSE). The story `lore.txt` and `SampleGlossary.tbx` were written for this kit.
- **KoboldCPP:** AGPL-3.0, see <https://github.com/LostRuins/koboldcpp>.
- **Gemma models (Gemma 3 4B, EmbeddingGemma):** [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
- **Qwen3 4B:** Apache 2.0, see the [model page](https://huggingface.co/Qwen/Qwen3-4B-GGUF).
- **Project Gutenberg texts:** public domain in the United States; check your country's rules and Gutenberg's terms before redistributing.
- **Microsoft terminology:** not included; subject to Microsoft's license.

Model and tool licenses can change; always check the original pages.
