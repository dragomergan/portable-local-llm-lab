# data/

Texts and databases for the exercises.

| Folder | What it holds |
| --- | --- |
| `anna_karenina/` | The ready-made database of the novel (`vectors.npy`, `chunks.json`, `meta.json`). The plain-text file itself is not stored: download it from [Project Gutenberg](https://www.gutenberg.org/ebooks/1399) if you want to build the database yourself with `src/create_rag_embeddings_novel.py`. |
| `anna_karenina_v2/` | A modified copy of the text where the character Stepan is renamed Sami, with its ready-made database. A model that knows the novel from its training will say "Stepan"; a RAG system that really reads the retrieved text will say "Sami". Use it to see whether answers come from retrieval or from the model's memory. |
| `Lore/` | "The Winter of Lanterns", a short story written for this kit, its `aliases.json`, an example `graph.json`, and an answer key for the test questions. |
| `terminology/` | A small sample glossary (`SampleGlossary.tbx` and `.json`) for the terminology exercises. Its database is not stored: build it with `src/create_rag_embeddings_glossary.py` (see the main README, section 8). |

Not included on purpose:

- **Novel text files, and other copyrighted or third-party texts.** Check the rules of your country before sharing them. Note that `chunks.json` contains the text of the book cut into pieces.
- **The Microsoft terminology files.** They are subject to Microsoft's license terms; download them yourself from the Microsoft Language Portal and do not redistribute them.
- **Your own databases** are yours to keep or share. The ready-made databases of the novels were built with EmbeddingGemma 300M (Q8_0): use the same embedding model with them, or rebuild them with the scripts in `src/`.
