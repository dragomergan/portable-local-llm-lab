---
name: term-extractor
description: Extract key terms, entities, keywords, or specialized vocabulary from a provided text. Use when the user asks to extract terms, glossary items, key concepts, or keywords from text.
---

# Term Extractor

Extract domain-specific terms, technical concepts, and proper nouns from a source passage for translation preparation.

## When to Use
- Extracting candidate glossary entries before translating a new text or domain.
- Building a bilingual or monolingual term list with context-grounded definitions.

## Steps
1. **Scan the Source Text:** Identify recurring domain terms, compound nouns, abbreviations, and named entities (skip everyday general vocabulary).
2. **Classify Each Term:** Note part of speech (`noun`, `verb`, `adj`) and category (`Domain Term`, `UI Label`, `Proper Noun / Do-Not-Translate`).
3. **Define from Context:** Write a concise 1-sentence definition grounded strictly in how the term is used in the passage (do not invent outside meanings).
4. **Format Output:** Present the result as a Markdown table (`Term | POS | Category | Contextual Definition`).
