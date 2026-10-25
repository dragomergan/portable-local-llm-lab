name: term-extractor
description: Extract key terms, entities, keywords, or specialized vocabulary from a provided text. Use when the user asks to extract terms, glossary items, key concepts, or keywords from text.

## Term Extractor

Extract key terms, domain-specific concepts, technical keywords, or named entities from a given text.

## When to Use
- When the user asks to extract terms, key concepts, keywords, or terminology from a passage or document.
- When building a glossary or index from source text.

## Steps for Term Extraction
1. **Analyze the Input Text**: Read through the text to identify recurring concepts, technical terms, proper nouns, and domain-specific jargon.
2. **Categorize the Terms** (if requested or helpful): Group extracted terms logically (e.g., Key Concepts, Technical Terms, Entities/Proper Nouns).
3. **Provide Definitions or Context**: Briefly define or explain each extracted term based on its context within the text.
4. **Format the Output**: Present the extracted terms clearly using markdown tables or bulleted lists for readability.

JSON Schema:
{'type': 'function', 'function': {'name': 'editor', 'description': 'An editor for controlled filesystem edits on the text file at the provided path. Provide `insert_line` to insert `new_text` at a specific line number. Otherwise, the tool replaces `old_text` with `new_text`, or creates the file with `new_text` if file does not exist. Use this tool for making small, precise edits to existing files or creating new files over shell commands. If several edits to different files or non-overlapping regions are already known, emit multiple editor tool calls in the same response instead of serializing them across turns.', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string', 'minLength': 1, 'description': 'The absolute path for the action to be performed on'}, 'old_text': {'description': 'Exact text to replace (must match exactly once). Omit this when creating a missing file or inserting via insert_line. Keep this at or below 6000 characters when possible; larger payloads should be split across multiple tool calls to avoid timeouts.', 'type': 'string'}, 'new_text': {'type': 'string', 'description': 'The new content to write when creating a missing file, the replacement text for edits, or the inserted text when insert_line is provided. Keep this at or below 6000 characters when possible; for large edits, use multiple calls with small chunks of old_text and new_text to iteratively edit the file.'}, 'insert_line': {'description': 'Optional positive one-based boundary line. When provided, the tool inserts new_text before that line instead of performing a replacement edit; use line_count + 1 to append at EOF.', 'anyOf': [{'type': 'integer', 'minimum': -9007199254740991, 'maximum': 9007199254740991}, {'type': 'null'}]}}, 'required': ['path', 'new_text'], 'additionalProperties': False, 'description': 'Edit a text file by replacing old_text with new_text, create the file with new_text if it does not exist, or insert new_text at insert_line when insert_line is provided. Prefer using this tool for file edits over shell commands. IMPORTANT: large edits can time out, so use small chunks and multiple calls when possible.'}}} 1, 1, 1} //Line 1, 1, 1