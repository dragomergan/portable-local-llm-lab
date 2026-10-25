# models/

Put the `.gguf` model files here. They are not stored in the repository (they are large and have their own licenses).

| Model | Download | File |
| --- | --- | --- |
| Gemma 3 4B Instruct (chat) | <https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF> | `gemma-3-4b-it-Q4_K_M.gguf` |
| Qwen3 4B (chat) | <https://huggingface.co/Qwen/Qwen3-4B-GGUF> | `Qwen3-4B-Q4_K_M.gguf` |
| EmbeddingGemma 300M (embeddings, needed for RAG) | <https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF> | `embeddinggemma-300M-Q8_0.gguf` |

Any other `.gguf` chat model also works: the start script lists every `.gguf` here, except files with `embed` or `mmproj` in the name. See the main README for details and licenses.
