# Briefcase

Briefcase is a local, source-grounded notebook assistant in the spirit of NotebookLM.
You create notebooks, add sources (files, pasted text, or URLs), and ask questions.
Every answer is grounded in, and cited back to, the exact passages in your own documents,
and Briefcase refuses to answer when the sources do not support a claim.

It is built on the retrieval and grounding ideas from
[Project Pragya](https://github.com/Hrishikesh2512/pragya), re-architected to run on a
laptop with **no mandatory cloud dependencies** and a clean web UI.

## Features

- **Notebooks**: group sources into isolated workspaces.
- **Reads all the files**: PDF, Word (.docx), PowerPoint (.pptx), Markdown, HTML,
  plain text, CSV/TSV/JSON, and source code (.py, .js, .ts, .go, .rs, .java, and more).
  Also add sources by URL or by pasting text.
- **OCR for scans**: scanned / image-only PDFs and image files (.png, .jpg, ...) are
  read with OCR when the optional engine is installed (auto-detected).
- **In-app Settings**: add / view (masked) / remove API keys, configure a local
  (Ollama) or online (Gemini) model, and pick which one is the current engine, all
  from a Settings panel, no env editing or restart required.
- **Grounded, cited answers**: hybrid retrieval (BM25 + dense vectors, fused with
  Reciprocal Rank Fusion) with inline `[n]` citations you can click to see the exact quote.
- **Refuses when unsupported**: no evidence, no answer.
- **Studio**: one-click grounded briefing and suggested questions per notebook.
- **Runs anywhere, offline**: default embeddings need no downloads and no API key.
- **Pluggable brains**: auto-detects Google Gemini (if `GEMINI_API_KEY` is set) or a local
  Ollama server for synthesized answers; otherwise falls back to a fully offline
  extractive engine that still returns cited passages.

## Quickstart

```bash
cd briefcase
./run.sh
```

Then open http://127.0.0.1:8000 in your browser. That is it.

`run.sh` creates a virtual environment and installs dependencies on first run.
To do it manually:

```bash
python3 -m venv .venv
./.venv/bin/pip install -e .
./.venv/bin/python -m briefcase.cli --reload
```

## Answer engines

Briefcase works with zero configuration in **extractive mode**: it retrieves the most
relevant passages and returns them verbatim with citations. For synthesized,
conversational answers, give it a generative model, it is auto-detected:

| Engine | How to enable |
| --- | --- |
| **Gemini** (cloud) | `export GEMINI_API_KEY=...` (get one at https://aistudio.google.com/apikey) |
| **Ollama** (local) | Install [Ollama](https://ollama.com), run `ollama pull llama3.2` |
| **Extractive** (offline) | Default when neither of the above is available |

The active engine is shown as a badge in the top-right of the app.

## Better retrieval (optional)

The default embedder is a dependency-free hashing embedder that pairs with BM25 for
solid recall. For stronger semantic search, install sentence-transformers:

```bash
./.venv/bin/pip install -e ".[semantic]"
export BRIEFCASE_EMBEDDINGS=sentence-transformers
```

## Configuration

All settings are environment variables (see `.env.example`). Copy it to `.env` and the
launcher will load it. Common ones:

- `BRIEFCASE_LLM` = `auto` | `gemini` | `ollama` | `extractive`
- `BRIEFCASE_EMBEDDINGS` = `hashing` | `sentence-transformers` | `gemini`
- `BRIEFCASE_DATA_DIR` = where the SQLite database and uploads live (default `./data`)

## Architecture

```
Browser SPA  (web/)  ─ 3-panel UI: Sources | Chat | Studio
      │  REST/JSON
FastAPI  (briefcase/app.py, api/routes.py)
      │
      ├─ ingest/     parsers (all file types) -> chunking -> embeddings
      ├─ rag/        hybrid retriever (BM25 + vectors + RRF)
      │              -> generator (LLM or extractive) -> citations + refusal
      │              -> studio (briefing, suggested questions)
      ├─ repository  SQLite (notebooks, sources, chunks, vectors)
      └─ config      env-driven settings
```

Everything persists to a single SQLite file under `data/`, so there is no external
database, queue, or vector service to run.

## How it compares to Pragya

Pragya is an enterprise, multi-tenant RAG platform (Postgres, Redis, ChromaDB, JWT auth,
Prometheus). Briefcase keeps Pragya's core retrieval and grounding approach (hybrid search,
RRF fusion, cited answers, evidence-based refusal) but trades the heavy infrastructure for
a single-file, single-user, batteries-included app focused on the NotebookLM experience:
notebooks, broad file ingestion, a polished UI, and offline-first defaults.

## Project layout

```
briefcase/
  app.py              FastAPI app + static hosting
  cli.py              `briefcase` launcher
  config.py           env-driven settings
  db.py               SQLite schema + connection
  models.py           pydantic models
  repository.py       notebook / source / chunk persistence
  text_utils.py       tokenization, stopwords, helpers
  ingest/
    parsers.py        PDF, DOCX, PPTX, Markdown, HTML, text, code, URL
    chunking.py       sentence-aware chunking with overlap
    service.py        ingestion pipeline (parse -> chunk -> embed -> store)
  rag/
    embeddings.py     hashing / sentence-transformers / gemini embedders
    retriever.py      BM25 + dense vectors fused with RRF
    llm.py            gemini / ollama / none providers (auto-detected)
    generator.py      grounded answering with citations + refusal
    studio.py         briefing + suggested questions
    prompts.py        system + task prompts
  api/routes.py       REST endpoints
web/
  index.html, styles.css, app.js   the single-page UI
```

## License

MIT.
