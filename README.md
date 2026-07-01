# Briefcase

Briefcase is a local, source-grounded notebook assistant in the spirit of NotebookLM.
You create notebooks, add sources (files, pasted text, or URLs), ask questions, and
write documents that stay grounded in your material. Every answer is grounded in, and
cited back to, the exact passages in your own sources, and Briefcase refuses to answer
when the sources do not support a claim.

It is built on the retrieval and grounding ideas from
[Project Pragya](https://github.com/Hrishikesh2512/pragya), re-architected to run on a
laptop with **no mandatory cloud dependencies** and a clean web UI.

## Two kinds of data: sources and the draft

Briefcase separates what you read from what you write:

- **Sources** are read-only. They are the ground truth: uploaded files, pasted text,
  and websites. Briefcase never changes them.
- **The draft (sink)** is yours to change. It is an editable document per notebook that
  you and the assistant build together, always grounded in the sources.

## Features

- **Notebooks**: group sources into isolated workspaces.
- **Reads all the files**: PDF, Word (.docx), PowerPoint (.pptx), Markdown, HTML,
  plain text, CSV/TSV/JSON, and source code (.py, .js, .ts, .go, .rs, .java, and more).
  Add sources by upload, by URL, or by pasting text.
- **OCR for scans**: scanned or image-only PDFs and image files (.png, .jpg, and similar)
  are read with OCR when the optional engine is installed (auto-detected).
- **Grounded, cited answers**: hybrid retrieval (BM25 plus dense vectors, fused with
  Reciprocal Rank Fusion) with inline `[n]` citations you can click to see the exact quote.
- **Refuses when unsupported**: no evidence, no answer.
- **Draft workspace**: a slide-over editor for the sink (see below).
- **Web sources**: add sites, switch each one on or off, and refresh for live content.
- **Studio**: one-click grounded briefing and suggested questions per notebook.
- **In-app Settings**: add, view (masked), or remove API keys, configure a local
  (Ollama) or online (Gemini) model, and pick the current engine, all without editing
  env files or restarting.
- **Runs anywhere, offline**: default embeddings need no downloads and no API key.
- **Pluggable brains**: auto-detects Google Gemini (if `GEMINI_API_KEY` is set) or a local
  Ollama server for synthesized answers, and otherwise falls back to a fully offline
  extractive engine that still returns cited passages.

## Draft workspace (the sink)

Open the **Draft** drawer from a notebook. It is additive and hidden until you open it,
so the rest of the app is unchanged. Inside you can:

- **Edit directly**: a Markdown editor that autosaves.
- **Edit with the assistant**: give an instruction (for example "make it formal" or
  "add a section on X"). The assistant reads the sources and the current draft and
  returns the complete revised document. It edits in place, it does not just append.
- **Rewrite a selection**: select text and tell the assistant how to change it.
- **Undo**: revert the last assistant edit from version history.
- **Import from any file**: seed the draft from text, PDF, DOCX, PPTX, Markdown, HTML,
  code, or a scanned image (via OCR).
- **Export**: download the draft as Markdown (.md), plain text (.txt), or Word (.docx).
- **Check against sources**: fact-check every claim in the draft. This uses the model
  for real entailment, so a claim the sources contradict is flagged as unsupported,
  not merely matched on shared words.

The draft editor and the assistant edits work best with a generative model configured
(Gemini or Ollama).

## Web sources

Every source, including websites, has an on/off switch, so you can turn each site on or
off precisely. A disabled source is dimmed and excluded from both answers and drafts.
Website sources also have a refresh action to pull the latest content.

Fetching is hardened against SSRF: only http and https URLs are allowed, and requests to
localhost, private, link-local, and other non-public addresses are refused (including
across redirects), with a response size cap.

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
relevant passages and returns them verbatim with citations. For synthesized answers and
for the draft assistant, give it a generative model. It is auto-detected:

| Engine | How to enable |
| --- | --- |
| **Gemini** (cloud) | set `GEMINI_API_KEY`, or add it in the Settings panel (get a key at https://aistudio.google.com/apikey) |
| **Ollama** (local) | install [Ollama](https://ollama.com), run `ollama pull llama3.2` |
| **Extractive** (offline) | default when neither of the above is available |

The active engine is shown as a badge in the top-right of the app. If a model has no
quota for the default name, pick another in Settings (for example `gemini-2.5-flash`).

## Optional extras

```bash
./.venv/bin/pip install -e ".[ocr]"       # OCR for scanned PDFs and images
./.venv/bin/pip install -e ".[gemini]"    # Google Gemini
./.venv/bin/pip install -e ".[semantic]"  # stronger local embeddings
./.venv/bin/pip install -e ".[full]"      # gemini + ocr
```

For stronger semantic search after installing `[semantic]`:

```bash
export BRIEFCASE_EMBEDDINGS=sentence-transformers
```

## Configuration

All settings are environment variables (see `.env.example`), or set them in the in-app
Settings panel. Copy `.env.example` to `.env` and the launcher will load it. Common ones:

- `BRIEFCASE_LLM` = `auto` | `gemini` | `ollama` | `extractive`
- `BRIEFCASE_EMBEDDINGS` = `hashing` | `sentence-transformers` | `gemini`
- `BRIEFCASE_OCR` = `auto` | `off`
- `BRIEFCASE_DATA_DIR` = where the SQLite database and uploads live (default `./data`)

## Architecture

```
Browser SPA  (web/)   3-panel UI (Sources, Chat, Studio) plus a Draft slide-over
      |  REST / JSON
FastAPI  (briefcase/app.py, api/routes.py)
      |
      |-- ingest/     parsers (all file types, OCR) -> chunking -> embeddings
      |               safe URL fetch (SSRF-guarded)
      |-- rag/        hybrid retriever (BM25 + vectors + RRF)
      |               generator (LLM or extractive) -> citations + refusal
      |               author (draft edit / rewrite / fact-check)
      |               studio (briefing, suggested questions)
      |-- repository  SQLite (notebooks, sources, chunks, vectors, drafts)
      |-- settings    runtime settings + API keys (data/settings.json)
      |-- config      env-driven defaults
```

Everything persists to a single SQLite file under `data/`, so there is no external
database, queue, or vector service to run. API keys are stored locally in
`data/settings.json`, which is never committed.

## How it compares to Pragya

Pragya is an enterprise, multi-tenant RAG platform (Postgres, Redis, ChromaDB, JWT auth,
Prometheus). Briefcase keeps Pragya's core retrieval and grounding approach (hybrid search,
RRF fusion, cited answers, evidence-based refusal) but trades the heavy infrastructure for
a single-file, single-user, batteries-included app focused on the NotebookLM experience:
notebooks, broad file ingestion, an editable grounded draft, a polished UI, and
offline-first defaults.

## Project layout

```
briefcase/
  app.py              FastAPI app + static hosting
  cli.py              briefcase launcher
  config.py           env-driven settings
  db.py               SQLite schema, migrations, connection
  models.py           pydantic models
  repository.py       notebooks / sources / chunks / drafts persistence
  settings_store.py   runtime settings and API keys
  text_utils.py       tokenization, stopwords, helpers
  ingest/
    parsers.py        PDF, DOCX, PPTX, Markdown, HTML, text, code, images
    ocr.py            OCR backend (RapidOCR or Tesseract), auto-detected
    chunking.py       sentence-aware chunking with overlap
    service.py        ingestion pipeline and SSRF-safe URL fetching
  rag/
    embeddings.py     hashing / sentence-transformers / gemini embedders
    retriever.py      BM25 + dense vectors fused with RRF
    llm.py            gemini / ollama / none providers (auto-detected, retrying)
    generator.py      grounded answering with citations + refusal
    author.py         draft edit, rewrite, and fact-check
    studio.py         briefing + suggested questions
    prompts.py        system + task prompts
  api/routes.py       REST endpoints
web/
  index.html, styles.css, app.js   the single-page UI
```

## License

MIT.
