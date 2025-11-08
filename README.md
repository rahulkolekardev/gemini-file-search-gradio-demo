# Gemini File Search · Gradio Demo

[![Space](https://img.shields.io/badge/Hugging%20Face-Space-ffcc4d.svg)](https://huggingface.co/spaces/syntheticbot/Gemini-File-Search-Demo)
[![License](https://img.shields.io/badge/License-MIT-informational.svg)](#license)

A minimal, polished Gradio app for **Gemini File Search**. create/manage stores. upload & index files. ask grounded questions with optional metadata filters.

## Features
- Paste your **Gemini API key** per session. keys are not stored server-side
- Create a File Search Store from a local `samples/` folder. the UI shows which classics are found or missing
- Or create an empty store for uploads. then add any file (txt, pdf, docx, etc.)
- Configure chunking. set `max_tokens_per_chunk` and `max_overlap_tokens`
- Progress cards for upload and indexing
- Grounded Q&A over your store using `gemini-2.5-flash` or `gemini-2.5-pro`
- Metadata filtering via AIP-160 style queries. e.g. `author="Jane Austen" AND year=1813`
- Inspect `grounding_metadata` and operation summaries
- Manage stores from the UI. list or delete
- Custom theme + sample chips

## Quickstart (local)
```bash
git clone https://github.com/<you>/gemini-file-search-gradio-demo.git
cd gemini-file-search-gradio-demo
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
