import time
import json
from pathlib import Path
import gradio as gr

from google import genai
from google.genai import types

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "gemini-2.5-flash"
SAMPLES_DIR_DEFAULT = "samples"

SAMPLE_FILES = [
    {"path": "Pride_and_Prejudice.txt", "title": "Pride and Prejudice", "author": "Jane Austen", "year": 1813},
    {"path": "Adventures_of_Sherlock_Holmes.txt", "title": "The Adventures of Sherlock Holmes", "author": "Arthur Conan Doyle", "year": 1892},
    {"path": "Alices_Adventures_in_Wonderland.txt", "title": "Alice's Adventures in Wonderland", "author": "Lewis Carroll", "year": 1865},
    {"path": "Moby_Dick.txt", "title": "Moby-Dick", "author": "Herman Melville", "year": 1851},
]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _file_exists(path: Path) -> bool:
    try:
        return path.exists() and path.is_file() and path.stat().st_size > 0
    except Exception:
        return False

def _size_human(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    kb = n / 1024
    if kb < 1024:
        return f"{kb:.1f} KB"
    mb = kb / 1024
    return f"{mb:.2f} MB"

def _require_client(client_obj):
    """Strict. never fallback to env. always require user-pasted key."""
    if client_obj is None:
        raise RuntimeError("Set your Gemini API key first.")
    return client_obj

def _progress_html(pct: float, text: str) -> str:
    pct = max(0, min(100, pct))
    return f"""
    <div class="progress-card">
      <div class="progress-head">{text}</div>
      <div class="pbar"><div class="pbar-fill" style="width:{pct:.0f}%"></div></div>
      <div class="pbar-foot">{pct:.0f}%</div>
    </div>
    """

# ──────────────────────────────────────────────────────────────────────────────
# Core flows
# ──────────────────────────────────────────────────────────────────────────────
def ui_set_api_key(api_key: str):
    """Initialize Gemini client. API key is mandatory in Spaces."""
    api_key = (api_key or "").strip()
    if not api_key:
        return None, "❌ API key required. paste your Gemini key and click Set key."
    try:
        client = genai.Client(api_key=api_key)
        return client, "✅ API key set for this session."
    except Exception as e:
        return None, f"❌ Failed to initialize client. {e}"

def create_store_with_samples(client_state, samples_dir: str, store_display_name: str,
                              progress=gr.Progress(track_tqdm=True)):
    """
    Generator. Creates a NEW store and imports whatever classics exist in samples_dir.
    Yields: store_state(str), store_status(md), progress_html(html), create_button(update)
    """
    if client_state is None:
        msg = "❌ Set your API key first in the section above."
        yield "", msg, _progress_html(0, "Waiting for API key"), gr.update(interactive=True)
        return

    client = _require_client(client_state)
    display_name = (store_display_name or "").strip() or "file-search-samples"
    store = client.file_search_stores.create(config={"display_name": display_name})
    store_name = store.name

    base = Path(samples_dir or SAMPLES_DIR_DEFAULT)
    present = [spec for spec in SAMPLE_FILES if _file_exists(base / spec["path"])]
    total = max(len(present), 1)

    logs = []
    header = f"**Creating store:** `{store_name}` · display name **{display_name}**\n\n**Folder:** `{base.resolve()}`\n"
    status_md = header

    # Disable button immediately and show initial progress
    yield (store_name, status_md, _progress_html(1, "Preparing indexing"),
           gr.update(interactive=False, value="Creating…"))

    spinner = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
    done_count = 0

    for spec in SAMPLE_FILES:
        p = base / spec["path"]
        if not _file_exists(p):
            logs.append(f"• Missing: {p}")
            status_md = header + "\n".join(logs)
            pct = (done_count / total) * 100
            yield (store_name, status_md, _progress_html(pct, "Checking files"), gr.update(interactive=False))
            continue

        # Upload
        progress(((done_count + 0.05) / total), desc=f"Uploading {p.name}")
        logs.append(f"• Uploading: {p.name}")
        status_md = header + "\n".join(logs)
        pct = (done_count / total) * 100 + 5
        yield (store_name, status_md, _progress_html(pct, f"Uploading {p.name}"), gr.update(interactive=False))

        uploaded = client.files.upload(file=str(p), config={"display_name": spec["title"]})

        # Import + metadata
        meta = [
            types.CustomMetadata(key="title", string_value=spec["title"]),
            types.CustomMetadata(key="author", string_value=spec["author"]),
            types.CustomMetadata(key="year", numeric_value=spec["year"]),
            types.CustomMetadata(key="local_path", string_value=str(p)),
        ]
        import_cfg = types.ImportFileConfig(custom_metadata=meta)
        op = client.file_search_stores.import_file(
            file_search_store_name=store_name,
            file_name=uploaded.name,
            config=import_cfg,
        )

        # Poll with visible progress
        tick = 0
        while not op.done:
            time.sleep(0.5)
            tick += 1
            step_pct = min(95, 5 + tick * 3)  # 5→95 for this file
            overall = (done_count / total) * 100 + (step_pct / total)
            progress(min(0.95, 0.05 + 0.03 * tick), desc=f"Indexing {p.name} {spinner[tick % len(spinner)]}")
            yield (store_name, status_md, _progress_html(overall, f"Indexing {p.name} {spinner[tick % len(spinner)]}"),
                   gr.update(interactive=False))
            op = client.operations.get(op)

        done_count += 1
        logs.append(f"• Indexed: {p.name}  (author={spec['author']}, year={spec['year']})")
        status_md = header + "\n".join(logs)
        pct = (done_count / total) * 100
        yield (store_name, status_md, _progress_html(pct, f"Indexed {p.name}"), gr.update(interactive=False))

    final_header = "✅ Store created and files indexed." if present else "⚠️ Store created. no classics found to index."
    status_md = f"{final_header}\n\n{status_md}"
    yield (store_name, status_md, _progress_html(100, "Finished"),
           gr.update(interactive=False, value="Store created"))

def make_empty_store(client_state, display_name: str):
    if client_state is None:
        return "", "❌ Set your API key first."
    client = _require_client(client_state)
    dn = (display_name or "").strip() or "file-search-uploads"
    store = client.file_search_stores.create(config={"display_name": dn})
    return store.name, f"✅ Created empty store for uploads. `{store.name}`"

def set_existing_store(client_state, name: str):
    if client_state is None:
        return "", "❌ Set your API key first."
    client = _require_client(client_state)
    name = (name or "").strip()
    if not name:
        return "", "⚠️ Paste a store resource like `fileSearchStores/...`"
    try:
        store = client.file_search_stores.get(name=name)
        return store.name, f"✅ Using existing store: `{store.name}`"
    except Exception as e:
        return "", f"❌ Could not get store. {e}"

def upload_and_index(client_state, store_name: str, file_obj, display_file_name: str,
                     max_tokens: int, overlap_tokens: int,
                     progress=gr.Progress(track_tqdm=True)):
    """
    Generator. Upload local file to selected store and index it.
    Yields: op_summary(code), upload_status(md), upload_progress(html)
    """
    if client_state is None:
        yield gr.update(value=""), "❌ Set your API key first.", _progress_html(0, "Waiting for API key")
        return
    client = _require_client(client_state)

    if not store_name:
        yield gr.update(value=""), "⚠️ Create or select a store first.", _progress_html(0, "Waiting")
        return
    if file_obj is None:
        yield gr.update(value=""), "⚠️ Choose a file to upload.", _progress_html(0, "Waiting")
        return

    # Chunking config
    chunk_cfg = None
    if max_tokens or overlap_tokens:
        chunk_cfg = types.ChunkingConfig(
            white_space_config=types.WhiteSpaceConfig(
                max_tokens_per_chunk=int(max_tokens or 200),
                max_overlap_tokens=int(overlap_tokens or 20),
            )
        )

    cfg = None
    if display_file_name or chunk_cfg:
        cfg = types.UploadToFileSearchStoreConfig(
            display_name=(display_file_name.strip() or None) if display_file_name else None,
            chunking_config=chunk_cfg,
        )

    # Upload + index with progress
    fname = Path(file_obj.name).name
    progress(0.05, desc=f"Uploading {fname}")
    yield gr.update(value=""), f"Uploading **{fname}** …", _progress_html(5, f"Uploading {fname}")

    op = client.file_search_stores.upload_to_file_search_store(
        file=file_obj.name,
        file_search_store_name=store_name,
        config=cfg,
    )

    tick = 0
    spinner = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
    while not op.done:
        time.sleep(0.5)
        tick += 1
        pct = min(95, 5 + tick * 3)
        progress(min(0.95, 0.05 + 0.03 * tick), desc=f"Indexing {fname} {spinner[tick % len(spinner)]}")
        yield gr.update(value=""), f"Indexing **{fname}** …", _progress_html(pct, f"Indexing {fname} {spinner[tick % len(spinner)]}")
        op = client.operations.get(op)

    progress(1.0, desc="Done")
    op_summary = getattr(op, "response", None)
    op_text = json.dumps(op_summary, indent=2, default=str) if op_summary else "Indexed."
    yield gr.update(value=op_text), f"✅ File indexed into `{store_name}`", _progress_html(100, "Finished")

def ask(client_state, store_name: str, history_msgs, question: str, model_id: str, metadata_filter: str):
    if client_state is None:
        return history_msgs, "", "❌ Set your API key first."
    client = _require_client(client_state)

    if not store_name:
        return history_msgs, "", "⚠️ Pick or create a store first."
    q = (question or "").strip()
    if not q:
        return history_msgs, "", "⚠️ Type a question."

    tool = types.Tool(
        file_search=types.FileSearch(
            file_search_store_names=[store_name],
            metadata_filter=(metadata_filter.strip() or None),
        )
    )
    resp = client.models.generate_content(
        model=model_id or DEFAULT_MODEL,
        contents=q,
        config=types.GenerateContentConfig(tools=[tool]),
    )

    answer = resp.text or "No answer text."
    history = list(history_msgs or [])
    history.append({"role": "user", "content": q})
    history.append({"role": "assistant", "content": answer})

    grounding = "No grounding_metadata returned."
    try:
        gm = resp.candidates[0].grounding_metadata if resp.candidates else None
        if hasattr(gm, "model_dump"):
            grounding = json.dumps(gm.model_dump(), indent=2, default=str)
        elif isinstance(gm, dict):
            grounding = json.dumps(gm, indent=2, default=str)
        elif gm is not None:
            grounding = str(gm)
    except Exception as e:
        grounding = f"(Could not parse grounding metadata: {e})"

    return history, grounding, "✅ Done."

def list_stores(client_state):
    if client_state is None:
        return "❌ Set your API key first."
    client = _require_client(client_state)
    items = []
    for s in client.file_search_stores.list():
        items.append(f"{s.name}  |  display_name={getattr(s, 'display_name', '')}")
    return "\n".join(items) or "(none)"

def delete_store(client_state, store_name: str):
    if client_state is None:
        return "❌ Set your API key first."
    client = _require_client(client_state)
    if not store_name:
        return "⚠️ Enter a store name to delete."
    try:
        client.file_search_stores.delete(name=store_name, config={"force": True})
        return f"🗑️ Deleted: `{store_name}`"
    except Exception as e:
        return f"❌ Delete failed. {e}"

# ──────────────────────────────────────────────────────────────────────────────
# Local samples UI
# ──────────────────────────────────────────────────────────────────────────────
def render_samples_panel(samples_dir: str):
    base = Path(samples_dir or SAMPLES_DIR_DEFAULT)

    files = []
    if base.exists() and base.is_dir():
        for p in sorted(base.iterdir(), key=lambda x: x.name.lower()):
            if p.is_file():
                size = _size_human(p.stat().st_size)
                ext = p.suffix[1:].upper() if p.suffix else "FILE"
                files.append(f"""
                <div class="file-card">
                  <div class="file-name">{p.name}</div>
                  <div class="file-meta"><span class="pill">{ext}</span> <span class="muted">{size}</span></div>
                  <div class="file-path">{p}</div>
                </div>
                """)
    else:
        files.append("<div class='muted'>Folder not found. Create it and add files.</div>")

    tiles = []
    for spec in SAMPLE_FILES:
        p = base / spec["path"]
        ok = _file_exists(p)
        badge = '<span class="badge-ok">FOUND</span>' if ok else '<span class="badge-miss">MISSING</span>'
        tiles.append(f"""
        <div class="tile">
          <div class="tile-hd">{spec['title']}</div>
          <div class="tile-sub">{spec['author']} · {spec['year']}</div>
          <div class="tile-path">{p}</div>
          <div class="tile-badge">{badge}</div>
        </div>
        """)

    html = f"""
    <div class="gallery">
      <div class="gallery-col">
        <div class="section-hd">All files in <code>{base.resolve()}</code></div>
        <div class="file-grid">{''.join(files) if files else "<div class='muted'>No files</div>"}</div>
      </div>
      <div class="gallery-col">
        <div class="section-hd">Classics this demo can auto-index</div>
        <div class="tiles-wrap">{''.join(tiles)}</div>
        <div class="muted" style="margin-top:6px;">Tip. missing tiles mean you still need to save those .txt files into the folder above.</div>
      </div>
    </div>
    """
    return html

# ──────────────────────────────────────────────────────────────────────────────
# UI (palette + theme)
# ──────────────────────────────────────────────────────────────────────────────
custom_css = """
/* ---------- Palette ---------- */
:root {
  --primary: #6C5CE7; --secondary: #00D2FF; --accent: #FF8A65;
  --surface: #FFFFFF; --bg: #F6F7FB; --ink: #1B1E2B; --muted: #9AA0B7;
  --success: #22C55E; --danger: #EF4444; --code: #0f172a;
}
html[data-theme="dark"] {
  --primary: #7A70FF; --secondary: #00BFEA; --accent: #FF9E80;
  --surface: #151728; --bg: #0E0F16; --ink: #E9ECFF; --muted: #AAB0D6;
  --success: #23D497; --danger: #FF6B6B; --code: #0B0F1C;
}

/* ---------- Layout ---------- */
body { background: var(--bg) !important; }
.gradio-container { max-width: 1120px !important; margin: auto !important; }

/* ---------- Hero ---------- */
.hero {
  border-radius: 18px;
  padding: 22px 26px;
  background: #3F51B5;
  box-shadow: 0 12px 36px rgba(0,0,0,.14);
  margin: 8px 0 16px 0;
}
.hero h1 { color: var(--surface); font-size: 30px; margin: 0 0 8px; }
.hero p  { color: color-mix(in oklab, var(--surface) 78%, transparent); margin: 0; }

/* ---------- Cards & panels ---------- */
.gr-box, .gr-panel {
  background: var(--surface) !important;
  border: 1px solid color-mix(in oklab, var(--ink) 10%, transparent) !important;
  border-radius: 14px !important;
  box-shadow: 0 6px 16px rgba(0,0,0,.06);
}

/* ---------- Inputs ---------- */
input, textarea, select, .gr-textbox, .gr-number, .gr-dropdown, .gr-file, .gr-code {
  background: color-mix(in oklab, var(--surface) 96%, transparent) !important;
  color: var(--ink) !important;
  border: 1px solid color-mix(in oklab, var(--ink) 12%, transparent) !important;
  border-radius: 10px !important;
}

/* ---------- Buttons ---------- */
button, .gr-button { border-radius: 10px !important; border: 1px solid color-mix(in oklab, var(--ink) 10%, transparent) !important; }
button.primary   { background: linear-gradient(135deg, var(--primary), var(--secondary)) !important; color: white !important; }
button.secondary { background: color-mix(in oklab, var(--surface) 90%, transparent) !important; color: var(--ink) !important; }

/* ---------- Accordion (Spaces uses <details>) ---------- */
.gradio-container details {
  margin: 10px 0;
  background: transparent !important;
  border: 0 !important;
}
.gradio-container details > summary {
  list-style: none;
  background: color-mix(in oklab, var(--surface) 98%, transparent);
  border: 1px solid color-mix(in oklab, var(--ink) 10%, transparent);
  border-radius: 12px;
  padding: 10px 14px;
  font-weight: 600;
  color: var(--ink);
  cursor: pointer;
}
.gradio-container details > summary::-webkit-details-marker { display: none; }
.gradio-container details[open] > summary {
  border-bottom-left-radius: 0; border-bottom-right-radius: 0;
}
.gradio-container details > div {
  background: var(--surface);
  border: 1px solid color-mix(in oklab, var(--ink) 10%, transparent);
  border-top: 0;
  border-bottom-left-radius: 12px; border-bottom-right-radius: 12px;
  padding: 12px 14px;
  box-shadow: 0 6px 16px rgba(0,0,0,.06);
}

/* ---------- Chatbot ---------- */
.gr-chatbot {
  background: color-mix(in oklab, var(--surface) 96%, transparent) !important;
  border: 1px solid color-mix(in oklab, var(--ink) 10%, transparent) !important;
  border-radius: 14px !important;
}
.gr-chatbot .message {
  border-radius: 14px; padding: 12px 14px; font-size: 17px; line-height: 1.5;
}
.gr-chatbot .message.user {
  background: color-mix(in oklab, var(--secondary) 12%, transparent) !important;
  border: 1px solid color-mix(in oklab, var(--secondary) 20%, transparent);
}
.gr-chatbot .message.bot {
  background: color-mix(in oklab, var(--primary) 12%, transparent) !important;
  border: 1px solid color-mix(in oklab, var(--primary) 20%, transparent);
}

/* ---------- Chips ---------- */
.chip { padding:8px 12px; border-radius:999px; background: color-mix(in oklab, var(--primary) 12%, var(--surface));
  color: var(--ink); border:1px solid color-mix(in oklab, var(--primary) 20%, transparent); user-select:none; }
.chip:hover { filter:brightness(1.04); }

/* ---------- Sample gallery ---------- */
.section-hd { font-weight: 700; margin-bottom: 8px; color: var(--ink); }
.gallery { display:grid; grid-template-columns: 1.3fr .9fr; gap: 14px; }
@media (max-width: 980px) { .gallery { grid-template-columns: 1fr; } }

.file-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 10px; }
.file-card { background: color-mix(in oklab, var(--surface) 96%, transparent); border:1px solid color-mix(in oklab, var(--ink) 10%, transparent);
  border-radius: 12px; padding: 10px 12px; box-shadow:0 6px 16px rgba(0,0,0,.05); }
.file-name { font-weight: 700; margin-bottom: 4px; color: var(--ink); }
.file-meta { font-size: 13px; color: var(--muted); display:flex; gap:8px; align-items:center; }
.file-path { color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px; word-break: break-all; margin-top: 4px; }
.pill { padding: 2px 8px; border-radius: 999px; background: color-mix(in oklab, var(--secondary) 14%, transparent); color: var(--ink);
  border:1px solid color-mix(in oklab, var(--secondary) 20%, transparent); }

.tiles-wrap { display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:12px; }
.tile { background: color-mix(in oklab, var(--surface) 96%, transparent); border:1px solid color-mix(in oklab, var(--ink) 10%, transparent);
  border-radius:12px; padding:12px; box-shadow:0 6px 16px rgba(0,0,0,.06); display:flex; flex-direction:column; gap:6px; }
.tile-hd { font-weight:700; color:var(--ink); }
.tile-sub { color: var(--muted); font-size: 14px; }
.tile-path { color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px; word-break: break-all; }
.badge-ok, .badge-miss { font-size: 11.5px; padding: 3px 8px; border-radius: 999px; color: white; width: fit-content; }
.badge-ok { background: var(--success); }
.badge-miss { background: var(--danger); }

/* ---------- Code blocks ---------- */
.gr-code { background: var(--code) !important; color: #d9e1ff !important; border: 1px solid #1f2a44 !important; }

/* ---------- Progress card ---------- */
.progress-card {
  background: color-mix(in oklab, var(--surface) 96%, transparent);
  border: 1px solid color-mix(in oklab, var(--ink) 10%, transparent);
  border-radius: 12px;
  padding: 10px 12px;
  box-shadow: 0 6px 16px rgba(0,0,0,.05);
}
.progress-head { font-weight: 600; color: var(--ink); margin-bottom: 6px; }
.pbar { height: 12px; width: 100%; background: #e8ebff; border-radius: 999px; overflow: hidden; }
html[data-theme="dark"] .pbar { background: #2a2f57; }
.pbar-fill { height: 100%; background: linear-gradient(90deg, var(--primary), var(--secondary));
  border-radius: 999px; transition: width .25s ease; }
.pbar-foot { font-size: 12px; color: var(--muted); margin-top: 6px; }
"""

# Pre-render samples gallery
_initial_gallery_html = render_samples_panel(SAMPLES_DIR_DEFAULT)

with gr.Blocks(
    title="Gemini File Search – Gradio Demo (Spaces)",
    css=custom_css,
    theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
) as demo:
    gr.HTML("""
    <div class="hero">
      <h1>Gemini File Search · RAG Demo</h1>
      <p>Paste your Gemini API key to begin. create a store from local classics. upload your own files. ask grounded questions.</p>
    </div>
    """)

    client_state = gr.State(value=None)
    store_state = gr.State(value="")
    chat_state = gr.State(value=[])

    # API key
    with gr.Accordion("API key", open=True):
        gr.Markdown("_This Space **requires** your Gemini API key for each session. keys are **not** stored server-side._")
        with gr.Row():
            api_tb = gr.Textbox(label="Gemini API key", placeholder="Paste your API key…", type="password")
            api_btn = gr.Button("Set key", elem_classes=["primary"])
        api_status = gr.Markdown()

    # Build from samples
    with gr.Group():
        gr.Markdown("### Build a store from your `samples/` folder")
        with gr.Row():
            samples_dir_tb = gr.Textbox(label="Samples folder path", value=SAMPLES_DIR_DEFAULT, scale=3)
            store_display_name = gr.Textbox(label="Store display name", value="file-search-samples", scale=2)
            create_from_samples_btn = gr.Button("Create store with these files", elem_classes=["primary"], scale=1)
        samples_preview = gr.HTML(value=_initial_gallery_html)
        samples_progress_html = gr.HTML(value="")
        store_status = gr.Markdown()
        store_name_out = gr.Textbox(label="Active store name", interactive=False)

    # Existing store
    with gr.Accordion("Use existing store (paste resource name)", open=False):
        gr.Markdown("_Paste a resource like `fileSearchStores/...` to switch the active store._")
        with gr.Row():
            use_existing_input = gr.Textbox(label="Existing store resource", placeholder="fileSearchStores/…", scale=4)
            use_existing_btn = gr.Button("Use store", elem_classes=["secondary"], scale=1)
        existing_status = gr.Markdown()

    # Upload own files
    with gr.Accordion("Upload & index your own files", open=False):
        gr.Markdown("_Create or select a store. then upload any local file here._")
        with gr.Row():
            upload_store_dn = gr.Textbox(label="Or create empty store for uploads (display name)", value="file-search-uploads", scale=3)
            create_empty_btn = gr.Button("Create empty store", elem_classes=["secondary"], scale=1)
        with gr.Row():
            file_uploader = gr.File(label="Choose a file (txt, pdf, docx, etc.)")
            disp_file_name = gr.Textbox(label="Display file name (for citations)", placeholder="my-notes.txt")
        with gr.Row():
            max_tokens = gr.Number(label="max_tokens_per_chunk", value=200, precision=0)
            overlap_tokens = gr.Number(label="max_overlap_tokens", value=20, precision=0)
        upload_btn = gr.Button("Upload & Index", elem_classes=["primary"])
        upload_progress_html = gr.HTML(value="")
        op_summary = gr.Code(label="Operation summary")
        upload_status = gr.Markdown()

    gr.Markdown("---")
    # Q&A
    gr.Markdown("### Ask grounded questions")
    with gr.Row():
        model = gr.Dropdown(label="Model", value=DEFAULT_MODEL, choices=["gemini-2.5-flash", "gemini-2.5-pro"], scale=1)
        metadata_filter = gr.Textbox(label="Optional metadata_filter (AIP-160)", placeholder='author="Jane Austen" AND year=1813', scale=2)
    chatbot = gr.Chatbot(label="Grounded Q&A", height=520, type="messages")

    # Example chips
    gr.Markdown("Try some examples:")
    with gr.Row():
        chip1 = gr.Button("Who is Mr. Darcy. what role does he play", elem_classes=["chip"])
        chip2 = gr.Button("What is the opening line of Moby-Dick", elem_classes=["chip"])
        chip3 = gr.Button("Summarize the tea party scene in Alice", elem_classes=["chip"])
        chip4 = gr.Button("Give 3 clues Holmes uses in a story", elem_classes=["chip"])
        chip5 = gr.Button("What is Pemberley. why is it important", elem_classes=["chip"])
        chip6 = gr.Button("Who is Captain Ahab. what drives him", elem_classes=["chip"])

    with gr.Row():
        question_tb = gr.Textbox(placeholder="Type your question…", show_label=False, scale=5)
        ask_btn = gr.Button("Ask", elem_classes=["primary"], scale=1)
        clear_btn = gr.Button("Clear", elem_classes=["secondary"], scale=1)
    with gr.Accordion("grounding_metadata (raw)", open=False):
        grounding_md = gr.Code(label="grounding_metadata")
    note = gr.Markdown()

    # Manage stores (optional tools)
    with gr.Accordion("Manage stores", open=False):
        list_btn = gr.Button("List my stores", elem_classes=["secondary"])
        list_out = gr.Code()
        with gr.Row():
            del_name = gr.Textbox(label="Store name to delete")
            del_btn = gr.Button("Delete store", elem_classes=["secondary"])
        del_status = gr.Markdown()

    # Wiring
    api_btn.click(ui_set_api_key, [api_tb], [client_state, api_status], show_progress=True)

    samples_dir_tb.change(render_samples_panel, [samples_dir_tb], [samples_preview], show_progress=False)

    create_from_samples_btn.click(
        create_store_with_samples,
        [client_state, samples_dir_tb, store_display_name],
        [store_state, store_status, samples_progress_html, create_from_samples_btn],
        show_progress=True,
    ).then(lambda s: s, [store_state], [store_name_out], show_progress=False)

    use_existing_btn.click(
        set_existing_store,
        [client_state, use_existing_input],
        [store_state, existing_status],
        show_progress=True,
    ).then(lambda s: s, [store_state], [store_name_out], show_progress=False)

    create_empty_btn.click(
        make_empty_store,
        [client_state, upload_store_dn],
        [store_state, upload_status],
        show_progress=True,
    ).then(lambda s: s, [store_state], [store_name_out], show_progress=False)

    upload_btn.click(
        upload_and_index,
        [client_state, store_state, file_uploader, disp_file_name, max_tokens, overlap_tokens],
        [op_summary, upload_status, upload_progress_html],
        show_progress=True,
    )

    def _ask_and_update(client_state, store_name, chat_messages, q, model, mfilter):
        history, grounding, msg = ask(client_state, store_name, chat_messages, q, model, mfilter)
        return history, grounding, msg, ""

    ask_btn.click(
        _ask_and_update,
        [client_state, store_state, chat_state, question_tb, model, metadata_filter],
        [chatbot, grounding_md, note, question_tb],
        show_progress=True,
    ).then(lambda h: h, [chatbot], [chat_state], show_progress=False)

    clear_btn.click(lambda: ([], "", ""), None, [chatbot, grounding_md, note], show_progress=False)\
             .then(lambda: [], None, [chat_state], show_progress=False)

    # Chips → prefill the question textbox
    chip1.click(lambda: "Who is Mr. Darcy. what role does he play in the story", None, [question_tb], show_progress=False)
    chip2.click(lambda: "What is the opening line of Moby-Dick", None, [question_tb], show_progress=False)
    chip3.click(lambda: "Summarize the tea party scene and why it is absurd", None, [question_tb], show_progress=False)
    chip4.click(lambda: "Give 3 clues Holmes uses in any one Sherlock Holmes story", None, [question_tb], show_progress=False)
    chip5.click(lambda: "What is Pemberley. why is it important in Pride and Prejudice", None, [question_tb], show_progress=False)
    chip6.click(lambda: "Who is Captain Ahab. what drives his obsession", None, [question_tb], show_progress=False)

    # Tools
    list_btn.click(list_stores, [client_state], [list_out], show_progress=True)
    del_btn.click(delete_store, [client_state, del_name], [del_status], show_progress=True)

if __name__ == "__main__":
    demo.queue().launch()
