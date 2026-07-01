"use strict";

const API = "/api";
const state = {
  notebook: null,
  sources: [],
  selected: new Set(),
  status: null,
};

/* ------------------------- helpers ------------------------- */
function el(id) { return document.getElementById(id); }

async function api(path, opts = {}) {
  const res = await fetch(API + path, opts);
  if (res.status === 204) return null;
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

function toast(message, kind = "") {
  const stack = el("toast-stack");
  const node = document.createElement("div");
  node.className = "toast " + kind;
  node.textContent = message;
  node.title = "Click to dismiss";
  node.onclick = () => node.remove();
  stack.appendChild(node);
  // Errors linger so they can't be missed; success/info fade quickly.
  setTimeout(() => node.remove(), kind === "error" ? 9000 : 3800);
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

function emojiFor(title) {
  const set = ["📘", "📗", "📙", "📕", "📓", "🗂️", "📎", "🔖", "🧠", "🔬"];
  let h = 0;
  for (const ch of title) h = (h * 31 + ch.charCodeAt(0)) % set.length;
  return set[h];
}

/* Very small markdown-ish renderer for answers (bold, bullets, paragraphs),
   plus turning [n] markers into clickable citation chips. */
function renderAnswer(text) {
  const lines = text.split("\n");
  let html = "";
  let inList = false;
  for (let raw of lines) {
    let line = raw.trimEnd();
    if (/^\s*[-*]\s+/.test(line)) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += "<li>" + inline(line.replace(/^\s*[-*]\s+/, "")) + "</li>";
    } else {
      if (inList) { html += "</ul>"; inList = false; }
      if (line.trim() === "") continue;
      html += "<p>" + inline(line) + "</p>";
    }
  }
  if (inList) html += "</ul>";
  return html;
}

function inline(s) {
  s = escapeHtml(s);
  s = s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/_(.+?)_/g, "<em>$1</em>");
  s = s.replace(/\[(\d+)\]/g, (m, n) =>
    `<span class="cite-chip" data-cite="${n}">${n}</span>`);
  return s;
}

/* ------------------------- routing ------------------------- */
function showHome() {
  state.notebook = null;
  el("home-view").hidden = false;
  el("notebook-view").hidden = true;
  el("notebook-title-bar").textContent = "";
  loadNotebooks();
}

async function openNotebook(id) {
  const nb = await api(`/notebooks/${id}`);
  state.notebook = nb;
  el("home-view").hidden = true;
  el("notebook-view").hidden = false;
  el("notebook-title-bar").textContent = nb.title;
  el("chat-notebook-name").textContent = "Chat";
  el("chat-scroll").innerHTML = `<div class="chat-welcome" id="chat-welcome">
      <div class="welcome-icon">💬</div>
      <p>Ask anything about your sources. Every answer is grounded in, and cited back to, your documents.</p></div>`;
  el("summary-body").innerHTML = `<p class="muted">Generate a grounded overview of the selected sources.</p>`;
  el("questions-body").innerHTML = `<p class="muted">Get question ideas based on your sources.</p>`;
  await loadSources();
}

/* ------------------------- notebooks ------------------------- */
async function loadNotebooks() {
  const grid = el("notebook-grid");
  const notebooks = await api("/notebooks");
  el("home-empty").hidden = notebooks.length > 0;
  grid.innerHTML = "";
  for (const nb of notebooks) {
    const card = document.createElement("div");
    card.className = "notebook-card";
    card.innerHTML = `
      <button class="nb-delete" title="Delete">🗑</button>
      <div class="nb-emoji">${emojiFor(nb.title)}</div>
      <h3>${escapeHtml(nb.title)}</h3>
      <p>${escapeHtml(nb.description || "No description")}</p>
      <div class="nb-meta">${nb.source_count} source${nb.source_count === 1 ? "" : "s"}</div>`;
    card.querySelector(".nb-emoji").onclick = () => openNotebook(nb.id);
    card.querySelector("h3").onclick = () => openNotebook(nb.id);
    card.onclick = (e) => { if (!e.target.closest(".nb-delete")) openNotebook(nb.id); };
    card.querySelector(".nb-delete").onclick = async (e) => {
      e.stopPropagation();
      if (!confirm(`Delete notebook "${nb.title}" and all its sources?`)) return;
      await api(`/notebooks/${nb.id}`, { method: "DELETE" });
      toast("Notebook deleted", "success");
      loadNotebooks();
    };
    grid.appendChild(card);
  }
}

/* ------------------------- sources ------------------------- */
async function loadSources() {
  const list = el("source-list");
  const sources = await api(`/notebooks/${state.notebook.id}/sources`);
  state.sources = sources;
  state.selected = new Set(sources.map((s) => s.id));
  el("select-all-sources").checked = true;
  el("sources-empty").hidden = sources.length > 0;
  list.innerHTML = "";
  for (const s of sources) {
    const item = document.createElement("div");
    item.className = "source-item";
    item.innerHTML = `
      <input type="checkbox" checked data-id="${s.id}" />
      <div class="si-main">
        <div class="si-title" title="${escapeHtml(s.title)}">${escapeHtml(s.title)}</div>
        <div class="si-meta"><span class="si-type">${s.source_type}</span>${s.chunk_count} chunk${s.chunk_count === 1 ? "" : "s"}</div>
      </div>
      <button class="si-delete" title="Remove">🗑</button>`;
    item.querySelector("input").onchange = (e) => {
      if (e.target.checked) state.selected.add(s.id); else state.selected.delete(s.id);
      el("select-all-sources").checked = state.selected.size === state.sources.length;
    };
    item.querySelector(".si-main").onclick = () => viewSource(s.id);
    item.querySelector(".si-delete").onclick = async (e) => {
      e.stopPropagation();
      if (!confirm(`Remove "${s.title}"?`)) return;
      await api(`/notebooks/${state.notebook.id}/sources/${s.id}`, { method: "DELETE" });
      toast("Source removed", "success");
      loadSources();
    };
    list.appendChild(item);
  }
}

function selectedSourceIds() {
  if (state.selected.size === state.sources.length) return [];
  return [...state.selected];
}

async function viewSource(id) {
  const data = await api(`/notebooks/${state.notebook.id}/sources/${id}/text`);
  openModal(data.title, `<div class="reader-text">${escapeHtml(data.text)}</div>`);
}

/* ------------------------- chat ------------------------- */
function appendUser(text) {
  el("chat-welcome")?.remove();
  const scroll = el("chat-scroll");
  const node = document.createElement("div");
  node.className = "msg msg-user";
  node.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
  scroll.appendChild(node);
  scroll.scrollTop = scroll.scrollHeight;
}

function appendThinking() {
  const scroll = el("chat-scroll");
  const node = document.createElement("div");
  node.className = "msg msg-assistant";
  node.innerHTML = `<div class="who">Briefcase</div><div class="answer">
    <div class="typing"><span></span><span></span><span></span></div></div>`;
  scroll.appendChild(node);
  scroll.scrollTop = scroll.scrollHeight;
  return node;
}

function renderAnswerNode(node, ans) {
  const conf = Math.round((ans.confidence || 0) * 100);
  let citesHtml = "";
  if (ans.citations && ans.citations.length) {
    citesHtml = `<div class="citations">` + ans.citations.map((c) => `
      <div class="citation-card" id="cite-${c.marker}">
        <div class="cc-head"><span class="cc-num">[${c.marker}]</span>
        <span class="cc-title">${escapeHtml(c.source_title)}</span></div>
        <div class="cc-quote">"${escapeHtml(c.quote)}"</div>
      </div>`).join("") + `</div>`;
  }
  const engine = escapeHtml(ans.engine || "");
  node.className = "msg msg-assistant" + (ans.refused ? " refused" : "");
  node.innerHTML = `<div class="who">Briefcase</div>
    <div class="answer">
      ${renderAnswer(ans.answer)}
      <div class="answer-meta">
        <span>${ans.refused ? "No answer" : "Confidence"}</span>
        ${ans.refused ? "" : `<span class="confidence-bar"><span class="confidence-fill" style="width:${conf}%"></span></span><span>${conf}%</span>`}
        <span style="margin-left:auto">⚙ ${engine}</span>
      </div>
      ${citesHtml}
    </div>`;
  node.querySelectorAll(".cite-chip").forEach((chip) => {
    chip.onclick = () => {
      const card = node.querySelector(`#cite-${chip.dataset.cite}`);
      if (card) { card.scrollIntoView({ behavior: "smooth", block: "nearest" });
        card.classList.remove("flash"); void card.offsetWidth; card.classList.add("flash"); }
    };
  });
}

async function sendMessage(question) {
  if (!question.trim()) return;
  appendUser(question);
  const node = appendThinking();
  const scroll = el("chat-scroll");
  try {
    const ans = await api(`/notebooks/${state.notebook.id}/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ question, source_ids: selectedSourceIds() }),
    });
    renderAnswerNode(node, ans);
  } catch (err) {
    node.querySelector(".answer").innerHTML = `<p style="color:var(--danger)">${escapeHtml(err.message)}</p>`;
  }
  scroll.scrollTop = scroll.scrollHeight;
}

/* ------------------------- studio ------------------------- */
async function generateSummary() {
  const body = el("summary-body");
  body.innerHTML = `<span class="spinner"></span> <span class="muted">Working...</span>`;
  try {
    const data = await api(`/notebooks/${state.notebook.id}/summary`, {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ source_ids: selectedSourceIds() }),
    });
    body.innerHTML = data.summary ? renderAnswer(data.summary) : `<p class="muted">No sources to summarize.</p>`;
  } catch (err) { body.innerHTML = `<p style="color:var(--danger)">${escapeHtml(err.message)}</p>`; }
}

async function suggestQuestions() {
  const body = el("questions-body");
  body.innerHTML = `<span class="spinner"></span> <span class="muted">Thinking...</span>`;
  try {
    const data = await api(`/notebooks/${state.notebook.id}/questions`, {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ source_ids: selectedSourceIds() }),
    });
    if (!data.questions || !data.questions.length) { body.innerHTML = `<p class="muted">No sources yet.</p>`; return; }
    body.innerHTML = "";
    for (const q of data.questions) {
      const btn = document.createElement("button");
      btn.className = "q-chip";
      btn.textContent = q;
      btn.onclick = () => { el("chat-text").value = q; sendFromInput(); };
      body.appendChild(btn);
    }
  } catch (err) { body.innerHTML = `<p style="color:var(--danger)">${escapeHtml(err.message)}</p>`; }
}

/* ------------------------- modals ------------------------- */
function openModal(title, bodyHtml, wide = false) {
  el("modal-title").textContent = title;
  el("modal-body").innerHTML = bodyHtml;
  el("modal").classList.toggle("wide", wide);
  el("modal-overlay").hidden = false;
}
function closeModal() { el("modal-overlay").hidden = true; el("modal").classList.remove("wide"); }

/* ------------------------- settings ------------------------- */
const ENGINE_META = {
  auto: { tag: "Smart", desc: "Use the best available automatically: Gemini, then Ollama, then offline." },
  gemini: { tag: "Online", desc: "Google's cloud models. Fast and capable. Needs an API key." },
  ollama: { tag: "Local", desc: "A model running on your own machine. Private and offline." },
  extractive: { tag: "Offline", desc: "No model. Returns the most relevant cited passages verbatim." },
};

async function openSettings() {
  const cfg = await api("/settings");
  const s = cfg.settings;
  const active = cfg.active.llm;
  const ocr = cfg.active.ocr;
  const provider = s.llm_provider || "auto";
  const key = s.gemini_api_key || { set: false };

  const engineOpts = cfg.options.llm_provider.map((p) => {
    const meta = ENGINE_META[p] || { tag: "", desc: "" };
    let avail = "";
    if (p === "gemini") avail = key.set ? "Key configured" : "No API key yet";
    if (p === "ollama") avail = cfg.available.ollama ? "Detected on this machine" : "Not reachable";
    return `<label class="engine-opt ${p === provider ? "selected" : ""}" data-engine="${p}">
      <input type="radio" name="engine" value="${p}" ${p === provider ? "checked" : ""} />
      <div><div class="eo-title">${p}<span class="tag">${meta.tag}</span></div>
      <div class="eo-desc">${meta.desc}${avail ? ` &middot; ${avail}` : ""}</div></div>
    </label>`;
  }).join("");

  const embOpts = cfg.options.embeddings_provider.map((p) =>
    `<option value="${p}" ${p === (s.embeddings_provider || "hashing") ? "selected" : ""}>${p}</option>`).join("");
  const ocrOpts = cfg.options.ocr.map((p) =>
    `<option value="${p}" ${p === (s.ocr || "auto") ? "selected" : ""}>${p}</option>`).join("");

  const keyPlaceholder = key.set
    ? (key.source === "environment" ? "Set from environment variable" : `Saved: ${key.masked}`)
    : "Paste your Gemini API key";

  openModal("Settings", `
    <div class="settings">
      <section class="s-block">
        <h4>Answer engine <span class="pill ${active.generative ? "on" : "off"}">active: ${escapeHtml(active.engine)}</span></h4>
        <p class="s-hint">Choose which model answers your questions.</p>
        <div class="engine-options" id="engine-options">${engineOpts}</div>
      </section>

      <section class="s-block">
        <h4>Online model &mdash; Google Gemini</h4>
        <p class="s-hint">Get a free key at <a class="link" href="https://aistudio.google.com/apikey" target="_blank" rel="noreferrer">aistudio.google.com/apikey</a></p>
        <label>API key</label>
        <div class="key-row">
          <input id="s-gemini-key" type="password" placeholder="${escapeHtml(keyPlaceholder)}" autocomplete="off" />
          <button class="btn btn-ghost" id="s-reveal" type="button" title="Show/hide">👁</button>
          ${key.set && key.source === "stored" ? `<button class="btn btn-ghost" id="s-remove-key" type="button">Remove</button>` : ""}
        </div>
        <label>Model</label>
        <input id="s-gemini-model" value="${escapeHtml(s.gemini_model || "")}" placeholder="gemini-2.0-flash" />
      </section>

      <section class="s-block">
        <h4>Local model &mdash; Ollama</h4>
        <p class="s-hint">Install from <a class="link" href="https://ollama.com" target="_blank" rel="noreferrer">ollama.com</a>, then e.g. <code>ollama pull llama3.2</code></p>
        <label>Server URL</label>
        <input id="s-ollama-url" value="${escapeHtml(s.ollama_url || "")}" placeholder="${escapeHtml(cfg.available.ollama_url || "http://localhost:11434")}" />
        <label>Model</label>
        <input id="s-ollama-model" value="${escapeHtml(s.ollama_model || "")}" placeholder="llama3.2" />
        <div style="margin-top:10px"><button class="btn btn-ghost btn-sm" id="s-test-ollama" type="button">Test connection</button><span class="s-status" id="s-ollama-status"></span></div>
      </section>

      <section class="s-block">
        <h4>Embeddings</h4>
        <p class="s-hint">How text is indexed for search. Changing this applies to newly added sources.</p>
        <select id="s-embeddings">${embOpts}</select>
      </section>

      <section class="s-block">
        <h4>OCR for scans &amp; images <span class="pill ${ocr.backend ? "on" : "off"}">${ocr.backend ? escapeHtml(ocr.backend) : "not installed"}</span></h4>
        <p class="s-hint">Reads text from scanned PDFs and image files. ${ocr.backend ? "Ready." : "Install: pip install rapidocr-onnxruntime pdf2image"}</p>
        <select id="s-ocr">${ocrOpts}</select>
      </section>

      <div class="modal-actions">
        <button class="btn btn-ghost" id="s-cancel" type="button">Close</button>
        <button class="btn btn-primary" id="s-save" type="button">Save changes</button>
      </div>
    </div>`, true);

  // Wire interactions
  el("engine-options").querySelectorAll(".engine-opt").forEach((opt) => {
    opt.onclick = () => {
      el("engine-options").querySelectorAll(".engine-opt").forEach((o) => o.classList.remove("selected"));
      opt.classList.add("selected");
      opt.querySelector("input").checked = true;
    };
  });
  el("s-reveal").onclick = () => {
    const box = el("s-gemini-key");
    box.type = box.type === "password" ? "text" : "password";
  };
  const removeBtn = el("s-remove-key");
  if (removeBtn) removeBtn.onclick = async () => {
    if (!confirm("Remove the saved Gemini API key?")) return;
    await api("/settings/gemini_api_key", { method: "DELETE" });
    toast("API key removed", "success");
    loadStatus(); openSettings();
  };
  el("s-test-ollama").onclick = async () => {
    const status = el("s-ollama-status");
    status.textContent = " testing…"; status.className = "s-status";
    try {
      const r = await api("/settings/test-ollama", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ url: el("s-ollama-url").value.trim() || null }),
      });
      if (r.reachable) {
        status.textContent = ` reachable · ${r.models.length} model(s)${r.models.length ? ": " + r.models.slice(0, 3).join(", ") : ""}`;
        status.className = "s-status ok";
      } else { status.textContent = " not reachable at " + r.url; status.className = "s-status bad"; }
    } catch (e) { status.textContent = " " + e.message; status.className = "s-status bad"; }
  };
  el("s-cancel").onclick = closeModal;
  el("s-save").onclick = saveSettings;
}

async function saveSettings() {
  const patch = {
    llm_provider: document.querySelector('input[name="engine"]:checked')?.value || "auto",
    gemini_model: el("s-gemini-model").value.trim(),
    ollama_url: el("s-ollama-url").value.trim(),
    ollama_model: el("s-ollama-model").value.trim(),
    embeddings_provider: el("s-embeddings").value,
    ocr: el("s-ocr").value,
  };
  const newKey = el("s-gemini-key").value.trim();
  if (newKey) patch.gemini_api_key = newKey;  // only overwrite if user typed one

  const btn = el("s-save"); btn.disabled = true; btn.innerHTML = `<span class="spinner"></span>`;
  try {
    const cfg = await api("/settings", {
      method: "PUT", headers: { "content-type": "application/json" },
      body: JSON.stringify(patch),
    });
    toast(`Saved. Active engine: ${cfg.active.llm.engine}`, "success");
    loadStatus();
    closeModal();
  } catch (e) {
    toast(e.message, "error"); btn.disabled = false; btn.textContent = "Save changes";
  }
}

function newNotebookModal() {
  openModal("New notebook", `
    <label>Title</label><input id="m-title" placeholder="e.g. Research on X" />
    <label>Description (optional)</label><input id="m-desc" placeholder="What is this about?" />
    <div class="modal-actions">
      <button class="btn btn-ghost" id="m-cancel">Cancel</button>
      <button class="btn btn-primary" id="m-create">Create</button>
    </div>`);
  el("m-title").focus();
  el("m-cancel").onclick = closeModal;
  el("m-create").onclick = async () => {
    const title = el("m-title").value.trim();
    if (!title) { toast("Title is required", "error"); return; }
    const nb = await api("/notebooks", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ title, description: el("m-desc").value.trim() }),
    });
    closeModal();
    toast("Notebook created", "success");
    openNotebook(nb.id);
  };
}

function addUrlModal() {
  openModal("Add a URL", `
    <label>URL</label><input id="m-url" placeholder="https://..." />
    <label>Title (optional)</label><input id="m-title" placeholder="Override the page title" />
    <div class="modal-actions">
      <button class="btn btn-ghost" id="m-cancel">Cancel</button>
      <button class="btn btn-primary" id="m-add">Add</button>
    </div>`);
  el("m-url").focus();
  el("m-cancel").onclick = closeModal;
  el("m-add").onclick = async () => {
    const url = el("m-url").value.trim();
    if (!url) { toast("URL is required", "error"); return; }
    const btn = el("m-add"); btn.disabled = true; btn.innerHTML = `<span class="spinner"></span>`;
    try {
      await api(`/notebooks/${state.notebook.id}/sources/url`, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ url, title: el("m-title").value.trim() || null }),
      });
      closeModal(); toast("URL added", "success"); loadSources();
    } catch (err) { toast(err.message, "error"); btn.disabled = false; btn.textContent = "Add"; }
  };
}

function addTextModal() {
  openModal("Paste text", `
    <label>Title</label><input id="m-title" placeholder="Give this note a title" />
    <label>Text</label><textarea id="m-text" placeholder="Paste or type your content..."></textarea>
    <div class="modal-actions">
      <button class="btn btn-ghost" id="m-cancel">Cancel</button>
      <button class="btn btn-primary" id="m-add">Add</button>
    </div>`);
  el("m-title").focus();
  el("m-cancel").onclick = closeModal;
  el("m-add").onclick = async () => {
    const title = el("m-title").value.trim();
    const text = el("m-text").value.trim();
    if (!title || !text) { toast("Title and text are required", "error"); return; }
    const btn = el("m-add"); btn.disabled = true; btn.innerHTML = `<span class="spinner"></span>`;
    try {
      await api(`/notebooks/${state.notebook.id}/sources/text`, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ title, text }),
      });
      closeModal(); toast("Text added", "success"); loadSources();
    } catch (err) { toast(err.message, "error"); btn.disabled = false; btn.textContent = "Add"; }
  };
}

async function uploadFile(file) {
  const form = new FormData();
  form.append("file", file);
  toast(`Uploading ${file.name}...`);
  try {
    await api(`/notebooks/${state.notebook.id}/sources/upload`, { method: "POST", body: form });
    toast("File added", "success");
    loadSources();
  } catch (err) { toast(err.message, "error"); }
}

/* ------------------------- input wiring ------------------------- */
function sendFromInput() {
  const box = el("chat-text");
  const text = box.value.trim();
  if (!text) return;
  box.value = ""; box.style.height = "auto";
  sendMessage(text);
}

function wire() {
  el("brand").onclick = showHome;
  el("new-notebook-btn").onclick = newNotebookModal;
  el("settings-btn").onclick = openSettings;
  el("modal-close").onclick = closeModal;
  el("modal-overlay").onclick = (e) => { if (e.target === el("modal-overlay")) closeModal(); };

  el("upload-btn").onclick = () => el("file-input").click();
  el("file-input").onchange = (e) => { if (e.target.files[0]) uploadFile(e.target.files[0]); e.target.value = ""; };
  el("url-btn").onclick = addUrlModal;
  el("text-btn").onclick = addTextModal;

  el("select-all-sources").onchange = (e) => {
    const on = e.target.checked;
    state.selected = on ? new Set(state.sources.map((s) => s.id)) : new Set();
    el("source-list").querySelectorAll("input[type=checkbox]").forEach((c) => (c.checked = on));
  };

  el("chat-form").onsubmit = (e) => { e.preventDefault(); sendFromInput(); };
  const box = el("chat-text");
  box.oninput = () => { box.style.height = "auto"; box.style.height = Math.min(box.scrollHeight, 160) + "px"; };
  box.onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendFromInput(); } };

  el("clear-chat-btn").onclick = () => {
    el("chat-scroll").innerHTML = `<div class="chat-welcome" id="chat-welcome">
      <div class="welcome-icon">💬</div><p>Ask anything about your sources.</p></div>`;
  };
  el("summary-btn").onclick = generateSummary;
  el("questions-btn").onclick = suggestQuestions;

  document.onkeydown = (e) => { if (e.key === "Escape") closeModal(); };
}

async function loadStatus() {
  try {
    state.status = await api("/status");
    const badge = el("engine-badge");
    const llm = state.status.llm;
    badge.textContent = llm.generative ? `⚙ ${llm.engine}` : "⚙ extractive mode";
    badge.classList.toggle("generative", llm.generative);
    badge.title = llm.note
      ? llm.note
      : (llm.generative
        ? `Answers synthesized by ${llm.engine}`
        : "No generative model active. Answers are extracted from your sources. Open Settings to add Gemini or Ollama.");
    // If a model is configured but not usable, tell the user why (once).
    if (llm.note && state._lastNote !== llm.note) {
      state._lastNote = llm.note;
      toast(llm.note, "error");
    }
  } catch (e) { /* ignore */ }
}

wire();
loadStatus();
showHome();
if (location.hash === "#settings") openSettings();
