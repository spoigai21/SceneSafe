"""Labelling page for the second labeller (EVAL-PROTOCOL.md, "Labels"). Python standard library only.

    PYTHONPATH=. .venv/bin/python scripts/label_server.py [--phase 4b]      then open http://localhost:8765

Shows the sampled pairs (Phase 4: 100, default_rng(0); Phase 4b and 4c: 50 new pairs, default_rng(1) and (2);
Phase 4d: 100, default_rng(3); Phase 4e to 4h: 100, default_rng(4) to (7); Phase 5 and 5b: 100 scenes, default_rng(8) and (9);
Phase 5c: 50 scenes, default_rng(10), shown as the larger 2 x 2 frame grid)
one at a time: the ad's category and
brief (from 4d, also its label-guide line; from 4e, its theme line and a third button for a thematic fit), 4 frames
from the scene and, from 4f, the scene's dialogue. Nothing on the page says which method
found the scene, and the primary labels are never sent to the browser. Phase 5 is different: no ad, just the scene,
and a checkbox for each content tag that is present (brand safety); the detector's tags are never sent either. Every click is saved straight to
results/<phase>/labels_author.json, so you can stop and come back.
"""

import argparse
import json
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

from scenesafe.common.results import REPO_ROOT
from scenesafe.match.labels import author_sample

# results dir, sample size, seed, frames dir, show the label guide
_P4_FRAMES = REPO_ROOT / "data" / "phase4" / "test" / "frames"
PHASES = {"4": ("phase4", 100, 0, _P4_FRAMES, False), "4b": ("phase4b", 50, 1, _P4_FRAMES, False),
          "4c": ("phase4c", 50, 2, _P4_FRAMES, False),
          "4d": ("phase4d", 100, 3, REPO_ROOT / "data" / "phase4d" / "frames", True),
          "4e": ("phase4e", 100, 4, REPO_ROOT / "data" / "phase4d" / "frames", True),
          "4f": ("phase4f", 100, 5, REPO_ROOT / "data" / "phase4d" / "frames", True),
          "4g": ("phase4g", 100, 6, REPO_ROOT / "data" / "phase4d" / "frames", True),
          "4h": ("phase4h", 100, 7, REPO_ROOT / "data" / "phase4d" / "frames", True),
          "5": ("phase5", 100, 8, REPO_ROOT / "data" / "phase4d" / "frames", False),
          "5b": ("phase5b", 100, 9, REPO_ROOT / "data" / "phase4d" / "frames", False),
          "5c": ("phase5c", 50, 10, REPO_ROOT / "data" / "phase5" / "grids", False),
          "5d": ("phase5d", 50, 11, REPO_ROOT / "data" / "phase5" / "grids", False)}
LEVELS = {"4e", "4f", "4g", "4h"}    # three labels: 2 literal fit, 1 thematic fit, 0 doesn't fit
PORT = 8765

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Scene labelling</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; background: #f6f6f4; color: #1d1d1b; margin: 0; }
  main { max-width: 860px; margin: 32px auto; padding: 0 16px; }
  .top { display: flex; justify-content: space-between; color: #666; font-size: 14px; }
  .bar { height: 4px; background: #ddd; border-radius: 2px; margin: 8px 0 24px; }
  .bar div { height: 4px; background: #1d1d1b; border-radius: 2px; }
  .cat { text-transform: uppercase; letter-spacing: .06em; font-size: 12px; color: #888; }
  h1 { font-size: 26px; margin: 4px 0 18px; }
  img { width: 100%; border-radius: 6px; background: #000; }
  .guide { font-size: 14px; color: #444; background: #fff; border: 1px solid #ddd; border-radius: 6px;
           padding: 8px 12px; margin: -6px 0 14px; }
  .dialogue { font-size: 14px; color: #5a3a00; background: #fffaf0; border: 1px solid #e8d9b5; border-radius: 6px;
              padding: 8px 12px; margin-top: 10px; max-height: 160px; overflow-y: auto; }
  .q { margin: 18px 0 12px; font-size: 15px; color: #444; }
  .buttons { display: flex; gap: 12px; }
  button { flex: 1; font-size: 17px; padding: 14px; border-radius: 8px; border: 1px solid #bbb; background: #fff;
           cursor: pointer; }
  button.yes { border-color: #2e7d32; } button.no { border-color: #b23b3b; } button.theme { border-color: #2b55a8; }
  button.chosen { background: #1d1d1b; color: #fff; }
  .small { margin-top: 14px; display: flex; justify-content: space-between; font-size: 13px; color: #777; }
  .small a { color: #555; cursor: pointer; }
</style></head>
<body><main>
  <div class="top"><span id="pos"></span><span>F = fits &middot; D = doesn't fit &middot; &larr; back</span></div>
  <div class="bar"><div id="fill"></div></div>
  <div id="card">
    <div class="cat" id="cat"></div>
    <h1 id="brief"></h1>
    <div class="guide" id="guide" hidden></div>
    <img id="img" alt="4 frames from the scene">
    <div class="dialogue" id="dialogue" hidden></div>
    <div class="q">Is this scene a natural place for this ad? (Relevance only. Ignore safety. Unsure = doesn't fit.)</div>
    <div class="buttons">
      <button class="yes" id="yes">Fits (F)</button>
      <button class="theme" id="theme" hidden>Thematic fit (T)</button>
      <button class="no" id="no">Doesn't fit (D)</button>
    </div>
    <div class="small"><a id="back">&larr; back</a><span id="saved"></span></div>
  </div>
  <div id="done" hidden><h1>All done. Thank you.</h1><p>Your labels are saved in labels_author.json.
     You can close this tab and stop the server (Ctrl+C).</p><p><a id="review">Review from the start</a></p></div>
</main>
<script>
let items = [], labels = {}, i = 0;
const $ = id => document.getElementById(id);
function show() {
  const n = Object.keys(labels).length;
  $("fill").style.width = (100 * n / items.length) + "%";
  $("pos").textContent = `${n} of ${items.length} labelled`;
  if (i >= items.length) { $("card").hidden = true; $("done").hidden = false; return; }
  $("card").hidden = false; $("done").hidden = true;
  const it = items[i];
  $("cat").textContent = `${i + 1} / ${items.length} · ${it.category}`;
  $("brief").textContent = it.brief;
  $("guide").hidden = !it.fits;
  if (it.fits) $("guide").textContent = it.theme
    ? `Literal fit: ${it.fits}. Thematic fit: ${it.theme}. Doesn't fit: neither.`
    : `Fits: ${it.fits}. Doesn't fit: ${it.not}.`;
  $("theme").hidden = !it.theme;
  $("yes").textContent = it.theme ? "Literal fit (F)" : "Fits (F)";
  $("theme").classList.toggle("chosen", labels[it.pair] === 1);
  $("img").src = "/frames/" + encodeURIComponent(it.scene) + ".jpg";
  $("dialogue").hidden = !("dialogue" in it);
  if ("dialogue" in it) $("dialogue").textContent = "Dialogue: " + (it.dialogue || "(no speech)");
  $("yes").classList.toggle("chosen", labels[it.pair] === (it.theme ? 2 : 1));
  $("no").classList.toggle("chosen", labels[it.pair] === 0);
}
async function choose(v) {
  if (i >= items.length) return;
  const it = items[i];
  const r = await fetch("/label", {method: "POST", headers: {"Content-Type": "application/json"},
                                   body: JSON.stringify({pair: it.pair, value: v})});
  if (!r.ok) { $("saved").textContent = "NOT saved - is the server running?"; return; }
  labels[it.pair] = v;
  $("saved").textContent = "saved";
  i++;
  while (i < items.length && items[i].pair in labels) i++;
  show();
}
const yesValue = () => (items[i] && items[i].theme ? 2 : 1);
$("yes").onclick = () => choose(yesValue());
$("theme").onclick = () => choose(1);
$("no").onclick = () => choose(0);
$("back").onclick = () => { if (i > 0) { i--; show(); } };
$("review").onclick = () => { i = 0; show(); };
document.addEventListener("keydown", e => {
  if (e.key === "f" || e.key === "F") choose(yesValue());
  else if ((e.key === "t" || e.key === "T") && items[i] && items[i].theme) choose(1);
  else if (e.key === "d" || e.key === "D") choose(0);
  else if (e.key === "ArrowLeft" && i > 0) { i--; show(); }
});
(async () => {
  items = await (await fetch("/items")).json();
  labels = await (await fetch("/labels")).json();
  while (i < items.length && items[i].pair in labels) i++;
  show();
})();
</script></body></html>
"""


SAFETY_PAGE = PAGE.split("<body>")[0] + """<body><main>
  <div class="top"><span id="pos"></span><span>tick what is present &middot; Enter = save &amp; next &middot; &larr; back</span></div>
  <div class="bar"><div id="fill"></div></div>
  <div id="card">
    <div class="cat" id="cat"></div>
    <h1>Which of these are present in the scene?</h1>
    <img id="img" alt="4 frames from the scene">
    <div class="dialogue" id="dialogue"></div>
    <div id="tags" style="margin-top:14px"></div>
    <div class="buttons" style="margin-top:14px"><button class="yes" id="save">Save &amp; next (Enter)</button></div>
    <div class="small"><a id="back" href="#">&larr; back</a> &middot; go to # <input id="goto" type="number" min="1"
      style="width:60px"><span id="saved"></span></div>
  </div>
  <div id="done" hidden><h1>All done. Thank you.</h1><p>Your labels are saved in labels_author.json.</p>
     <p><a id="review" href="#1">Review from the start</a> &middot; go to # <input id="goto2" type="number" min="1"
      style="width:60px"></p></div>
</main>
<script>
let items = [], labels = {}, tags = {}, i = 0;
const $ = id => document.getElementById(id);
function show() {
  const n = Object.keys(labels).length;
  $("fill").style.width = (100 * n / items.length) + "%";
  $("pos").textContent = `${n} of ${items.length} labelled`;
  if (i >= items.length) { $("card").hidden = true; $("done").hidden = false; return; }
  $("card").hidden = false; $("done").hidden = true;
  const it = items[i], have = labels[it.pair] || [];
  $("cat").textContent = `${i + 1} / ${items.length} · nothing ticked = none of them`;
  $("img").src = "/frames/" + encodeURIComponent(it.scene) + ".jpg";
  $("dialogue").textContent = "Dialogue: " + (it.dialogue || "(no speech)");
  $("tags").innerHTML = Object.entries(tags).map(([t, d]) =>
    `<label style="display:block;margin:6px 0;font-size:15px"><input type="checkbox" value="${t}"
      ${have.includes(t) ? "checked" : ""}> <b>${t}</b> <span style="color:#666">${d}</span></label>`).join("");
}
async function save() {
  if (i >= items.length) return;
  const it = items[i];
  const v = [...document.querySelectorAll("#tags input:checked")].map(x => x.value);
  const r = await fetch("/label", {method: "POST", headers: {"Content-Type": "application/json"},
                                   body: JSON.stringify({pair: it.pair, value: v})});
  if (!r.ok) { $("saved").textContent = "NOT saved - is the server running?"; return; }
  const again = it.pair in labels;
  labels[it.pair] = v; $("saved").textContent = "saved"; i++;
  if (!again) while (i < items.length && items[i].pair in labels) i++;   // reviewing steps one by one
  show();
}
$("save").onclick = save;
$("back").onclick = e => { e.preventDefault(); if (i > 0) { i--; show(); } };
$("review").onclick = e => { e.preventDefault(); i = 0; show(); };
function go(v) { const n = parseInt(v); if (n >= 1 && n <= items.length) { i = n - 1; show(); } }
for (const id of ["goto", "goto2"]) $(id).addEventListener("keydown", e => {
  if (e.key === "Enter") { e.preventDefault(); e.stopPropagation(); go(e.target.value); e.target.value = ""; } });
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" && e.target.type === "number") return;
  if (e.key === "Enter") save();
  else if (e.key === "ArrowLeft" && i > 0) { i--; show(); }
});
(async () => {
  tags = await (await fetch("/tags")).json();
  items = await (await fetch("/items")).json();
  labels = await (await fetch("/labels")).json();
  while (i < items.length && items[i].pair in labels) i++;
  if (location.hash.length > 1) go(location.hash.slice(1));
  show();
})();
</script></body></html>
"""


def safety_items(size: int, seed: int) -> list[dict]:
    pool = json.loads(POOL.read_text())["items"]
    return [{"pair": pool[i]["id"], "scene": pool[i]["scene"], "dialogue": pool[i]["dialogue"]}
            for i in author_sample(len(pool), size, seed)]


def valid(value) -> bool:
    if PHASE.startswith("5"):
        return isinstance(value, list) and all(v in TAG_DEFS for v in value) and len(set(value)) == len(value)
    return value in VALUES


def load_labels() -> dict:
    return json.loads(OUT.read_text())["labels"] if OUT.exists() else {}


def save_labels(labels: dict):
    record = {"labeller": "project author", "protocol": "EVAL-PROTOCOL.md", "labels": labels}
    fd, tmp = tempfile.mkstemp(dir=OUT.parent, suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(record, f, indent=1, sort_keys=True)
    os.replace(tmp, OUT)


def items(size: int, seed: int, with_guide: bool, with_themes: bool = False) -> list[dict]:
    pool = json.loads(POOL.read_text())["pairs"]
    keys = ("pair", "ad", "category", "brief", "scene", "dialogue")
    out = [{k: pool[i][k] for k in keys if k in pool[i]} for i in author_sample(len(pool), size, seed)]
    if with_guide:
        from scenesafe.match.catalogue import label_guide
        guide = label_guide()
        out = [{**it, **guide[it["ad"]]} for it in out]
    if with_themes:
        from scenesafe.match.catalogue import themes
        th = themes()
        out = [{**it, "theme": th[it["ad"]]["theme"]} for it in out]
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, ctype: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self._send((SAFETY_PAGE if PHASE.startswith("5") else PAGE).encode(), "text/html; charset=utf-8")
        elif self.path == "/tags":
            self._send(json.dumps(TAG_DEFS).encode(), "application/json")
        elif self.path == "/items":
            self._send(json.dumps(ITEMS).encode(), "application/json")
        elif self.path == "/labels":
            self._send(json.dumps(load_labels()).encode(), "application/json")
        elif self.path.startswith("/frames/"):
            name = unquote(self.path[len("/frames/"):])
            path = (FRAMES / name).resolve()
            if path.parent != FRAMES.resolve() or not path.exists():
                self._send(b"not found", "text/plain", 404)
            else:
                self._send(path.read_bytes(), "image/jpeg")
        else:
            self._send(b"not found", "text/plain", 404)

    def do_POST(self):
        if self.path != "/label":
            return self._send(b"not found", "text/plain", 404)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body.get("pair") not in ALLOWED or not valid(body.get("value")):
            return self._send(b"bad label", "text/plain", 400)
        with LOCK:
            labels = load_labels()
            labels[body["pair"]] = body["value"]
            save_labels(labels)
        self._send(b"{}", "application/json")

    def log_message(self, *args):
        pass


LOCK = threading.Lock()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=list(PHASES), default="4")
    phase = PHASE = ap.parse_args().phase
    phase_dir, size, seed, FRAMES, with_guide = PHASES[phase]
    VALUES = (0, 1, 2) if phase in LEVELS else (0, 1)
    POOL = REPO_ROOT / "results" / phase_dir / "pool.json"
    OUT = REPO_ROOT / "results" / phase_dir / "labels_author.json"
    from scenesafe.safety.tags import TAGS, TAGS_OBVIOUS
    TAG_DEFS = TAGS_OBVIOUS if phase == "5d" else TAGS                       # 5d: the obvious definitions
    ITEMS = safety_items(size, seed) if phase.startswith("5") else items(size, seed, with_guide, phase in LEVELS)
    ALLOWED = {it["pair"] for it in ITEMS}
    print(f"{len(ITEMS)} pairs to label, {len(load_labels())} done so far. Open http://localhost:{PORT}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
