#!/usr/bin/env python3
"""Human-in-the-loop data game for Auralis corpora.

This standalone server serves a small browser UI for rating, correcting,
ranking, and turning raw corpus snippets into supervised data. It avoids any
third-party web framework and writes append-only JSONL feedback records.

Usage:
    python scripts/monitor/data_game_app.py --host 127.0.0.1 --port 8777
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import re
import threading
import time
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


WHITESPACE_RE = re.compile(r"\s+")


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def doc_hash(text: str) -> str:
    return hashlib.blake2b(normalize_text(text).encode("utf-8"), digest_size=12).hexdigest()


# The edu classifier scores (and the review pool stores) only the first N chars of
# each doc. Legacy pools were hard-cut at this cap mid-word; detect & clean that.
POOL_PREVIEW_CHARS = 2000


def clean_excerpt(text: str) -> str:
    """Trim a dangling partial word off a hard-cut preview and mark it as an
    excerpt, so it reads as 'shortened' rather than broken. No-op if already
    marked. Only for display — the dedup hash is computed on the raw text."""
    if text.endswith("[…]"):
        return text
    sp = text.rfind(" ")
    if sp > 0 and sp >= len(text) - 80:
        text = text[:sp]
    return text.rstrip() + " […]"


def safe_relative(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path)


@dataclass
class DataTask:
    task_id: str
    source_path: str
    source_line: int
    doc_hash: str
    text: str
    char_count: int
    word_count: int
    model_score: float | None = None   # edu classifier's predicted 0-5 (when reviewing a scored pool)
    truncated: bool = False            # preview is a word-boundary excerpt of a longer doc


class TaskStore:
    def __init__(
        self,
        root: Path,
        sources: list[Path],
        output: Path,
        queue_size: int,
        scan_lines: int,
        min_chars: int,
        max_chars: int,
        seed: int,
        pool: Path | None = None,
        boundary: float = 2.0,
    ) -> None:
        self.root = root
        self.sources = sources
        self.output = output
        self.queue_size = queue_size
        self.scan_lines = scan_lines
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.seed = seed
        self.pool = pool
        self.boundary = boundary
        self.lock = threading.Lock()
        self.tasks: list[DataTask] = []
        self.cursor = 0
        self.saved_count = 0
        self.skipped_count = 0
        self.reviewed_hashes = self._load_reviewed_hashes()
        self.last_saved: dict[str, Any] | None = None
        self.reload()

    def _load_reviewed_hashes(self) -> set[str]:
        seen: set[str] = set()
        if not self.output.exists():
            return seen
        try:
            with self.output.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    h = row.get("doc_hash")
                    if isinstance(h, str) and h:
                        seen.add(h)
                        self.saved_count += 1
        except OSError:
            pass
        return seen

    def _reload_from_pool(self) -> None:
        """Build the queue from a pre-scored JSONL pool (rows: text, model_score,
        source, source_line), sorted most-uncertain-first (closest to the
        classifier decision boundary). This is active learning: human time goes to
        the docs the model is least sure about, not random re-rating of raw text."""
        rows: list[DataTask] = []
        try:
            with self.pool.open("r", encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = normalize_text(str(r.get("text", "")))
                    if not (self.min_chars <= len(text) <= self.max_chars):
                        continue
                    h = doc_hash(text)            # hash on RAW pool text → dedup stays stable
                    if h in self.reviewed_hashes:
                        continue
                    # Generator marks truncation explicitly; legacy pools are inferred
                    # from the hard cap. Clean the display text but keep the raw hash.
                    truncated = bool(r.get("truncated"))
                    if r.get("truncated") is None and len(text) >= POOL_PREVIEW_CHARS:
                        truncated = True
                    display_text = clean_excerpt(text) if (truncated and not text.endswith("[…]")) else text
                    ms = r.get("model_score")
                    src = str(r.get("source") or r.get("source_path") or self.pool.name)
                    ln = int(r.get("source_line", i))
                    tid = hashlib.blake2b(f"{src}:{ln}:{h}".encode("utf-8"), digest_size=10).hexdigest()
                    rows.append(
                        DataTask(
                            task_id=tid, source_path=src, source_line=ln, doc_hash=h,
                            text=display_text, char_count=len(text), word_count=len(text.split()),
                            model_score=(float(ms) if ms is not None else None),
                            truncated=truncated,
                        )
                    )
        except OSError:
            rows = []
        rows.sort(key=lambda t: abs((t.model_score if t.model_score is not None else self.boundary) - self.boundary))
        with self.lock:
            self.tasks = rows[: self.queue_size]
            self.cursor = 0

    def reload(self) -> None:
        if self.pool is not None and self.pool.exists():
            self._reload_from_pool()
            return
        rng = random.Random(self.seed + int(time.time() // 3600))
        candidates: list[DataTask] = []
        for source in self.sources:
            if len(candidates) >= self.queue_size * 3:
                break
            if not source.exists():
                continue
            rel_source = safe_relative(self.root, source)
            try:
                with source.open("r", encoding="utf-8", errors="replace") as f:
                    for line_no, raw in enumerate(f, start=1):
                        if line_no > self.scan_lines:
                            break
                        text = normalize_text(raw)
                        if not (self.min_chars <= len(text) <= self.max_chars):
                            continue
                        h = doc_hash(text)
                        if h in self.reviewed_hashes:
                            continue
                        task_id = hashlib.blake2b(
                            f"{rel_source}:{line_no}:{h}".encode("utf-8"),
                            digest_size=10,
                        ).hexdigest()
                        candidates.append(
                            DataTask(
                                task_id=task_id,
                                source_path=rel_source,
                                source_line=line_no,
                                doc_hash=h,
                                text=text,
                                char_count=len(text),
                                word_count=len(text.split()),
                            )
                        )
            except OSError:
                continue
        rng.shuffle(candidates)
        with self.lock:
            self.tasks = candidates[: self.queue_size]
            self.cursor = 0

    def status(self) -> dict[str, Any]:
        with self.lock:
            remaining = max(0, len(self.tasks) - self.cursor)
            return {
                "queue_size": len(self.tasks),
                "remaining": remaining,
                "cursor": self.cursor,
                "saved_count": self.saved_count,
                "skipped_count": self.skipped_count,
                "reviewed_unique_docs": len(self.reviewed_hashes),
                "output": safe_relative(self.root, self.output),
                "sources": [safe_relative(self.root, p) for p in self.sources],
                "scan_lines": self.scan_lines,
                "pool": safe_relative(self.root, self.pool) if self.pool else None,
                "boundary": self.boundary,
                "last_saved": self.last_saved,
            }

    def next_task(self, mode: str) -> dict[str, Any]:
        with self.lock:
            if self.cursor >= len(self.tasks):
                return {"empty": True}
            task = self.tasks[self.cursor]
            payload = asdict(task)
            payload["mode"] = mode
            payload["empty"] = False
            if mode == "rank":
                other_idx = min(self.cursor + 1, len(self.tasks) - 1)
                if other_idx != self.cursor:
                    payload["other"] = asdict(self.tasks[other_idx])
            return payload

    def skip(self, task_id: str | None) -> None:
        with self.lock:
            if self.cursor < len(self.tasks):
                current = self.tasks[self.cursor]
                if task_id in (None, current.task_id):
                    self.cursor += 1
                    self.skipped_count += 1

    def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            current = self.tasks[self.cursor] if self.cursor < len(self.tasks) else None
            if current is None:
                raise ValueError("No task is available")
            if payload.get("task_id") != current.task_id:
                raise ValueError("Task id no longer matches the queue head")
            row = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "task_id": current.task_id,
                "mode": payload.get("mode", "quality"),
                "source_path": current.source_path,
                "source_line": current.source_line,
                "doc_hash": current.doc_hash,
                "text": current.text[:2400],
                "doc_excerpt": current.text[:1400],
                "model_score": current.model_score,
                "char_count": current.char_count,
                "word_count": current.word_count,
                "label": payload.get("label"),
                "score": payload.get("score"),
                "tags": payload.get("tags", []),
                "question": payload.get("question", ""),
                "answer": payload.get("answer", ""),
                "wrong_answer": payload.get("wrong_answer", ""),
                "correction": payload.get("correction", ""),
                "rewrite": payload.get("rewrite", ""),
                "notes": payload.get("notes", ""),
                "rank_choice": payload.get("rank_choice"),
                "other_doc_hash": payload.get("other_doc_hash"),
            }
            self.output.parent.mkdir(parents=True, exist_ok=True)
            with self.output.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            self.reviewed_hashes.add(current.doc_hash)
            self.saved_count += 1
            self.last_saved = row
            self.cursor += 2 if row["mode"] == "rank" and row.get("other_doc_hash") else 1
            return row


INDEX_HTML = r"""<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Auralis Datenwerkstatt</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0b1014;
      --panel: #101920;
      --panel-2: #15232b;
      --line: rgba(255,255,255,.1);
      --text: #eef6f4;
      --muted: #91a5a8;
      --cyan: #50d7d0;
      --green: #74df8b;
      --amber: #f0b75f;
      --red: #ff6b6b;
      --blue: #7eb7ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        linear-gradient(180deg, rgba(80,215,208,.08), transparent 340px),
        radial-gradient(circle at 100% 0%, rgba(126,183,255,.12), transparent 430px),
        var(--bg);
      color: var(--text);
      letter-spacing: 0;
    }
    button, textarea, input {
      font: inherit;
    }
    .app {
      display: grid;
      grid-template-columns: 260px minmax(720px, 1fr) 310px;
      gap: 14px;
      width: min(1720px, calc(100vw - 28px));
      margin: 0 auto;
      padding: 14px 0 28px;
      min-height: 100vh;
      align-items: start;
    }
    aside, main, .right {
      min-width: 0;
    }
    aside, .right {
      position: sticky;
      top: 14px;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 18px;
    }
    .mark {
      width: 38px;
      height: 38px;
      border: 1px solid rgba(80,215,208,.42);
      background: linear-gradient(145deg, rgba(80,215,208,.22), rgba(126,183,255,.12));
      border-radius: 7px;
      display: grid;
      place-items: center;
      color: var(--cyan);
      font-weight: 800;
    }
    h1, h2, h3, p { margin: 0; }
    h1 { font-size: 18px; line-height: 1.1; }
    .sub { color: var(--muted); font-size: 12px; margin-top: 4px; }
    .panel {
      border: 1px solid var(--line);
      background: rgba(16,25,32,.88);
      border-radius: 8px;
      padding: 12px;
      box-shadow: 0 14px 46px rgba(0,0,0,.2);
    }
    .stack { display: grid; gap: 12px; }
    .stats {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .metric {
      display: flex;
      min-width: 0;
      min-height: 78px;
      flex-direction: column;
      justify-content: space-between;
      gap: 8px;
      padding: 10px;
      border: 1px solid rgba(255,255,255,.08);
      border-radius: 8px;
      background: rgba(7,12,16,.42);
    }
    .metric.wide { grid-column: 1 / -1; min-height: 62px; }
    .metric span {
      color: var(--muted);
      font-size: 11px;
      overflow-wrap: anywhere;
    }
    .metric strong { font-size: 23px; line-height: 1; }
    .modebar {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      margin-bottom: 10px;
    }
    .modebar button, .score button, .pill, .primary, .ghost {
      border: 1px solid var(--line);
      background: rgba(21,35,43,.82);
      color: var(--text);
      border-radius: 7px;
      padding: 10px 12px;
      cursor: pointer;
      transition: border-color .15s, background .15s, transform .15s;
    }
    button:hover { border-color: rgba(80,215,208,.42); background: rgba(31,51,61,.92); }
    button:active { transform: translateY(1px); }
    .modebar button.active {
      border-color: rgba(80,215,208,.65);
      background: rgba(80,215,208,.18);
      color: #dffffb;
    }
    .doc-head {
      display: flex;
      gap: 10px;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 12px;
    }
    #meta, #hash {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .doc {
      white-space: pre-wrap;
      line-height: 1.55;
      font-size: 15px;
      min-height: 165px;
      max-height: 34vh;
      overflow: auto;
      padding: 16px;
      border: 1px solid rgba(255,255,255,.08);
      border-radius: 8px;
      background: rgba(7,12,16,.62);
      scrollbar-color: rgba(80,215,208,.35) transparent;
    }
    .doc.two {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      max-height: none;
      background: transparent;
      border: 0;
      padding: 0;
    }
    .snippet {
      border: 1px solid rgba(255,255,255,.08);
      border-radius: 8px;
      background: rgba(7,12,16,.62);
      padding: 16px;
      max-height: 45vh;
      overflow: auto;
      line-height: 1.58;
    }
    .rubric {
      color: var(--muted);
      font-size: 11.5px;
      line-height: 1.5;
      margin-top: 12px;
      padding: 8px 10px;
      border: 1px dashed rgba(255,255,255,.14);
      border-radius: 7px;
      background: rgba(7,12,16,.42);
    }
    .rubric b { color: var(--text); }
    .help {
      border: 1px solid rgba(80,215,208,.3);
      background: rgba(16,25,32,.7);
      border-radius: 8px;
      padding: 10px 14px;
      margin-bottom: 12px;
      font-size: 13px;
    }
    .help summary {
      cursor: pointer;
      font-weight: 700;
      color: var(--cyan);
      list-style: none;
    }
    .help summary::-webkit-details-marker { display: none; }
    .helpbody { color: var(--muted); line-height: 1.55; margin-top: 8px; }
    .helpbody b { color: var(--text); }
    .helpbody ol { margin: 6px 0 6px 18px; padding: 0; }
    .helpbody li { margin: 3px 0; }
    .example {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin: 8px 0;
    }
    .example > div {
      border: 1px solid rgba(255,255,255,.1);
      border-radius: 7px;
      padding: 9px 10px;
      background: rgba(7,12,16,.5);
      font-size: 12.5px;
      line-height: 1.5;
    }
    .example span { color: var(--cyan); font-size: 11px; font-weight: 700; }
    .hint { color: var(--muted); font-size: 12px; margin-top: 6px; }
    .tagshead { color: var(--muted); font-size: 12px; margin: 12px 0 -2px; }
    @media (max-width: 760px) { .example { grid-template-columns: 1fr; } }
    .score {
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 7px;
      margin-top: 10px;
    }
    .score button { min-height: 46px; padding: 7px 8px; }
    .score small { display: block; color: var(--muted); font-size: 10px; margin-top: 4px; }
    .score button.good { border-color: rgba(116,223,139,.4); }
    .score button.bad { border-color: rgba(255,107,107,.4); }
    .tags {
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
      margin: 10px 0;
    }
    .pill.active {
      border-color: rgba(126,183,255,.58);
      background: rgba(126,183,255,.18);
    }
    .ranksel {
      border-color: rgba(80,215,208,.85) !important;
      background: rgba(80,215,208,.28) !important;
      color: #eafffb;
    }
    .needhi {
      border-color: var(--amber) !important;
      box-shadow: 0 0 0 1px rgba(240,183,95,.5);
    }
    label {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
    }
    textarea, input {
      width: 100%;
      min-height: 64px;
      resize: vertical;
      border: 1px solid rgba(255,255,255,.1);
      border-radius: 8px;
      background: rgba(6,10,14,.74);
      color: var(--text);
      padding: 11px 12px;
      outline: none;
    }
    input { min-height: 42px; }
    textarea:focus, input:focus { border-color: rgba(80,215,208,.58); }
    .actions {
      display: flex;
      gap: 10px;
      justify-content: flex-end;
      margin-top: 10px;
    }
    .primary {
      background: linear-gradient(135deg, rgba(80,215,208,.25), rgba(116,223,139,.18));
      border-color: rgba(80,215,208,.6);
      font-weight: 700;
    }
    .ghost { color: var(--muted); }
    .last {
      font-size: 12px;
      color: var(--muted);
      white-space: pre-wrap;
      max-height: 36vh;
      overflow: auto;
      padding: 12px;
      background: rgba(7,12,16,.62);
      border: 1px solid rgba(255,255,255,.08);
      border-radius: 8px;
    }
    .status {
      min-height: 22px;
      color: var(--cyan);
      font-size: 13px;
      margin-top: 10px;
    }
    .hidden { display: none !important; }
    .hotkeys {
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
    }
    .hotkeys span {
      padding: 7px 8px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: rgba(7,12,16,.42);
    }
    .hk-note { color: var(--muted); font-size: 11px; line-height: 1.45; margin-top: 8px; }
    .hk-note b { color: var(--text); }
    @media (max-width: 1100px) {
      .app {
        width: calc(100vw - 20px);
        grid-template-columns: 1fr;
        padding: 10px 0 24px;
      }
      aside, .right { position: static; }
      .doc.two { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <div class="brand">
        <div class="mark">A</div>
        <div>
          <h1>Auralis Datenwerkstatt</h1>
          <div class="sub">Human feedback fuer bessere Trainingsdaten</div>
        </div>
      </div>
      <div class="panel stack">
        <div class="stats">
          <div class="metric"><span>Queue</span><strong id="remaining">-</strong></div>
          <div class="metric"><span>Gespeichert</span><strong id="saved">-</strong></div>
          <div class="metric"><span>Unique Docs</span><strong id="unique">-</strong></div>
          <div class="metric"><span>Modus</span><strong id="activeMode">-</strong></div>
          <div class="metric wide"><span>Output</span><span id="output">-</span></div>
          <div class="metric wide"><span>Diese Session</span><span id="session">0 gespeichert</span></div>
        </div>
      </div>
      <div style="height:12px"></div>
      <div class="panel">
        <h3 style="font-size:14px;margin-bottom:10px">Hotkeys</h3>
        <div class="hotkeys">
          <span>1-6 Bewerten</span><span>Ctrl+Enter Speichern</span>
          <span>N Skip</span><span>R Reload</span>
        </div>
        <div class="hk-note">Tasten <b>1–6 / N / R</b> wirken nur, wenn KEIN Textfeld fokussiert ist (sonst werden sie getippt). <b>Ctrl+Enter</b> speichert immer — auch aus einem Textfeld heraus.</div>
      </div>
    </aside>

    <main>
      <details class="help" open>
        <summary>Anleitung lesen — was tue ich hier?</summary>
        <div class="helpbody">
          <p><b>Ziel:</b> Trainingsdaten besser machen. Du siehst je ein deutsches Textstueck mit „Modell-Tipp" (dem 0–5-Score des Edu-Klassifikators) und hilfst auf zwei Arten: bewerten, ob es gutes Lernmaterial ist — und wo es sich lohnt, eine bessere Version schreiben.</p>
          <p><b>So gehst du vor (Modus „Qualitaet"):</b></p>
          <ol>
            <li><b>Bewerten 0–5</b> (Tasten 1–6) nach der Rubrik unten. Steht ein „Modell-Tipp" da, nur bestaetigen oder korrigieren.</li>
            <li><b>Maengel markieren:</b> was ist faul? z.B. <i>boilerplate</i> (Werbung/Navigation), <i>too_shallow</i> (zu seicht), <i>needs_rewrite</i> (gesprochen/holprig), <i>unsafe</i>, <i>duplicate</i>.</li>
            <li><b>Optional umschreiben:</b> Inhalt ok, aber Stil holprig (typisch fuer Reden/Transkripte)? Dann schreib eine saubere Version ins Feld „Bessere Version".</li>
            <li><b>Speichern</b> mit Ctrl+Enter. Unklar? „Skip" (Taste N).</li>
          </ol>
          <p><b>Vorlage — so soll eine Umschreibung aussehen:</b></p>
          <div class="example">
            <div><span>VORHER (holprig, gesprochen)</span><br>„…Werbung von Online-Glücksspielen verbieten sollte, die junge Menschen als Zielgruppe haben. In der Tschechischen Republik hängt man da leider hinten an. Man hängt da zurück, was die EU betrifft. Denn hier gibt es gar keine Beschränkungen für solche Werbung…"</div>
            <div><span>NACHHER (klar, geschrieben)</span><br>„Werbung für Online-Glücksspiele, die sich an junge Menschen als Zielgruppe richtet, sollte verboten werden. Die Tschechische Republik hinkt hier im EU-Vergleich leider hinterher, da es dort keinerlei Beschränkungen für solche Werbung gibt…"</div>
          </div>
          <p class="hint">Modi oben: <b>Qualitaet</b> = bewerten + maengeln + umschreiben (das Wichtigste fuer die Datenqualitaet). „QA bauen" / „Korrigieren" / „A/B Rank" sind fuer die spaetere Instruction-Phase.</p>
        </div>
      </details>
      <div class="modebar">
        <button data-mode="quality" class="active">Qualitaet</button>
        <button data-mode="qa">QA bauen</button>
        <button data-mode="fix">Korrigieren</button>
        <button data-mode="rank">A/B Rank</button>
      </div>

      <section class="panel">
        <div class="doc-head">
          <span id="meta">lade...</span>
          <span id="hash"></span>
        </div>
        <div id="doc" class="doc"></div>
        <div id="rankDoc" class="doc two hidden"></div>

        <div id="qualityControls">
          <div class="rubric">Edu-Rubrik: <b>0</b> Müll/kaputt · <b>1</b> fast nur Boilerplate/Werbung · <b>2</b> berührt ein Thema, aber dünn/unvollständig · <b>3</b> vermittelt nachvollziehbares Wissen, sachlich (lohnt sich) · <b>4</b> lehrbuch-/lexikonartig · <b>5</b> herausragend lehrreich.<br><b>Grenze 2↔3:</b> Würde ein Lernender hier etwas Konkretes mitnehmen? Ja → mind. 3. Bei „Modell-Tipp" nur bestätigen/korrigieren.</div>
          <div class="score">
            <button class="bad" data-score="0" data-label="trash" title="Kaputt/sinnlos: Zeichensalat, abgeschnitten, reiner Unsinn.">0<small>Muellsatz</small></button>
            <button class="bad" data-score="1" data-label="weak" title="Fast nur Werbung/Navigation/Floskeln; kaum Inhalt.">1<small>schwach</small></button>
            <button data-score="2" data-label="thin" title="Berührt ein Thema, bleibt aber dünn/unvollständig — Lernender nimmt wenig mit.">2<small>duenn</small></button>
            <button data-score="3" data-label="usable" title="Vermittelt nachvollziehbares Wissen, sachlich — lohnt sich (Behalten-Grenze).">3<small>brauchbar</small></button>
            <button class="good" data-score="4" data-label="good" title="Lehrbuch-/lexikonartig: klar strukturiert, gehaltvoll.">4<small>gut</small></button>
            <button class="good" data-score="5" data-label="excellent" title="Herausragend lehrreich: tief, präzise, frei von Boilerplate.">5<small>lehrreich</small></button>
          </div>
          <div class="tagshead">Eigenschaften &amp; Maengel markieren (gespeichert wird der englische Schluessel)</div>
          <div class="tags" id="tags"></div>
          <label style="margin-top:10px">Bessere Version (optional)
            <textarea id="rewrite" placeholder="Nur wenn der Inhalt ok ist, aber der Stil holprig/gesprochen: schreib den Text sauber & gut lesbar um (siehe Vorlage in der Anleitung). Sonst leer lassen."></textarea>
          </label>
        </div>

        <div id="qaControls" class="stack hidden" style="margin-top:14px">
          <label>Frage aus dem Text<input id="question" placeholder="Was sollte Auralis hier lernen?" /></label>
          <label>Ideale Antwort<textarea id="answer" placeholder="Kurze, korrekte Antwort in natuerlicher Sprache"></textarea></label>
        </div>

        <div id="fixControls" class="stack hidden" style="margin-top:14px">
          <label>Typische falsche Antwort<textarea id="wrongAnswer" placeholder="Was wuerde ein schwaches Modell hier falsch sagen?"></textarea></label>
          <label>Korrekte Antwort / Reparatur<textarea id="correction" placeholder="Was soll es stattdessen sicher lernen?"></textarea></label>
        </div>

        <div id="rankControls" class="hidden" style="margin-top:14px">
          <div class="actions" style="justify-content:center">
            <button class="primary" id="rankA">A ist besser</button>
            <button class="primary" id="rankB">B ist besser</button>
          </div>
        </div>

        <label style="margin-top:14px">Notiz<textarea id="notes" placeholder="Optional: Warum behalten, droppen oder umschreiben?"></textarea></label>
        <div class="actions">
          <button class="ghost" id="reload">Queue neu laden</button>
          <button class="ghost" id="skip">Skip</button>
          <button class="primary" id="save">Speichern</button>
        </div>
        <div class="status" id="status"></div>
      </section>
    </main>

    <aside class="right">
      <div class="panel stack">
        <h3 style="font-size:14px">Letzter Save</h3>
        <div id="last" class="last">Noch nichts gespeichert.</div>
      </div>
      <div style="height:12px"></div>
      <div class="panel stack">
        <h3 style="font-size:14px">Quellen</h3>
        <div id="sources" class="last"></div>
      </div>
    </aside>
  </div>

<script>
const tagNames = ["keep", "edu", "fact", "german", "style", "boilerplate", "duplicate", "needs_rewrite", "too_shallow", "unsafe", "legal", "medical", "financial", "claims_need_verification"];
// German display labels only — the stored/exported tag value stays the English canonical key.
const tagLabels = {
  keep: "behalten", edu: "Bildungswert", fact: "Faktenwissen", german: "gutes Deutsch",
  style: "guter Stil", boilerplate: "Boilerplate/Werbung", duplicate: "Dublette",
  needs_rewrite: "umschreiben", too_shallow: "oberflächlich", unsafe: "unsicher",
  legal: "Recht", medical: "Medizin", financial: "Finanzen", claims_need_verification: "Claim prüfen",
};
let mode = "quality";
let current = null;
let selectedScore = null;
let selectedLabel = null;
let tags = new Set();
let selectedRank = null;
let session = { saves: 0, score: {}, tag: {} };

function $(id) { return document.getElementById(id); }

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, ch => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;" }[ch]));
}

function setStatus(msg, bad=false) {
  $("status").textContent = msg;
  $("status").style.color = bad ? "var(--red)" : "var(--cyan)";
}

function renderTags() {
  const wrap = $("tags");
  wrap.innerHTML = "";
  for (const t of tagNames) {
    const b = document.createElement("button");
    b.className = "pill" + (tags.has(t) ? " active" : "");
    b.textContent = tagLabels[t] || t;
    b.title = t;  // canonical English tag — this is what gets saved
    b.onclick = () => { tags.has(t) ? tags.delete(t) : tags.add(t); renderTags(); };
    wrap.appendChild(b);
  }
  // Highlight "Bessere Version" when a rewrite is flagged (#4).
  $("rewrite").classList.toggle("needhi", tags.has("needs_rewrite"));
}

function renderSession() {
  const parts = [];
  for (const s of [0,1,2,3,4,5]) if (session.score[s]) parts.push(`${s}:${session.score[s]}`);
  const rw = session.tag["needs_rewrite"] ? ` · ${session.tag["needs_rewrite"]}× rewrite` : "";
  $("session").textContent = `${session.saves} gespeichert` + (parts.length ? ` (${parts.join(" ")})` : "") + rw;
}

function resetInputs() {
  selectedScore = null;
  selectedLabel = null;
  selectedRank = null;
  tags = new Set();
  for (const el of document.querySelectorAll(".score button")) el.classList.remove("active");
  $("rankA").classList.remove("ranksel");
  $("rankB").classList.remove("ranksel");
  $("question").value = "";
  $("answer").value = "";
  $("wrongAnswer").value = "";
  $("correction").value = "";
  $("rewrite").value = "";
  $("notes").value = "";
  renderTags();
}

function showModeControls() {
  $("activeMode").textContent = ({quality: "Rating", qa: "QA", fix: "Fix", rank: "A/B"}[mode] || mode);
  $("qualityControls").classList.toggle("hidden", mode !== "quality");
  $("qaControls").classList.toggle("hidden", mode !== "qa");
  $("fixControls").classList.toggle("hidden", mode !== "fix");
  $("rankControls").classList.toggle("hidden", mode !== "rank");
  $("doc").classList.toggle("hidden", mode === "rank");
  $("rankDoc").classList.toggle("hidden", mode !== "rank");
}

async function refreshStatus() {
  const res = await fetch("/api/status");
  const st = await res.json();
  $("remaining").textContent = st.remaining;
  $("saved").textContent = st.saved_count;
  $("unique").textContent = st.reviewed_unique_docs;
  $("output").textContent = st.output;
  $("sources").textContent = st.sources.join("\n");
  if (st.last_saved) $("last").textContent = JSON.stringify(st.last_saved, null, 2);
}

function renderTask(t) {
  current = t;
  resetInputs();
  showModeControls();
  if (t.empty) {
    $("doc").textContent = "Queue leer. Lade neu oder erhoehe --scan-lines.";
    $("rankDoc").innerHTML = "";
    $("meta").textContent = "keine Aufgabe";
    $("hash").textContent = "";
    return;
  }
  $("meta").textContent = `${t.source_path}:${t.source_line} · ${t.word_count} Woerter · ${t.char_count} Zeichen`
    + (t.model_score != null ? ` · Modell-Tipp: ${Number(t.model_score).toFixed(1)}` : "")
    + (t.truncated ? " · Vorschau (Modell bewertet nur den Anfang; volles Dok bleibt im Training)" : "");
  $("hash").textContent = t.doc_hash;
  $("doc").textContent = t.text;
  if (mode === "rank") {
    const other = t.other;
    $("rankDoc").innerHTML = `
      <div class="snippet"><b>A</b><br><br>${esc(t.text)}</div>
      <div class="snippet"><b>B</b><br><br>${esc(other ? other.text : "Keine zweite Probe in der Queue.")}</div>`;
  }
}

async function nextTask() {
  const res = await fetch(`/api/next?mode=${encodeURIComponent(mode)}`);
  renderTask(await res.json());
  await refreshStatus();
  setStatus("");
}

let submitting = false;
async function submit(extra={}) {
  if (!current || current.empty || submitting) return;   // re-entrancy guard: no double-save / key-autorepeat race
  submitting = true;
  try {
    if (mode === "quality" && selectedScore === null) {
      setStatus("Erst eine Bewertung 0-5 waehlen (dann Tags / Notiz / Bessere Version, dann Speichern).", true);
      return;
    }
    if (mode === "rank" && !selectedRank && !extra.rank_choice) {
      setStatus("Erst A oder B waehlen, dann Speichern.", true);
      return;
    }
    if (mode === "rank") {
      extra = {
        rank_choice: extra.rank_choice || selectedRank,
        other_doc_hash: current.other ? current.other.doc_hash : null,
        ...extra,
      };
    }
    if (tags.has("needs_rewrite") && !$("rewrite").value.trim()) {
      if (!confirm("'umschreiben' ist markiert, aber 'Bessere Version' ist leer. Trotzdem speichern?")) {
        return;
      }
    }
    const payload = {
      task_id: current.task_id,
      mode,
      score: selectedScore,
      label: selectedLabel,
      tags: Array.from(tags),
      question: $("question").value.trim(),
      answer: $("answer").value.trim(),
      wrong_answer: $("wrongAnswer").value.trim(),
      correction: $("correction").value.trim(),
      rewrite: $("rewrite").value.trim(),
      notes: $("notes").value.trim(),
      ...extra
    };
    const res = await fetch("/api/submit", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    const body = await res.json();
    if (!res.ok) {
      setStatus(body.error || "Speichern fehlgeschlagen", true);
      return;
    }
    $("last").textContent = JSON.stringify(body.saved, null, 2);
    session.saves += 1;
    if (selectedScore !== null) session.score[selectedScore] = (session.score[selectedScore] || 0) + 1;
    for (const t of tags) session.tag[t] = (session.tag[t] || 0) + 1;
    renderSession();
    const tagStr = Array.from(tags).join(", ") || "–";
    const rwStr = $("rewrite").value.trim() ? " · +Bessere Version" : "";
    setStatus(`Gespeichert: score=${selectedScore ?? "–"} · tags=${tagStr}${rwStr}`);
    await nextTask();
  } finally {
    submitting = false;
  }
}

async function skip() {
  if (!current || current.empty) return;
  await fetch("/api/skip", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ task_id: current.task_id })
  });
  await nextTask();
}

async function reloadQueue() {
  setStatus("Queue wird neu geladen...");
  await fetch("/api/reload", { method: "POST" });
  await nextTask();
}

document.querySelectorAll(".modebar button").forEach(btn => {
  btn.onclick = async () => {
    mode = btn.dataset.mode;
    document.querySelectorAll(".modebar button").forEach(b => b.classList.toggle("active", b === btn));
    await nextTask();
  };
});

document.querySelectorAll(".score button").forEach(btn => {
  btn.onclick = () => {
    // Select only — do NOT auto-save, so Tags / Notiz / Bessere Version can be
    // added first. Save explicitly via the button or Ctrl+Enter.
    selectedScore = Number(btn.dataset.score);
    selectedLabel = btn.dataset.label;
    document.querySelectorAll(".score button").forEach(b => b.classList.toggle("active", b === btn));
    if (selectedScore <= 1) tags.add("drop");
    if (selectedScore >= 4) tags.add("keep");
    renderTags();
    setStatus(`Bewertung ${selectedScore} gewaehlt — ergaenze Tags / Notiz / Bessere Version, dann Speichern (Ctrl+Enter).`);
  };
});

$("save").onclick = () => submit();
$("skip").onclick = skip;
$("reload").onclick = reloadQueue;
function selectRank(choice) {
  selectedRank = choice;
  $("rankA").classList.toggle("ranksel", choice === "A");
  $("rankB").classList.toggle("ranksel", choice === "B");
  setStatus(`${choice} gewaehlt — optional Notiz, dann Speichern (Ctrl+Enter).`);
}
$("rankA").onclick = () => selectRank("A");
$("rankB").onclick = () => selectRank("B");

document.addEventListener("keydown", (e) => {
  if (e.ctrlKey && e.key === "Enter") { e.preventDefault(); submit(); return; }
  if (e.target && ["TEXTAREA", "INPUT"].includes(e.target.tagName)) return;
  if (e.key.toLowerCase() === "n") skip();
  if (e.key.toLowerCase() === "r") reloadQueue();
  const n = Number(e.key);
  if (mode === "quality" && n >= 1 && n <= 6) {
    const btn = document.querySelector(`.score button[data-score="${n - 1}"]`);
    if (btn) btn.click();
  }
});

renderTags();
renderSession();
nextTask().catch(err => setStatus(String(err), true));
</script>
</body>
</html>
"""


class DataGameHandler(BaseHTTPRequestHandler):
    server_version = "AuralisDataGame/0.1"

    @property
    def store(self) -> TaskStore:
        return self.server.store  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(HTTPStatus.OK, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/status":
            self._json(HTTPStatus.OK, self.store.status())
            return
        if parsed.path == "/api/next":
            mode = parse_qs(parsed.query).get("mode", ["quality"])[0]
            self._json(HTTPStatus.OK, self.store.next_task(mode))
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/submit":
                row = self.store.submit(self._read_json())
                self._json(HTTPStatus.OK, {"saved": row, "status": self.store.status()})
                return
            if parsed.path == "/api/skip":
                payload = self._read_json()
                self.store.skip(payload.get("task_id"))
                self._json(HTTPStatus.OK, {"ok": True, "status": self.store.status()})
                return
            if parsed.path == "/api/reload":
                self.store.reload()
                self._json(HTTPStatus.OK, {"ok": True, "status": self.store.status()})
                return
        except Exception as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})


def parse_args() -> argparse.Namespace:
    root = repo_root_from_here()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8777)
    p.add_argument("--root", type=Path, default=root)
    p.add_argument("--source", type=Path, action="append", default=None)
    p.add_argument("--output", type=Path, default=root / "data" / "human_feedback" / "auralis_data_game_v1.jsonl")
    p.add_argument("--queue-size", type=int, default=300)
    p.add_argument("--scan-lines", type=int, default=20000)
    p.add_argument("--min-chars", type=int, default=320)
    p.add_argument("--max-chars", type=int, default=3600)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--pool", type=Path, default=None,
                   help="Pre-scored JSONL (rows: text, model_score, source) from "
                        "score_corpus_edu.py --review-pool. When set, the queue is built from this "
                        "and sorted most-uncertain-first (active learning), and the classifier's "
                        "score is shown next to each doc.")
    p.add_argument("--boundary", type=float, default=2.0,
                   help="Classifier decision boundary used to rank pool uncertainty (default 2.0).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    sources = args.source or [root / "data" / "training" / "curated_40b" / "german.txt"]
    sources = [(root / s).resolve() if not s.is_absolute() else s.resolve() for s in sources]
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    pool = None
    if args.pool is not None:
        pool = (root / args.pool).resolve() if not args.pool.is_absolute() else args.pool.resolve()
    store = TaskStore(
        root=root,
        sources=sources,
        output=output,
        queue_size=args.queue_size,
        scan_lines=args.scan_lines,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        seed=args.seed,
        pool=pool,
        boundary=args.boundary,
    )
    server = ThreadingHTTPServer((args.host, args.port), DataGameHandler)
    server.store = store  # type: ignore[attr-defined]
    print(f"Auralis Datenwerkstatt listening on http://{args.host}:{args.port}")
    print(f"Output: {output}")
    server.serve_forever()


if __name__ == "__main__":
    main()
