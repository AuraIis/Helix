#!/usr/bin/env python3
"""Auralis Dataset Market local web app.

The app helps discover public Hugging Face datasets, score whether they fit a
training goal, assemble a data mix, and generate a first cleaning/download plan.
It intentionally uses the Python standard library for serving so it can run on
the local Windows box without adding a frontend stack.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import re
import subprocess
import sys
import textwrap
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = Path("I:/KI/Auralis_datasets")
DEFAULT_MAX_RSS_GB = 8.0
MAX_REQUEST_BODY_BYTES = 1_000_000
MAX_URL_RESPONSE_BYTES = 8_000_000
MAX_SAMPLE_CHARS = 4_000
MAX_PREVIEW_FIELD_CHARS = 2_000


PERMISSIVE_LICENSES = {
    "apache-2.0",
    "mit",
    "bsd-3-clause",
    "bsd-2-clause",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "cc0-1.0",
    "odc-by",
}
RISKY_LICENSES = {"unknown", "other", "non-commercial", "cc-by-nc", "cc-by-nc-sa"}
BASE_BAD_HINTS = ("sft", "instruction", "chat", "rlhf", "dpo", "preference", "alignment")
BASE_GOOD_HINTS = ("pretrain", "corpus", "web", "wiki", "text", "clean", "edu", "german", "de")
MATH_HINTS = ("math", "gsm", "competition", "olympiad", "proof", "problem")
CODE_HINTS = ("code", "python", "programming", "github", "stack")
TEXT_FILE_EXTS = (".txt", ".jsonl", ".json", ".parquet", ".arrow", ".csv", ".tsv")
BAD_SAMPLE_HINTS = (
    "<html",
    "</div>",
    "cookie",
    "newsletter",
    "javascript",
    "subscribe",
    "<|im_start|>",
    "### instruction",
    "### response",
)
GERMAN_SIGNAL_WORDS = (
    "der",
    "die",
    "das",
    "und",
    "ist",
    "nicht",
    "mit",
    "eine",
    "einer",
    "werden",
    "deutsch",
)


@dataclass
class DatasetCard:
    id: str
    downloads: int = 0
    likes: int = 0
    tags: list[str] = field(default_factory=list)
    license: str = "unknown"
    languages: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    size_tags: list[str] = field(default_factory=list)
    gated: bool = False
    private: bool = False
    last_modified: str = ""
    score: float = 0.0
    score_label: str = "unknown"
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cleaning_route: str = "auralis_structure"


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _tags(info: Any) -> list[str]:
    raw = _field(info, "tags", []) or []
    return sorted({str(t).lower() for t in raw if t is not None})


def _card_data(info: Any) -> dict[str, Any]:
    data = _field(info, "cardData", None) or _field(info, "card_data", None) or {}
    return data if isinstance(data, dict) else {}


def _find_tag_values(tags: list[str], prefix: str) -> list[str]:
    values: list[str] = []
    for tag in tags:
        if tag.startswith(prefix + ":"):
            values.append(tag.split(":", 1)[1])
    return sorted(set(values))


def _license(info: Any, tags: list[str]) -> str:
    data = _card_data(info)
    lic = data.get("license") or data.get("licenses")
    if isinstance(lic, list) and lic:
        lic = lic[0]
    if isinstance(lic, str) and lic:
        return lic.lower()
    tag_values = _find_tag_values(tags, "license")
    return tag_values[0] if tag_values else "unknown"


def _last_modified(info: Any) -> str:
    for name in ("lastModified", "last_modified", "createdAt", "created_at"):
        value = _field(info, name, None)
        if value:
            return str(value)
    return ""


def _cleaning_route(card: DatasetCard, goal: str) -> str:
    hay = " ".join([card.id.lower(), *card.tags])
    if goal == "math" or any(h in hay for h in MATH_HINTS):
        return "math_structure_min_language"
    if goal == "code" or any(h in hay for h in CODE_HINTS):
        return "code_preserve_structure"
    if any(h in hay for h in ("html", "commoncrawl", "web", "crawl")):
        return "extract_then_auralis"
    return "auralis_structure"


def _score(info: Any, *, goal: str, language: str) -> DatasetCard:
    tags = _tags(info)
    card_id = str(_field(info, "id", "") or _field(info, "datasetId", "") or "")
    downloads = _safe_int(_field(info, "downloads", 0))
    likes = _safe_int(_field(info, "likes", 0))
    languages = _find_tag_values(tags, "language")
    tasks = _find_tag_values(tags, "task_categories")
    size_tags = _find_tag_values(tags, "size_categories")
    license_name = _license(info, tags)
    hay = " ".join([card_id.lower(), *tags])

    score = 25.0
    reasons: list[str] = []
    warnings: list[str] = []

    if downloads:
        score += min(18.0, math.log10(downloads + 1) * 4.0)
        reasons.append(f"{downloads:,} downloads")
    if likes:
        score += min(12.0, math.log10(likes + 1) * 4.0)
        reasons.append(f"{likes:,} likes")

    if license_name in PERMISSIVE_LICENSES:
        score += 18.0
        reasons.append(f"permissive license: {license_name}")
    elif license_name in RISKY_LICENSES or license_name == "unknown":
        score -= 8.0
        warnings.append(f"license needs review: {license_name}")
    else:
        score += 4.0
        warnings.append(f"license check required: {license_name}")

    if language:
        lang_norm = language.lower()
        if lang_norm in languages or f"language:{lang_norm}" in tags or lang_norm in hay:
            score += 12.0
            reasons.append(f"language match: {language}")
        elif languages:
            score -= 10.0
            warnings.append(f"language mismatch: {', '.join(languages[:4])}")

    if goal == "base_pretrain":
        good_hits = [h for h in BASE_GOOD_HINTS if h in hay]
        bad_hits = [h for h in BASE_BAD_HINTS if h in hay]
        score += min(15.0, len(good_hits) * 4.0)
        score -= min(22.0, len(bad_hits) * 7.0)
        if good_hits:
            reasons.append("base-pretrain hints: " + ", ".join(good_hits[:5]))
        if bad_hits:
            warnings.append("post-training/template hints: " + ", ".join(bad_hits[:5]))
    elif goal == "sft":
        hits = [h for h in BASE_BAD_HINTS if h in hay]
        score += min(18.0, len(hits) * 5.0)
        if hits:
            reasons.append("SFT hints: " + ", ".join(hits[:5]))
    elif goal == "math":
        hits = [h for h in MATH_HINTS if h in hay]
        score += min(22.0, len(hits) * 6.0)
        if hits:
            reasons.append("math hints: " + ", ".join(hits[:5]))
    elif goal == "code":
        hits = [h for h in CODE_HINTS if h in hay]
        score += min(22.0, len(hits) * 6.0)
        if hits:
            reasons.append("code hints: " + ", ".join(hits[:5]))

    gated = bool(_field(info, "gated", False))
    private = bool(_field(info, "private", False))
    if gated:
        score -= 12.0
        warnings.append("gated dataset")
    if private:
        score -= 30.0
        warnings.append("private dataset")

    score = round(max(0.0, min(100.0, score)), 1)
    if score >= 78:
        label = "strong"
    elif score >= 58:
        label = "usable"
    elif score >= 38:
        label = "risky"
    else:
        label = "weak"

    card = DatasetCard(
        id=card_id,
        downloads=downloads,
        likes=likes,
        tags=tags[:80],
        license=license_name,
        languages=languages,
        tasks=tasks,
        size_tags=size_tags,
        gated=gated,
        private=private,
        last_modified=_last_modified(info),
        score=score,
        score_label=label,
        reasons=reasons[:8],
        warnings=warnings[:8],
    )
    card.cleaning_route = _cleaning_route(card, goal)
    return card


def _shorten(text: str, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _trim_text(text: str, limit: int = MAX_SAMPLE_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated]"


def _trim_json_value(value: Any, limit: int = MAX_PREVIEW_FIELD_CHARS) -> Any:
    if isinstance(value, str):
        return _trim_text(value, limit)
    if isinstance(value, list):
        return [_trim_json_value(item, limit) for item in value[:20]]
    if isinstance(value, dict):
        return {str(k): _trim_json_value(v, limit) for k, v in list(value.items())[:40]}
    return value


def _row_text(row: Any) -> str:
    if isinstance(row, str):
        return _trim_text(row)
    if not isinstance(row, dict):
        return ""
    preferred = ("text", "content", "document", "article", "body", "prompt", "response", "completion")
    parts: list[str] = []
    for key in preferred:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(_trim_text(value))
    if parts:
        return "\n".join(parts)
    for value in row.values():
        if isinstance(value, str) and len(value.strip()) >= 40:
            parts.append(_trim_text(value))
        elif isinstance(value, list):
            parts.extend(_trim_text(v) for v in value[:20] if isinstance(v, str) and len(v.strip()) >= 40)
    return _trim_text("\n".join(parts[:4]))


def _rss_bytes() -> int | None:
    if os.name == "nt":
        class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        if ok:
            return int(counters.PrivateUsage or counters.WorkingSetSize)
        return None
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return None


def _guard_memory(max_rss_gb: float) -> None:
    rss = _rss_bytes()
    if rss is None:
        return
    max_bytes = int(max_rss_gb * 1024**3)
    if rss > max_bytes:
        raise MemoryError(f"Dataset Market memory guard tripped: rss={rss / 1024**3:.1f} GB limit={max_rss_gb:.1f} GB")


def _sample_quality(samples: list[str], *, goal: str, language: str) -> dict[str, Any]:
    clean = [s.strip() for s in samples if s and s.strip()]
    if not clean:
        return {
            "score_delta": -10,
            "sample_count": 0,
            "avg_chars": 0,
            "estimated_keep_rate": 0.0,
            "signals": [],
            "warnings": ["no readable text samples found"],
            "examples": [],
        }

    avg_chars = sum(len(s) for s in clean) / len(clean)
    lower = "\n".join(clean).lower()
    original = "\n".join(clean)
    bad_hits = sum(1 for hint in BAD_SAMPLE_HINTS if hint in lower)
    url_ratio = len(re.findall(r"https?://|www\.", lower)) / max(1, len(clean))
    html_ratio = len(re.findall(r"<[a-z][^>]{0,80}>", lower)) / max(1, len(clean))
    german_hits = sum(len(re.findall(rf"\b{re.escape(word)}\b", lower)) for word in GERMAN_SIGNAL_WORDS)
    code_chars = sum(lower.count(ch) for ch in "{}();=<>")
    template_hits = len(re.findall(r"###\s*(frage|antwort|instruction|response|aufgabe)|<\|im_", lower))
    ocr_pattern_hits = len(
        re.findall(
            r"\b\w{1,3}\^\w{1,4}\b|\^[A-Za-z]|[A-Za-z][A-Z]{2,}[a-z]|[A-Za-z]{2,}\d",
            original,
        )
    )
    allowed_punctuation = set(".,;:!?()\"'/%+-=[]{}<>_@#&*|\\\n\r\t ")
    allowed_punctuation.update("•„“”‚‘’»«–—…€§©®™·")
    strange_chars = sum(1 for ch in original if not ch.isalnum() and ch not in allowed_punctuation)
    ocr_noise_hits = ocr_pattern_hits + strange_chars

    score_delta = 0
    signals: list[str] = []
    warnings: list[str] = []

    if avg_chars >= 650:
        score_delta += 10
        signals.append("long-form text samples")
    elif avg_chars >= 220:
        score_delta += 5
        signals.append("medium-length text samples")
    else:
        score_delta -= 8
        warnings.append("samples are short for base pretraining")

    if language.lower() == "de":
        if german_hits >= 30:
            score_delta += 8
            signals.append("strong German language signal")
        elif german_hits >= 8:
            score_delta += 3
            signals.append("some German language signal")
        elif goal not in {"math", "code"}:
            score_delta -= 10
            warnings.append("weak German language signal in samples")

    if bad_hits:
        score_delta -= min(18, bad_hits * 5)
        warnings.append(f"boilerplate/template hints in samples: {bad_hits}")
    if ocr_noise_hits >= max(4, len(clean) // 2):
        score_delta -= min(20, 6 + ocr_noise_hits)
        warnings.append(f"OCR or character-noise hints in samples: {ocr_noise_hits}")
    if url_ratio > 0.5 or html_ratio > 0.35:
        score_delta -= 12
        warnings.append("samples look web-noisy or HTML-heavy")
    if template_hits and goal == "base_pretrain":
        score_delta -= 10
        warnings.append("instruction/chat markers found in base-pretrain candidate")
    if goal == "code" and code_chars > 40:
        score_delta += 8
        signals.append("code-like structure found")
    if goal == "math" and re.search(r"\d+\s*[+\-*/=]|\\frac|\\sum|proof|beweis", lower):
        score_delta += 8
        signals.append("math/problem structure found")

    estimated_keep_rate = 0.86
    estimated_keep_rate -= min(0.35, bad_hits * 0.06)
    estimated_keep_rate -= min(0.30, ocr_noise_hits * 0.035)
    estimated_keep_rate -= min(0.25, html_ratio * 0.35)
    if avg_chars < 120:
        estimated_keep_rate -= 0.18
    estimated_keep_rate = round(max(0.05, min(0.98, estimated_keep_rate)), 2)

    return {
        "score_delta": score_delta,
        "sample_count": len(clean),
        "avg_chars": round(avg_chars, 1),
        "estimated_keep_rate": estimated_keep_rate,
        "signals": signals[:8],
        "warnings": warnings[:8],
        "examples": [_shorten(s) for s in clean[:3]],
    }


def _final_verdict(score: float, warnings: list[str]) -> str:
    if score >= 82 and not warnings:
        return "Sehr guter Kandidat fuer den Mix."
    if score >= 72:
        return "Guter Kandidat, aber vor Full-Download sample-clean testen."
    if score >= 55:
        return "Brauchbar mit Risiko. Nur mit strenger Pipeline aufnehmen."
    if score >= 38:
        return "Schwach. Eher nur fuer Spezialanteile oder nach manueller Pruefung."
    return "Nicht empfehlen fuer den Hauptmix."


def _terms(text: str) -> list[str]:
    return [part.strip().lower() for part in re.split(r"[,;\n]+", text or "") if part.strip()]


def _matches_terms(card: DatasetCard, terms: list[str], *, require_all: bool) -> bool:
    if not terms:
        return True
    hay = " ".join([card.id.lower(), card.license.lower(), card.cleaning_route.lower(), *card.tags, *card.reasons, *card.warnings])
    checks = [term in hay for term in terms]
    return all(checks) if require_all else any(checks)


def _hf_json(path: str, params: dict[str, str], timeout: int = 18) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    url = f"https://datasets-server.huggingface.co/{path}?{query}"
    request = urllib.request.Request(url, headers={"user-agent": "auralis-dataset-market/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read(MAX_URL_RESPONSE_BYTES + 1)
        if len(raw) > MAX_URL_RESPONSE_BYTES:
            raise RuntimeError(f"HF datasets-server response too large for safe preview: >{MAX_URL_RESPONSE_BYTES:,} bytes")
        return json.loads(raw.decode("utf-8", errors="ignore"))


def _hf_first_rows(dataset_id: str, limit: int = 8) -> tuple[list[str], str]:
    splits = _hf_json("splits", {"dataset": dataset_id})
    split_rows = splits.get("splits") or []
    chosen = None
    for row in split_rows:
        if row.get("split") == "train":
            chosen = row
            break
    if chosen is None and split_rows:
        chosen = split_rows[0]
    if not chosen:
        return [], "no public split found"

    params = {
        "dataset": dataset_id,
        "config": str(chosen.get("config") or "default"),
        "split": str(chosen.get("split") or "train"),
    }
    payload = _hf_json("first-rows", params)
    samples: list[str] = []
    for item in payload.get("rows", [])[:limit]:
        row = item.get("row", item)
        text = _row_text(row)
        if text:
            samples.append(text)
    return samples, ""


def _hf_rows_for_split(dataset_id: str, config: str, split: str, limit: int = 8) -> tuple[list[dict[str, Any]], str]:
    try:
        payload = _hf_json("first-rows", {"dataset": dataset_id, "config": config, "split": split})
    except Exception as exc:
        return [], _shorten(repr(exc), 260)
    rows = []
    for item in payload.get("rows", [])[:limit]:
        row = item.get("row", item)
        if isinstance(row, dict):
            rows.append(row)
    return rows, ""


def _choose_split(splits: list[dict[str, Any]], config: str = "", split: str = "") -> dict[str, str]:
    if config and split:
        for row in splits:
            if row.get("config") == config and row.get("split") == split:
                return {"config": config, "split": split}
    for row in splits:
        if row.get("config") == "default" and row.get("split") == "train":
            return {"config": "default", "split": "train"}
    for row in splits:
        if row.get("split") == "train":
            return {"config": str(row.get("config") or "default"), "split": "train"}
    if splits:
        row = splits[0]
        return {"config": str(row.get("config") or "default"), "split": str(row.get("split") or "train")}
    return {"config": "default", "split": "train"}


def _column_profile(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for row in rows:
        for key, value in row.items():
            item = stats.setdefault(key, {"name": key, "types": set(), "non_empty": 0, "example": ""})
            item["types"].add(type(value).__name__)
            if value not in (None, "", [], {}):
                item["non_empty"] += 1
                if not item["example"]:
                    item["example"] = _shorten(json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value, 120)
    output = []
    for item in stats.values():
        output.append(
            {
                "name": item["name"],
                "types": sorted(item["types"]),
                "non_empty": item["non_empty"],
                "example": item["example"],
            }
        )
    return sorted(output, key=lambda x: x["name"])


def _local_stream_samples(dataset_id: str, limit: int = 8, timeout: int = 18) -> tuple[list[str], str]:
    code = r'''
import json
import sys
from datasets import load_dataset

dataset_id = sys.argv[1]
limit = int(sys.argv[2])
MAX_SAMPLE_CHARS = 4000

def trim(text):
    text = str(text)
    if len(text) <= MAX_SAMPLE_CHARS:
        return text
    return text[:MAX_SAMPLE_CHARS].rstrip() + "\n...[truncated]"

def row_text(row):
    if isinstance(row, str):
        return trim(row)
    if not isinstance(row, dict):
        return ""
    preferred = ("text", "content", "document", "article", "body", "prompt", "response", "completion", "context")
    parts = []
    for key in preferred:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(trim(value))
    if parts:
        return trim("\n".join(parts))
    for value in row.values():
        if isinstance(value, str) and len(value.strip()) >= 40:
            parts.append(trim(value))
        elif isinstance(value, list):
            parts.extend(trim(v) for v in value[:20] if isinstance(v, str) and len(v.strip()) >= 40)
    return trim("\n".join(parts[:4]))

stream = load_dataset(dataset_id, split="train", streaming=True)
samples = []
for row in stream.take(limit):
    text = row_text(row)
    if text:
        samples.append(text)
print(json.dumps(samples, ensure_ascii=False))
'''
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code, dataset_id, str(limit)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return [], "local streaming timed out"
    if completed.returncode != 0:
        return [], _shorten(completed.stderr or completed.stdout or "local streaming failed", 260)
    try:
        payload = json.loads(completed.stdout or "[]")
    except json.JSONDecodeError as exc:
        return [], _shorten(f"local streaming returned invalid json: {exc}", 260)
    return [_trim_text(str(item)) for item in payload if str(item).strip()], ""


class DatasetMarket:
    def __init__(self, output_root: Path = DEFAULT_OUTPUT_ROOT) -> None:
        self.output_root = output_root
        self._api = None
        self.max_rss_gb = DEFAULT_MAX_RSS_GB

    @property
    def api(self):
        if self._api is None:
            _guard_memory(self.max_rss_gb)
            from huggingface_hub import HfApi

            self._api = HfApi()
        return self._api

    def search(
        self,
        query: str,
        language: str,
        goal: str,
        limit: int,
        min_score: float = 0.0,
        license_mode: str = "any",
        language_mode: str = "soft",
        include_terms: str = "",
        exclude_terms: str = "",
        sort_mode: str = "quality",
    ) -> dict[str, Any]:
        _guard_memory(self.max_rss_gb)
        fetch_limit = max(1, min(max(limit * 4, limit), 100))
        kwargs: dict[str, Any] = {
            "search": query or None,
            "limit": fetch_limit,
            "full": True,
        }
        if language:
            kwargs["language"] = language
        try:
            infos = list(self.api.list_datasets(sort="downloads", **kwargs))
        except TypeError:
            kwargs.pop("full", None)
            infos = list(self.api.list_datasets(**kwargs))
        include = _terms(include_terms)
        exclude = _terms(exclude_terms)
        lang_norm = language.lower().strip()
        cards = []
        for info in infos:
            card = _score(info, goal=goal, language=language)
            if card.score < min_score:
                continue
            if license_mode == "open" and card.license not in PERMISSIVE_LICENSES:
                continue
            if license_mode == "no_unknown" and card.license == "unknown":
                continue
            if language_mode == "strict" and lang_norm:
                if lang_norm not in card.languages and f"language:{lang_norm}" not in card.tags:
                    continue
            if include and not _matches_terms(card, include, require_all=True):
                continue
            if exclude and _matches_terms(card, exclude, require_all=False):
                continue
            cards.append(card)
        if sort_mode == "downloads":
            cards.sort(key=lambda c: (c.downloads, c.score, c.likes), reverse=True)
        elif sort_mode == "likes":
            cards.sort(key=lambda c: (c.likes, c.score, c.downloads), reverse=True)
        elif sort_mode == "low_risk":
            cards.sort(key=lambda c: (len(c.warnings) == 0, c.score, c.downloads), reverse=True)
        else:
            cards.sort(key=lambda c: (c.score, c.downloads, c.likes), reverse=True)
        cards = cards[: max(1, min(limit, 100))]
        _guard_memory(self.max_rss_gb)
        return {
            "items": [asdict(card) for card in cards],
            "count": len(cards),
            "fetched": len(infos),
            "filters": {
                "min_score": min_score,
                "license_mode": license_mode,
                "language_mode": language_mode,
                "include_terms": include,
                "exclude_terms": exclude,
                "sort_mode": sort_mode,
            },
        }

    def analyze(self, dataset_id: str, goal: str, language: str) -> dict[str, Any]:
        _guard_memory(self.max_rss_gb)
        info = self.api.dataset_info(dataset_id)
        card = _score(info, goal=goal, language=language)
        files: list[str] = []
        readme_excerpt = ""
        try:
            files = list(self.api.list_repo_files(dataset_id, repo_type="dataset"))[:250]
        except Exception:
            files = []
        try:
            from huggingface_hub import hf_hub_download

            readme_path = hf_hub_download(dataset_id, "README.md", repo_type="dataset")
            readme_excerpt = _shorten(Path(readme_path).read_text(encoding="utf-8", errors="ignore"), 900)
        except Exception:
            readme_excerpt = ""

        text_files = [name for name in files if name.lower().endswith(TEXT_FILE_EXTS)]
        binary_or_media = [
            name
            for name in files
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".wav", ".mp3", ".flac", ".mp4"))
        ]
        file_signals: list[str] = []
        file_warnings: list[str] = []
        file_delta = 0
        if text_files:
            file_delta += 5
            file_signals.append(f"text-like files visible: {len(text_files)}")
        if binary_or_media and len(binary_or_media) > len(text_files):
            file_delta -= 12
            file_warnings.append("mostly media files, weak fit for text pretraining")
        if not files:
            file_warnings.append("file list unavailable")

        samples: list[str] = []
        sample_error = ""
        try:
            samples, sample_error = _hf_first_rows(dataset_id, limit=8)
        except Exception as exc:
            sample_error = _shorten(repr(exc), 260)
        if not samples:
            local_samples, local_error = _local_stream_samples(dataset_id, limit=8)
            if local_samples:
                samples = local_samples
                sample_error = ""
            elif local_error:
                sample_error = (sample_error + " | " + local_error).strip(" |")

        if sample_error and not samples:
            sample_report = {
                "score_delta": 0,
                "sample_count": 0,
                "avg_chars": 0,
                "estimated_keep_rate": None,
                "signals": [],
                "warnings": [],
                "examples": [],
            }
        else:
            sample_report = _sample_quality(samples, goal=goal, language=language)
        score = round(max(0.0, min(100.0, card.score + file_delta + sample_report["score_delta"])), 1)
        warnings = [*card.warnings, *file_warnings, *sample_report["warnings"]]
        signals = [*card.reasons, *file_signals, *sample_report["signals"]]
        confidence = "sampled"
        if sample_error:
            warnings.append("sample streaming failed; metadata-only verdict")
        if sample_error and not samples:
            confidence = "metadata_only"
            score = min(score, 74.0)
            warnings.append("AI score capped because samples were not verified")

        return {
            "id": dataset_id,
            "goal": goal,
            "language": language,
            "metadata_score": card.score,
            "ai_score": score,
            "verdict": _final_verdict(score, warnings),
            "confidence": confidence,
            "route": card.cleaning_route,
            "license": card.license,
            "downloads": card.downloads,
            "likes": card.likes,
            "signals": signals[:12],
            "warnings": warnings[:12],
            "sample": sample_report,
            "files": {
                "total_seen": len(files),
                "text_like": text_files[:15],
                "media_like_count": len(binary_or_media),
            },
            "readme_excerpt": readme_excerpt,
            "sample_error": sample_error,
        }

    def preview(
        self,
        dataset_id: str,
        goal: str,
        language: str,
        config: str = "",
        split: str = "",
    ) -> dict[str, Any]:
        _guard_memory(self.max_rss_gb)
        split_error = ""
        parquet_error = ""
        splits: list[dict[str, Any]] = []
        parquet_files: list[dict[str, Any]] = []
        try:
            splits_payload = _hf_json("splits", {"dataset": dataset_id})
            splits = [
                {"config": str(row.get("config") or "default"), "split": str(row.get("split") or "train")}
                for row in splits_payload.get("splits", [])
            ]
        except Exception as exc:
            split_error = _shorten(repr(exc), 260)

        chosen = _choose_split(splits, config=config, split=split)
        try:
            parquet_payload = _hf_json("parquet", {"dataset": dataset_id})
            for row in parquet_payload.get("parquet_files", []):
                if chosen["config"] and row.get("config") != chosen["config"]:
                    continue
                if chosen["split"] and row.get("split") != chosen["split"]:
                    continue
                parquet_files.append(
                    {
                        "config": row.get("config"),
                        "split": row.get("split"),
                        "filename": row.get("filename"),
                        "size": row.get("size") or 0,
                        "url": row.get("url"),
                    }
                )
                if len(parquet_files) >= 500:
                    break
        except Exception as exc:
            parquet_error = _shorten(repr(exc), 260)

        rows, row_error = _hf_rows_for_split(dataset_id, chosen["config"], chosen["split"], limit=12)
        rows = [_trim_json_value(row) for row in rows]
        samples = [_row_text(row) for row in rows]
        quality = _sample_quality(samples, goal=goal, language=language) if samples else {
            "score_delta": 0,
            "sample_count": 0,
            "avg_chars": 0,
            "estimated_keep_rate": None,
            "signals": [],
            "warnings": ["no readable preview samples"],
            "examples": [],
        }
        size_bytes = sum(int(row.get("size") or 0) for row in parquet_files)
        return {
            "id": dataset_id,
            "chosen": chosen,
            "splits": splits[:80],
            "split_count": len(splits),
            "columns": _column_profile(rows),
            "rows": rows[:5],
            "sample_quality": quality,
            "parquet": {
                "files_seen": len(parquet_files),
                "visible_size_bytes": size_bytes,
                "visible_size_gb": round(size_bytes / (1024**3), 3),
                "files": parquet_files[:12],
            },
            "errors": [err for err in (split_error, parquet_error, row_error) if err],
        }

    def plan_mix(self, selected: list[dict[str, Any]], target_tokens: int, goal: str) -> dict[str, Any]:
        _guard_memory(self.max_rss_gb)
        cards = [DatasetCard(**item) for item in selected]
        if not cards:
            return {"target_tokens": target_tokens, "items": [], "warnings": ["no datasets selected"]}
        weights = []
        for card in cards:
            base = max(0.05, card.score / 100.0)
            if card.score_label == "strong":
                base *= 1.25
            if card.warnings:
                base *= 0.75
            weights.append(base)
        total = sum(weights) or 1.0
        items = []
        for card, weight in zip(cards, weights, strict=False):
            share = weight / total
            items.append(
                {
                    "id": card.id,
                    "weight": round(share, 4),
                    "target_tokens": int(target_tokens * share),
                    "score": card.score,
                    "route": card.cleaning_route,
                    "license": card.license,
                    "warnings": card.warnings,
                }
            )
        return {
            "goal": goal,
            "target_tokens": target_tokens,
            "items": sorted(items, key=lambda x: x["weight"], reverse=True),
        }

    def pipeline(self, selected: list[dict[str, Any]], mix: dict[str, Any]) -> dict[str, Any]:
        _guard_memory(self.max_rss_gb)
        root = self.output_root
        raw_root = root / "market_raw"
        clean_root = root / "market_clean"
        commands = [
            f'New-Item -ItemType Directory -Force -Path "{raw_root}"',
            f'New-Item -ItemType Directory -Force -Path "{clean_root}"',
        ]
        manifest_items = []
        for item in mix.get("items", []):
            ds_id = item["id"]
            slug = re.sub(r"[^A-Za-z0-9_.-]+", "__", ds_id)
            route = item["route"]
            raw_path = raw_root / slug
            assembled_text = clean_root / f"{slug}.assembled.txt"
            clean_txt = clean_root / f"{slug}.structured.txt"
            clean_jsonl = clean_root / f"{slug}.structured.jsonl"
            commands.append(f'huggingface-cli download --repo-type dataset "{ds_id}" --local-dir "{raw_path}"')
            commands.append(
                "# Export/assemble the downloaded dataset into one UTF-8 text file before cleaning. "
                f'Replace DATA_FILE with the real .txt/.jsonl/.parquet export: "{assembled_text}"'
            )
            if route == "math_structure_min_language":
                extra = "--min-language-signal 0.0"
            elif route == "code_preserve_structure":
                extra = "--min-words 20 --min-language-signal 0.0"
            else:
                extra = "--min-language-signal 0.10"
            commands.append(
                "python scripts/data/structure_clean_pretrain.py "
                f'--input "{assembled_text}" '
                f'--output-jsonl "{clean_jsonl}" '
                f'--output-text "{clean_txt}" '
                f"--min-score 0.62 {extra}"
            )
            manifest_items.append(
                {
                    "dataset": ds_id,
                    "route": route,
                    "target_tokens": item["target_tokens"],
                    "weight": item["weight"],
                    "raw_dir": str(raw_path),
                    "assembled_text": str(assembled_text),
                    "clean_text": str(clean_txt),
                }
            )
        manifest = {"version": 1, "output_root": str(root), "mix": mix, "datasets": manifest_items}
        return {"commands": commands, "manifest": manifest}


APP_HTML = r"""
<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Auralis Dataset Market</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0b0f12;
      --panel: rgba(18, 24, 29, 0.92);
      --panel-2: rgba(30, 39, 45, 0.92);
      --line: #26333a;
      --line-soft: rgba(255,255,255,0.07);
      --text: #edf4f1;
      --muted: #9baaa9;
      --good: #64d98a;
      --warn: #f0bd5d;
      --bad: #ef746c;
      --blue: #78b7ff;
      --aqua: #66d9c5;
      --ink: #0f1519;
      --shadow: 0 18px 60px rgba(0,0,0,0.34);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        linear-gradient(135deg, rgba(102,217,197,0.11), transparent 34%),
        linear-gradient(225deg, rgba(240,189,93,0.08), transparent 40%),
        repeating-linear-gradient(90deg, rgba(255,255,255,0.025) 0 1px, transparent 1px 72px),
        var(--bg);
      color: var(--text);
      font: 14px/1.45 system-ui, -apple-system, Segoe UI, sans-serif;
    }
    header {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 20px;
      align-items: center;
      padding: 18px 22px;
      border-bottom: 1px solid var(--line);
      background: rgba(11, 15, 18, 0.88);
      backdrop-filter: blur(18px);
      position: sticky;
      top: 0;
      z-index: 2;
    }
    .brand { display: flex; gap: 14px; align-items: center; }
    .brand-mark {
      width: 42px;
      height: 42px;
      border-radius: 8px;
      border: 1px solid rgba(102,217,197,0.55);
      background:
        linear-gradient(135deg, rgba(102,217,197,0.3), rgba(120,183,255,0.1)),
        #111a1d;
      box-shadow: 0 0 24px rgba(102,217,197,0.16);
      position: relative;
      overflow: hidden;
    }
    .brand-mark::before,
    .brand-mark::after {
      content: "";
      position: absolute;
      inset: 10px;
      border-top: 2px solid var(--aqua);
      border-bottom: 2px solid var(--warn);
      transform: skewY(-18deg);
    }
    .brand-mark::after {
      inset: 16px 8px;
      border-color: var(--blue);
      opacity: 0.8;
    }
    h1 { margin: 0; font-size: 21px; font-weight: 800; letter-spacing: 0; }
    .subtitle { color: var(--muted); margin-top: 2px; font-size: 12px; }
    #status {
      min-width: 110px;
      text-align: center;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 7px 12px;
      background: rgba(255,255,255,0.04);
      color: var(--aqua);
    }
    main {
      display: grid;
      grid-template-columns: 320px minmax(520px, 1fr) minmax(430px, 25vw);
      gap: 16px;
      padding: 16px;
      min-height: calc(100vh - 79px);
    }
    aside, section {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    aside, .results, .planner { padding: 16px; }
    aside { align-self: start; position: sticky; top: 96px; }
    label { display: block; color: var(--muted); font-size: 12px; margin: 12px 0 6px; }
    input, select, button, textarea {
      width: 100%;
      border: 1px solid var(--line);
      background: rgba(9, 13, 16, 0.86);
      color: var(--text);
      border-radius: 7px;
      padding: 10px 11px;
      font: inherit;
      outline: none;
      transition: border-color .18s ease, background .18s ease, transform .18s ease;
    }
    input:focus, select:focus {
      border-color: rgba(102,217,197,0.72);
      box-shadow: 0 0 0 3px rgba(102,217,197,0.10);
    }
    button {
      cursor: pointer;
      background: linear-gradient(135deg, #223a3a, #243142);
      border-color: #405269;
      font-weight: 720;
    }
    button:hover { transform: translateY(-1px); border-color: var(--aqua); }
    button.secondary { background: rgba(255,255,255,0.05); }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .toolbar { display: flex; gap: 8px; align-items: center; justify-content: space-between; margin-bottom: 12px; }
    .toolbar strong, .planner strong { font-size: 13px; text-transform: uppercase; color: #dbe9e6; }
    .muted { color: var(--muted); }
    .mini-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 12px 0; }
    .stat {
      border: 1px solid var(--line-soft);
      border-radius: 8px;
      padding: 10px;
      background: rgba(255,255,255,0.035);
    }
    .stat b { display: block; font-size: 19px; color: var(--aqua); }
    .stat span { color: var(--muted); font-size: 11px; }
    .list { display: grid; gap: 12px; }
    .dataset {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background:
        linear-gradient(135deg, rgba(102,217,197,0.055), transparent 38%),
        var(--panel-2);
      min-height: 126px;
      transition: transform .2s ease, border-color .2s ease, background .2s ease;
      animation: rise .25s ease both;
    }
    .dataset:hover {
      transform: translateY(-2px);
      border-color: rgba(102,217,197,0.58);
      background: rgba(31, 42, 48, 0.96);
    }
    .dataset h2 {
      margin: 0 0 8px;
      font-size: 16px;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }
    .dataset-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      align-items: end;
    }
    .meter {
      height: 7px;
      border-radius: 999px;
      background: #11181c;
      overflow: hidden;
      border: 1px solid rgba(255,255,255,0.06);
      margin: 8px 0;
    }
    .meter > span {
      display: block;
      height: 100%;
      width: var(--score-width);
      background: linear-gradient(90deg, var(--bad), var(--warn), var(--good));
      transition: width .45s ease;
    }
    .meta { display: flex; flex-wrap: wrap; gap: 6px; margin: 7px 0; }
    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      border-radius: 999px;
      padding: 2px 8px;
      background: rgba(255,255,255,0.055);
      color: var(--muted);
      font-size: 12px;
      border: 1px solid var(--line-soft);
    }
    .score { font-weight: 850; color: var(--blue); }
    .strong { color: var(--good); }
    .usable { color: var(--blue); }
    .risky { color: var(--warn); }
    .weak { color: var(--bad); }
    .reason { color: var(--muted); margin: 3px 0; font-size: 12px; }
    .reason-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 7px;
      color: var(--muted);
      font-size: 12px;
    }
    .warn { color: var(--warn); }
    .actions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 0; justify-content: flex-end; }
    .actions button { width: auto; min-width: 92px; }
    .selected-item {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      border-bottom: 1px solid var(--line);
      padding: 9px 0;
      overflow-wrap: anywhere;
    }
    pre, .ai-card {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: rgba(7, 10, 12, 0.78);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 11px;
      max-height: 360px;
      overflow: auto;
    }
    .ai-card { max-height: none; margin-top: 8px; overflow: visible; }
    .ai-score {
      display: grid;
      grid-template-columns: 88px 1fr;
      gap: 12px;
      align-items: center;
      margin-bottom: 12px;
    }
    .dial {
      width: 76px;
      height: 76px;
      border-radius: 50%;
      display: grid;
      place-items: center;
      background: conic-gradient(var(--aqua) calc(var(--ai-score) * 1%), #182026 0);
      position: relative;
      font-weight: 850;
    }
    .dial::after {
      content: "";
      position: absolute;
      inset: 7px;
      border-radius: 50%;
      background: var(--ink);
      border: 1px solid var(--line);
    }
    .dial span { position: relative; z-index: 1; }
    .chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }
    .overlay {
      position: fixed;
      inset: 0;
      background: rgba(5, 8, 10, 0.72);
      backdrop-filter: blur(10px);
      display: none;
      align-items: center;
      justify-content: center;
      padding: 24px;
      z-index: 8;
    }
    .overlay.open { display: flex; }
    .modal {
      width: min(1040px, 94vw);
      max-height: 86vh;
      overflow: auto;
      border: 1px solid rgba(102,217,197,0.32);
      border-radius: 8px;
      background:
        linear-gradient(135deg, rgba(102,217,197,0.08), transparent 32%),
        #10171b;
      box-shadow: 0 28px 90px rgba(0,0,0,0.55);
      padding: 18px;
      animation: rise .18s ease both;
    }
    .modal-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }
    .modal-body {
      display: grid;
      grid-template-columns: 260px minmax(0, 1fr);
      gap: 16px;
    }
    .sample-box {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: rgba(0,0,0,0.24);
      margin-top: 8px;
      max-height: 240px;
      overflow: auto;
    }
    .table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
      font-size: 12px;
    }
    .table th, .table td {
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      padding: 7px 6px;
    }
    .split-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      max-height: 116px;
      overflow: auto;
      margin: 8px 0;
    }
    @keyframes rise {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }
    @media (max-width: 1150px) {
      main { grid-template-columns: 1fr; }
      aside { position: static; }
      .modal-body { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <div class="brand-mark" aria-hidden="true"></div>
      <div>
        <h1>Auralis Dataset Market</h1>
        <div class="subtitle">Findet, bewertet und plant Trainingsdaten fuer sauberes Pretraining.</div>
      </div>
    </div>
    <div class="muted" id="status">bereit</div>
  </header>
  <main>
    <aside>
      <div class="mini-grid">
        <div class="stat"><b id="statFound">0</b><span>Kandidaten</span></div>
        <div class="stat"><b id="statPicked">0</b><span>im Mix</span></div>
      </div>
      <label>Suchbegriff</label>
      <input id="query" value="german pretrain text" placeholder="z. B. german web corpus">
      <div class="row">
        <div>
          <label>Sprache</label>
          <input id="language" value="de" placeholder="de, en">
        </div>
        <div>
          <label>Ziel</label>
          <select id="goal">
            <option value="base_pretrain">Base Pretraining</option>
            <option value="math">Math</option>
            <option value="code">Code</option>
            <option value="sft">SFT</option>
          </select>
        </div>
      </div>
      <div class="row">
        <div>
          <label>Min Score</label>
          <input id="minScore" type="number" value="55" min="0" max="100">
        </div>
        <div>
          <label>Lizenz</label>
          <select id="licenseMode">
            <option value="open">Nur offen</option>
            <option value="no_unknown">Ohne unknown</option>
            <option value="any">Alle</option>
          </select>
        </div>
      </div>
      <div class="row">
        <div>
          <label>Sprache-Modus</label>
          <select id="languageMode">
            <option value="soft">Weich</option>
            <option value="strict">Strikt HF-Tag</option>
          </select>
        </div>
        <div>
          <label>Sortierung</label>
          <select id="sortMode">
            <option value="quality">Qualitaet</option>
            <option value="low_risk">Wenig Risiko</option>
            <option value="downloads">Downloads</option>
            <option value="likes">Likes</option>
          </select>
        </div>
      </div>
      <label>Include Begriffe</label>
      <input id="includeTerms" value="" placeholder="z. B. fineweb, wiki, corpus">
      <label>Exclude Begriffe</label>
      <input id="excludeTerms" value="sft, chat, dpo, rlhf, preference, audio, asr, image, embed, toxicity" placeholder="sft, chat, audio, image">
      <label>Treffer</label>
      <input id="limit" type="number" value="24" min="1" max="100">
      <label>Target Tokens</label>
      <input id="tokens" type="number" value="1000000000" min="1000000">
      <div style="height:12px"></div>
      <button id="search">Suchen und bewerten</button>
      <div style="height:8px"></div>
      <button id="mix">Mix planen</button>
      <div style="height:8px"></div>
      <button id="pipeline">Pipeline erzeugen</button>
    </aside>
    <section class="results">
      <div class="toolbar">
        <strong>Dataset-Kandidaten</strong>
        <span class="muted" id="count"></span>
      </div>
      <div class="list" id="results"></div>
    </section>
    <section class="planner">
      <strong>Auswahl</strong>
      <div id="selected" style="margin:10px 0 18px"></div>
      <strong>Mix</strong>
      <pre id="mixOut">Noch kein Mix geplant.</pre>
      <strong>Pipeline</strong>
      <pre id="pipelineOut">Noch keine Pipeline erzeugt.</pre>
      <strong>KI Analyse</strong>
      <div class="ai-card" id="aiOut">Waehle bei einem Dataset "KI pruefen", um Metadaten, Dateien, README und echte Samples bewerten zu lassen.</div>
    </section>
  </main>
  <div class="overlay" id="aiOverlay">
    <div class="modal">
      <div class="modal-head">
        <strong>KI Dataset Review</strong>
        <button class="secondary" style="width:auto" onclick="closeAIOverlay()">Schliessen</button>
      </div>
      <div id="aiModalBody"></div>
    </div>
  </div>
<script>
const state = { items: [], selected: [], mix: null };
const $ = (id) => document.getElementById(id);
function status(text) { $("status").textContent = text; }
function esc(s) { return String(s ?? "").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function hfDatasetUrl(id) {
  return "https://huggingface.co/datasets/" + String(id).split("/").map(encodeURIComponent).join("/");
}
function updateStats() {
  $("statFound").textContent = state.items.length;
  $("statPicked").textContent = state.selected.length;
}
async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}
function renderResults() {
  $("count").textContent = `${state.items.length} Treffer`;
  $("results").innerHTML = state.items.map(item => `
    <article class="dataset">
      <div class="dataset-grid">
        <div>
          <h2>${esc(item.id)}</h2>
          <div class="meter" style="--score-width:${Math.max(2, item.score)}%"><span></span></div>
          <div class="meta">
            <span class="pill score ${esc(item.score_label)}">${item.score} ${esc(item.score_label)}</span>
            <span class="pill">${item.downloads.toLocaleString()} downloads</span>
            <span class="pill">${item.likes.toLocaleString()} likes</span>
            <span class="pill">${esc(item.license)}</span>
            <span class="pill">${esc(item.cleaning_route)}</span>
          </div>
          <div class="reason-row">
            ${(item.reasons || []).slice(0,2).map(r => `<span>${esc(r)}</span>`).join("")}
            ${(item.warnings || []).slice(0,2).map(w => `<span class="warn">${esc(w)}</span>`).join("")}
          </div>
        </div>
        <div class="actions">
          <button onclick="selectItem('${esc(item.id)}')">Auswaehlen</button>
          <button class="secondary" onclick="previewItem('${esc(item.id)}')">Preview</button>
          <button class="secondary" onclick="analyzeItem('${esc(item.id)}')">KI pruefen</button>
          <button onclick="window.open(hfDatasetUrl('${esc(item.id)}'),'_blank')">Oeffnen</button>
        </div>
      </div>
    </article>
  `).join("");
  updateStats();
}
function renderSelected() {
  $("selected").innerHTML = state.selected.length ? state.selected.map(item => `
    <div class="selected-item">
      <span>${esc(item.id)}<br><span class="muted">${item.score} / ${esc(item.cleaning_route)}</span></span>
      <button onclick="removeItem('${esc(item.id)}')">Entfernen</button>
    </div>
  `).join("") : '<span class="muted">Noch nichts ausgewaehlt.</span>';
  updateStats();
}
window.selectItem = (id) => {
  const item = state.items.find(x => x.id === id);
  if (item && !state.selected.some(x => x.id === id)) state.selected.push(item);
  renderSelected();
};
window.removeItem = (id) => {
  state.selected = state.selected.filter(x => x.id !== id);
  renderSelected();
};
function aiMarkup(data, compact=false) {
  const signalChips = (data.signals || []).slice(0,8).map(x => `<span class="pill">${esc(x)}</span>`).join("");
  const warningChips = (data.warnings || []).slice(0,8).map(x => `<span class="pill warn">${esc(x)}</span>`).join("");
  const examples = ((data.sample || {}).examples || []).map(x => `<div class="sample-box">${esc(x)}</div>`).join("");
  const files = ((data.files || {}).text_like || []).slice(0,5).map(x => `<div class="reason">${esc(x)}</div>`).join("");
  const limitDetails = compact ? 4 : 8;
  return `
    <div class="ai-score">
      <div class="dial" style="--ai-score:${Math.max(0, Math.min(100, data.ai_score || 0))}"><span>${esc(data.ai_score)}</span></div>
      <div>
        <h2 style="margin:0 0 4px;font-size:16px">${esc(data.id)}</h2>
        <div class="reason">${esc(data.verdict)}</div>
        <div class="meta">
          <span class="pill">metadata ${esc(data.metadata_score)}</span>
          <span class="pill">${esc(data.confidence || "sampled")}</span>
          <span class="pill">${esc(data.route)}</span>
          <span class="pill">${esc(data.license)}</span>
        </div>
      </div>
    </div>
    <strong>Gute Signale</strong>
    <div class="chips">${(data.signals || []).slice(0,limitDetails).map(x => `<span class="pill">${esc(x)}</span>`).join("") || '<span class="muted">Keine starken Signale.</span>'}</div>
    <strong>Risiken</strong>
    <div class="chips">${(data.warnings || []).slice(0,limitDetails).map(x => `<span class="pill warn">${esc(x)}</span>`).join("") || '<span class="muted">Keine harten Warnungen.</span>'}</div>
    <strong>Sample-Profil</strong>
    <div class="reason">Samples: ${esc((data.sample || {}).sample_count)} / Avg chars: ${esc((data.sample || {}).avg_chars)} / Keep-Schaetzung: ${esc((data.sample || {}).estimated_keep_rate)}</div>
    ${compact ? "" : ""}
    <strong>Dateien</strong>
    <div class="reason">Gesehen: ${esc((data.files || {}).total_seen)} / Medien: ${esc((data.files || {}).media_like_count)}</div>
    ${compact ? "" : files}
    ${data.sample_error ? `<div class="reason warn">${esc(data.sample_error)}</div>` : ""}
    ${compact ? '<button class="secondary" style="margin-top:10px" onclick="openAIOverlay()">Gross anzeigen</button>' : ""}
  `;
}
function renderAI(data) {
  state.lastAI = data;
  $("aiOut").innerHTML = aiMarkup(data, true);
  $("aiModalBody").innerHTML = `<div class="modal-body"><div>${aiMarkup(data, false)}</div><div><strong>Beispiele</strong>${((data.sample || {}).examples || []).map(x => `<div class="sample-box">${esc(x)}</div>`).join("") || '<div class="reason">Keine Samples verfuegbar.</div>'}</div></div>`;
}
function renderPreview(data) {
  const splits = (data.splits || []).slice(0, 40).map(s => `<span class="pill">${esc(s.config)}/${esc(s.split)}</span>`).join("");
  const columns = (data.columns || []).map(c => `
    <tr><td>${esc(c.name)}</td><td>${esc((c.types || []).join(", "))}</td><td>${esc(c.non_empty)}</td><td>${esc(c.example)}</td></tr>
  `).join("");
  const files = ((data.parquet || {}).files || []).map(f => `
    <tr><td>${esc(f.filename)}</td><td>${esc(f.config)}/${esc(f.split)}</td><td>${((f.size || 0) / (1024*1024)).toFixed(1)} MB</td></tr>
  `).join("");
  const samples = ((data.sample_quality || {}).examples || []).map(x => `<div class="sample-box">${esc(x)}</div>`).join("");
  const errors = (data.errors || []).map(x => `<div class="reason warn">${esc(x)}</div>`).join("");
  $("aiOut").innerHTML = `
    <h2 style="margin:0 0 8px;font-size:16px">${esc(data.id)}</h2>
    <div class="meta">
      <span class="pill">preview</span>
      <span class="pill">${esc((data.chosen || {}).config)}/${esc((data.chosen || {}).split)}</span>
      <span class="pill">${esc(data.split_count)} splits</span>
      <span class="pill">${esc((data.parquet || {}).visible_size_gb)} GB visible</span>
    </div>
    <div class="reason">Samples: ${esc((data.sample_quality || {}).sample_count)} / Keep-Schaetzung: ${esc((data.sample_quality || {}).estimated_keep_rate)}</div>
    ${errors}
    <button class="secondary" style="margin-top:10px" onclick="openAIOverlay()">Gross anzeigen</button>
  `;
  $("aiModalBody").innerHTML = `
    <div class="modal-body">
      <div>
        <h2 style="margin-top:0">${esc(data.id)}</h2>
        <div class="chips">
          <span class="pill">${esc((data.chosen || {}).config)}/${esc((data.chosen || {}).split)}</span>
          <span class="pill">${esc(data.split_count)} splits</span>
          <span class="pill">${esc((data.parquet || {}).files_seen)} parquet files</span>
          <span class="pill">${esc((data.parquet || {}).visible_size_gb)} GB visible</span>
        </div>
        <strong>Splits</strong>
        <div class="split-grid">${splits || '<span class="muted">Keine Splits sichtbar.</span>'}</div>
        <strong>Sample-Qualitaet</strong>
        <div class="reason">Avg chars: ${esc((data.sample_quality || {}).avg_chars)} / Keep: ${esc((data.sample_quality || {}).estimated_keep_rate)}</div>
        <div class="chips">${((data.sample_quality || {}).warnings || []).map(x => `<span class="pill warn">${esc(x)}</span>`).join("") || '<span class="muted">Keine Preview-Warnungen.</span>'}</div>
        ${errors}
      </div>
      <div>
        <strong>Spalten</strong>
        <table class="table"><thead><tr><th>Name</th><th>Typ</th><th>Non-empty</th><th>Beispiel</th></tr></thead><tbody>${columns || '<tr><td colspan="4">Keine Spalten erkannt.</td></tr>'}</tbody></table>
        <strong>Parquet-Shards</strong>
        <table class="table"><thead><tr><th>Datei</th><th>Split</th><th>Groesse</th></tr></thead><tbody>${files || '<tr><td colspan="3">Keine Parquet-Dateien sichtbar.</td></tr>'}</tbody></table>
        <strong>Beispiele</strong>
        ${samples || '<div class="reason">Keine Textsamples verfuegbar.</div>'}
      </div>
    </div>
  `;
}
window.openAIOverlay = () => {
  $("aiOverlay").classList.add("open");
};
window.closeAIOverlay = () => {
  $("aiOverlay").classList.remove("open");
};
$("aiOverlay").onclick = (event) => {
  if (event.target.id === "aiOverlay") closeAIOverlay();
};
window.addEventListener("keydown", event => {
  if (event.key === "Escape") closeAIOverlay();
});
window.analyzeItem = async (id) => {
  status("KI prueft...");
  $("aiOut").innerHTML = '<div class="reason">Analyse laeuft: Metadaten, Dateien, README und Samples werden geprueft.</div>';
  try {
    const data = await api("/api/analyze", {
      method: "POST",
      headers: {"content-type": "application/json"},
      body: JSON.stringify({id, goal: $("goal").value, language: $("language").value})
    });
    renderAI(data);
    openAIOverlay();
    status("KI Analyse fertig");
  } catch (err) {
    $("aiOut").textContent = String(err);
    status("Analysefehler");
  }
};
window.previewItem = async (id) => {
  status("Preview laeuft...");
  $("aiOut").innerHTML = '<div class="reason">Preview laeuft: Splits, Parquet, Spalten und Samples werden geladen.</div>';
  try {
    const data = await api("/api/preview", {
      method: "POST",
      headers: {"content-type": "application/json"},
      body: JSON.stringify({id, goal: $("goal").value, language: $("language").value})
    });
    renderPreview(data);
    openAIOverlay();
    status("Preview fertig");
  } catch (err) {
    $("aiOut").textContent = String(err);
    status("Previewfehler");
  }
};
$("search").onclick = async () => {
  status("suche...");
  const params = new URLSearchParams({
    query: $("query").value,
    language: $("language").value,
    goal: $("goal").value,
    limit: $("limit").value,
    min_score: $("minScore").value,
    license_mode: $("licenseMode").value,
    language_mode: $("languageMode").value,
    include_terms: $("includeTerms").value,
    exclude_terms: $("excludeTerms").value,
    sort: $("sortMode").value
  });
  const data = await api(`/api/search?${params}`);
  state.items = data.items;
  renderResults();
  status("fertig");
};
$("mix").onclick = async () => {
  status("plane mix...");
  state.mix = await api("/api/mix", {
    method: "POST",
    headers: {"content-type": "application/json"},
    body: JSON.stringify({selected: state.selected, target_tokens: Number($("tokens").value), goal: $("goal").value})
  });
  $("mixOut").textContent = JSON.stringify(state.mix, null, 2);
  status("mix fertig");
};
$("pipeline").onclick = async () => {
  if (!state.mix) await $("mix").onclick();
  status("erzeuge pipeline...");
  const data = await api("/api/pipeline", {
    method: "POST",
    headers: {"content-type": "application/json"},
    body: JSON.stringify({selected: state.selected, mix: state.mix})
  });
  $("pipelineOut").textContent = data.commands.join("\n") + "\n\n" + JSON.stringify(data.manifest, null, 2);
  status("pipeline fertig");
};
renderSelected();
updateStats();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    market: DatasetMarket

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("cache-control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: Any) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _read_json_body(self) -> tuple[dict[str, Any] | None, str | None]:
        length = int(self.headers.get("content-length", "0") or "0")
        if length > MAX_REQUEST_BODY_BYTES:
            return None, f"request body too large: {length:,} bytes"
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError as exc:
            return None, str(exc)
        if not isinstance(payload, dict):
            return None, "request body must be a JSON object"
        return payload, None

    def do_GET(self) -> None:  # noqa: N802
        try:
            _guard_memory(self.market.max_rss_gb)
        except MemoryError as exc:
            self._json(503, {"error": str(exc)})
            return
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self._send(200, APP_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/search":
            q = urllib.parse.parse_qs(parsed.query)
            try:
                payload = self.market.search(
                    query=(q.get("query", [""])[0] or "").strip(),
                    language=(q.get("language", [""])[0] or "").strip(),
                    goal=(q.get("goal", ["base_pretrain"])[0] or "base_pretrain"),
                    limit=int(q.get("limit", ["24"])[0] or 24),
                    min_score=float(q.get("min_score", ["0"])[0] or 0),
                    license_mode=(q.get("license_mode", ["any"])[0] or "any"),
                    language_mode=(q.get("language_mode", ["soft"])[0] or "soft"),
                    include_terms=(q.get("include_terms", [""])[0] or ""),
                    exclude_terms=(q.get("exclude_terms", [""])[0] or ""),
                    sort_mode=(q.get("sort", ["quality"])[0] or "quality"),
                )
                self._json(200, payload)
            except Exception as exc:  # pragma: no cover - UI error path
                self._json(500, {"error": repr(exc)})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        try:
            _guard_memory(self.market.max_rss_gb)
        except MemoryError as exc:
            self._json(503, {"error": str(exc)})
            return
        payload, error = self._read_json_body()
        if error:
            self._json(400, {"error": error})
            return
        if parsed.path == "/api/mix":
            self._json(
                200,
                self.market.plan_mix(
                    selected=payload.get("selected", []),
                    target_tokens=int(payload.get("target_tokens", 1_000_000_000)),
                    goal=payload.get("goal", "base_pretrain"),
                ),
            )
            return
        if parsed.path == "/api/pipeline":
            self._json(200, self.market.pipeline(payload.get("selected", []), payload.get("mix", {})))
            return
        if parsed.path == "/api/analyze":
            try:
                self._json(
                    200,
                    self.market.analyze(
                        dataset_id=str(payload.get("id", "")).strip(),
                        goal=payload.get("goal", "base_pretrain"),
                        language=payload.get("language", "de"),
                    ),
                )
            except Exception as exc:
                self._json(500, {"error": repr(exc)})
            return
        if parsed.path == "/api/preview":
            try:
                self._json(
                    200,
                    self.market.preview(
                        dataset_id=str(payload.get("id", "")).strip(),
                        goal=payload.get("goal", "base_pretrain"),
                        language=payload.get("language", "de"),
                        config=str(payload.get("config", "") or ""),
                        split=str(payload.get("split", "") or ""),
                    ),
                )
            except Exception as exc:
                self._json(500, {"error": repr(exc)})
            return
        self._json(404, {"error": "not found"})

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[dataset-market] " + fmt % args + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--max-rss-gb",
        type=float,
        default=float(os.environ.get("AURALIS_DATASET_MARKET_MAX_RSS_GB", DEFAULT_MAX_RSS_GB)),
        help="Stop serving expensive requests once this Python process exceeds the RSS/private-memory limit.",
    )
    args = parser.parse_args()

    Handler.market = DatasetMarket(args.output_root)
    Handler.market.max_rss_gb = max(1.0, args.max_rss_gb)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.request_queue_size = 16
    print(
        textwrap.dedent(
            f"""
            Auralis Dataset Market running
              URL: http://{args.host}:{args.port}
              output_root: {args.output_root}
              max_rss_gb: {Handler.market.max_rss_gb:.1f}
            """
        ).strip(),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
