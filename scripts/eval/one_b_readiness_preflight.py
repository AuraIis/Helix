#!/usr/bin/env python3
"""Preflight checks for the Auralis 1B run.

This does not train. It answers whether the 1B launch is allowed to proceed:

- all configured candidate train files exist
- 1B data path configs are not empty placeholders
- frozen eval prompts do not collide with train prompts/text
- a clear markdown report is written for review
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

import yaml


HELIX_USER_RE = re.compile(r"<\|user\|>\n(.*?)\n<\|end\|>", re.DOTALL)
FAST_TEXT_SCAN_BYTES = 1_000_000_000


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text).replace("\x00", " "))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    replacements = {
        "österreich": "oesterreich",
        "osterreich": "oesterreich",
        "münchen": "muenchen",
        "munchen": "muenchen",
        "früher": "frueher",
        "verläss": "verlaess",
        "weiß": "weiss",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def blake(text: str) -> str:
    return hashlib.blake2b(normalize(text).encode("utf-8", errors="replace"), digest_size=16).hexdigest()


def blake_norm(norm_text: str) -> str:
    return hashlib.blake2b(norm_text.encode("utf-8", errors="replace"), digest_size=16).hexdigest()


def fix_mojibake(text: str) -> str:
    """Best-effort repair for UTF-8 text that was decoded as latin-1."""
    try:
        return text.encode("latin1").decode("utf-8")
    except UnicodeError:
        return text


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_eval_prompts(paths: list[Path]) -> list[dict[str, str]]:
    prompts: list[dict[str, str]] = []
    for path in paths:
        data = load_yaml(path)
        for probe in data.get("probes", []):
            prompt = str(probe["prompt"])
            prompts.append(
                {
                    "file": str(path),
                    "id": str(probe.get("id", "")),
                    "prompt": prompt,
                    "hash": blake(prompt),
                    "norm": normalize(prompt),
                }
            )
    return prompts


SUBSTRING_ANCHOR_STOPWORDS = {
    "welche", "welcher", "welchem", "welchen", "welches", "warum", "wieso",
    "heute", "aktuell", "weiterhin", "stadt", "deutschland", "deutsche",
    "deutschen", "deutscher", "hauptstadt", "frage", "antwort", "kurz",
    "oder", "und", "eine", "einer", "einen", "einem", "ist", "war", "hat",
    "gilt", "soll", "sollte", "macht", "machen", "nennen", "nenne", "was",
}


def choose_substring_anchor(norm_prompt: str) -> str:
    """Choose a token that must occur if ``norm_prompt`` is a substring.

    The old implementation checked every eval prompt against every training
    line. On 80+ GB text corpora that is needlessly slow. A long, non-generic
    token from the normalized prompt is a safe prefilter: if the full prompt is
    contained in a line, the anchor token is contained as well.
    """

    tokens = [
        tok for tok in norm_prompt.split()
        if len(tok) >= 5 and tok not in SUBSTRING_ANCHOR_STOPWORDS
    ]
    if not tokens:
        tokens = [tok for tok in norm_prompt.split() if len(tok) >= 5]
    if not tokens:
        return norm_prompt.split()[0] if norm_prompt.split() else ""
    return max(tokens, key=lambda tok: (len(tok), tok))


def build_substring_anchor_index(eval_prompts: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    index: dict[str, list[dict[str, str]]] = {}
    for item in eval_prompts:
        if len(item["norm"]) < 30:
            continue
        anchor = choose_substring_anchor(item["norm"])
        if anchor:
            index.setdefault(anchor, []).append(item)
    return index


def iter_train_units(path: Path):
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, 1):
            raw = line.strip()
            if not raw:
                continue
            if suffix == ".jsonl":
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    obj = {}
                text = str(obj.get("text") or obj.get("prompt") or obj.get("content") or raw)
                match = HELIX_USER_RE.search(text)
                if match:
                    yield line_no, match.group(1).strip(), "helix_user"
                else:
                    yield line_no, text, "jsonl_text"
            else:
                yield line_no, raw, "text_line"


def literal_variants(text: str) -> list[str]:
    variants = [text, fix_mojibake(text)]
    out: list[str] = []
    seen: set[str] = set()
    for item in variants:
        item = item.strip()
        if len(item) < 8 or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def grep_large_text_file(eval_prompts: list[dict[str, str]], path: Path) -> list[dict[str, Any]]:
    """Fast literal prompt scan for huge raw text files.

    Full normalized hash/substring scanning over 80+ GB text is too slow for an
    interactive preflight. For large text corpora, use GNU grep's fixed-string
    engine against the literal prompt variants. SFT/JSONL files still use the
    stricter normalized hash path.
    """

    pattern_to_items: dict[str, list[dict[str, str]]] = {}
    for item in eval_prompts:
        for variant in literal_variants(item["prompt"]):
            pattern_to_items.setdefault(variant.casefold(), []).append(item)
    if not pattern_to_items:
        return []

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
        pattern_file = Path(tmp.name)
        for pattern in pattern_to_items:
            tmp.write(pattern + "\n")
    hits: list[dict[str, Any]] = []
    try:
        proc = subprocess.run(
            ["grep", "-F", "-i", "-n", "-H", "-f", str(pattern_file), str(path)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode not in (0, 1):
            raise RuntimeError(f"grep failed for {path}: {proc.stderr.strip()}")
        for line in proc.stdout.splitlines():
            # GNU grep with one file emits path:line:content. Paths here are
            # container POSIX paths and do not contain colons.
            parts = line.split(":", 2)
            if len(parts) != 3:
                continue
            _file, line_no_raw, content = parts
            content_cf = content.casefold()
            for pattern, items in pattern_to_items.items():
                if pattern in content_cf:
                    for item in items:
                        hits.append(
                            {
                                "probe_id": item["id"],
                                "eval_file": item["file"],
                                "train_file": str(path),
                                "line_no": int(line_no_raw) if line_no_raw.isdigit() else None,
                                "kind": "large_text_literal_grep",
                                "prompt": item["prompt"],
                            }
                        )
    finally:
        try:
            pattern_file.unlink()
        except OSError:
            pass
    return hits


def check_collisions(eval_prompts: list[dict[str, str]], train_files: list[Path]) -> dict[str, Any]:
    by_hash = {item["hash"]: item for item in eval_prompts}
    substring_index = build_substring_anchor_index(eval_prompts)
    collisions: list[dict[str, Any]] = []
    substring_hits: list[dict[str, Any]] = []
    fast_text_files: list[dict[str, Any]] = []
    scanned_units = 0
    for path in train_files:
        if not path.is_file():
            continue
        if path.suffix.lower() == ".txt" and path.stat().st_size >= FAST_TEXT_SCAN_BYTES:
            substring_hits.extend(grep_large_text_file(eval_prompts, path))
            fast_text_files.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "mode": "large_text_literal_grep",
                }
            )
            continue
        for line_no, text, kind in iter_train_units(path):
            scanned_units += 1
            norm = normalize(text)
            h = blake_norm(norm)
            if h in by_hash:
                item = by_hash[h]
                collisions.append(
                    {
                        "probe_id": item["id"],
                        "eval_file": item["file"],
                        "train_file": str(path),
                        "line_no": line_no,
                        "kind": kind,
                        "hash": h,
                        "prompt": item["prompt"],
                    }
                )
            candidates: dict[str, dict[str, str]] = {}
            for token in set(norm.split()):
                for item in substring_index.get(token, []):
                    candidates[item["id"]] = item
            for item in candidates.values():
                if item["norm"] in norm:
                    substring_hits.append(
                        {
                            "probe_id": item["id"],
                            "eval_file": item["file"],
                            "train_file": str(path),
                            "line_no": line_no,
                            "kind": kind,
                            "prompt": item["prompt"],
                        }
                    )
    return {
        "eval_prompts": len(eval_prompts),
        "train_files_scanned": len([p for p in train_files if p.is_file()]),
        "train_units_scanned": scanned_units,
        "fast_text_files_scanned": fast_text_files,
        "hash_collisions": collisions,
        "substring_hits": substring_hits,
        "passed": not collisions and not substring_hits,
    }


def resolve_data_config_path(data_root: Path, value: str) -> Path:
    candidate = Path(str(value))
    if candidate.is_absolute():
        return candidate
    return data_root / candidate


def inspect_data_path_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False, "issues": ["missing_config"]}
    data = load_yaml(path)
    data_root_raw = str(data.get("data_root") or path.parent)
    data_root = Path(data_root_raw)
    if not data_root.is_absolute():
        data_root = path.parent / data_root
    cleaned = data.get("cleaned", {}) or {}
    issues: list[str] = []
    cleaned_counts = {
        key: len(value or []) if isinstance(value, list) else 0
        for key, value in cleaned.items()
    }
    if not cleaned_counts or sum(cleaned_counts.values()) == 0:
        issues.append("cleaned_paths_empty")
    tokenized = data.get("tokenized", {}) or {}
    if not tokenized:
        issues.append("tokenized_paths_empty")

    cleaned_paths: list[dict[str, Any]] = []
    for group, values in cleaned.items():
        for value in values or []:
            resolved = resolve_data_config_path(data_root, str(value))
            cleaned_paths.append(
                {
                    "group": group,
                    "path": str(resolved),
                    "exists": resolved.is_file(),
                    "bytes": resolved.stat().st_size if resolved.is_file() else None,
                }
            )
    missing_cleaned = [item["path"] for item in cleaned_paths if not item["exists"]]
    if missing_cleaned:
        issues.append(f"cleaned_paths_missing:{missing_cleaned}")

    tokenized_paths: list[dict[str, Any]] = []
    for name, value in tokenized.items():
        resolved = resolve_data_config_path(data_root, str(value))
        idx = resolved.with_suffix(".idx") if resolved.suffix == ".bin" else None
        bin_exists = resolved.is_file()
        idx_exists = idx.is_file() if idx is not None else None
        tokenized_paths.append(
            {
                "name": name,
                "path": str(resolved),
                "exists": bin_exists,
                "bytes": resolved.stat().st_size if bin_exists else None,
                "idx_path": str(idx) if idx is not None else None,
                "idx_exists": idx_exists,
            }
        )
    missing_tokenized = [
        item["path"]
        for item in tokenized_paths
        if not item["exists"] or (item["idx_exists"] is False)
    ]
    if missing_tokenized:
        issues.append(f"tokenized_paths_missing:{missing_tokenized}")
    return {
        "path": str(path),
        "exists": True,
        "data_root": str(data_root),
        "cleaned_counts": cleaned_counts,
        "tokenized_count": len(tokenized),
        "cleaned_paths": cleaned_paths,
        "tokenized_paths": tokenized_paths,
        "issues": issues,
    }


def file_status(paths: list[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths:
        out.append(
            {
                "path": str(path),
                "exists": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else None,
            }
        )
    return out


def manifest_status(paths: list[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths:
        exists = path.is_file()
        rows = 0
        if exists:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for rows, _line in enumerate(fh, 1):
                    pass
        out.append(
            {
                "path": str(path),
                "exists": exists,
                "bytes": path.stat().st_size if exists else None,
                "rows": rows if exists else None,
                "issues": ([] if exists and rows > 0 else ["missing_or_empty_manifest"]),
            }
        )
    return out


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# Auralis 1B Readiness Preflight",
        "",
        f"- ready_to_launch: {report['ready_to_launch']}",
        f"- eval_prompts: {report['collision_check']['eval_prompts']}",
        f"- train_units_scanned: {report['collision_check']['train_units_scanned']}",
        f"- fast_text_files_scanned: {len(report['collision_check'].get('fast_text_files_scanned', []))}",
        f"- hash_collisions: {len(report['collision_check']['hash_collisions'])}",
        f"- substring_hits: {len(report['collision_check']['substring_hits'])}",
        "",
        "## Blocking Issues",
        "",
    ]
    if not report["blocking_issues"]:
        lines.append("None.")
    else:
        for issue in report["blocking_issues"]:
            lines.append(f"- {issue}")
    lines.extend(["", "## Train Files", ""])
    for item in report["train_files"]:
        size = item["bytes"] if item["bytes"] is not None else "-"
        lines.append(f"- {item['path']}: exists={item['exists']} bytes={size}")
    fast_files = report["collision_check"].get("fast_text_files_scanned", [])
    if fast_files:
        lines.extend(["", "## Large Text Scan Mode", ""])
        for item in fast_files:
            lines.append(f"- {item['path']}: mode={item['mode']} bytes={item['bytes']}")
    lines.extend(["", "## Data Path Configs", ""])
    for item in report["data_path_configs"]:
        lines.append(f"- {item['path']}: exists={item['exists']} issues={item.get('issues', [])}")
        if item.get("exists"):
            lines.append(f"  - data_root: {item.get('data_root')}")
            lines.append(f"  - cleaned_counts: {item.get('cleaned_counts', {})}")
            lines.append(f"  - tokenized_count: {item.get('tokenized_count', 0)}")
    lines.extend(["", "## Source-Disjoint Manifests", ""])
    for item in report.get("source_disjoint_manifests", []):
        lines.append(
            f"- {item['path']}: exists={item['exists']} rows={item.get('rows')} issues={item.get('issues', [])}"
        )
    lines.extend(["", "## Policy", ""])
    for key, value in report["policy"].items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-md", type=Path, required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    repo = args.config.resolve().parents[2] if "configs" in args.config.parts else Path.cwd()
    eval_files = [repo / cfg["readiness_gate"], repo / cfg["frozen_sft_gate"]]
    train_files = [repo / p for p in cfg.get("candidate_train_files", [])]
    data_path_configs = [repo / p for p in cfg.get("data_path_configs", [])]
    source_manifests = [repo / p for p in cfg.get("source_disjoint_manifests", [])]
    eval_prompts = load_eval_prompts(eval_files)
    collisions = check_collisions(eval_prompts, train_files)
    train_status = file_status(train_files)
    data_status = [inspect_data_path_config(path) for path in data_path_configs]
    source_manifest_status = manifest_status(source_manifests)

    blocking: list[str] = []
    missing = [item["path"] for item in train_status if not item["exists"]]
    if missing:
        blocking.append(f"missing_train_files:{missing}")
    if not collisions["passed"]:
        blocking.append("eval_prompt_collision_or_substring_hit")
    for item in data_status:
        if item.get("issues"):
            blocking.append(f"data_path_config_not_ready:{item['path']}:{item['issues']}")
    if (cfg.get("policy", {}) or {}).get("require_source_disjoint_manifest", False):
        for item in source_manifest_status:
            if item.get("issues"):
                blocking.append(f"source_disjoint_manifest_not_ready:{item['path']}:{item['issues']}")

    report = {
        "config": str(args.config),
        "model_config": cfg.get("model_config"),
        "policy": cfg.get("policy", {}),
        "ready_to_launch": not blocking,
        "blocking_issues": blocking,
        "train_files": train_status,
        "data_path_configs": data_status,
        "source_disjoint_manifests": source_manifest_status,
        "collision_check": collisions,
        "must_have_concepts": cfg.get("must_have_concepts", {}),
        "canary_checkpoints": cfg.get("canary_checkpoints", {}),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_md(report), encoding="utf-8")
    print(json.dumps({"ready_to_launch": report["ready_to_launch"], "blocking_issues": blocking}, ensure_ascii=False, indent=2))
    return 0 if report["ready_to_launch"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
