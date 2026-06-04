"""Seed-Collection — picks training-candidate passages for later teacher processing.

This is NOT tokenisation. It does not touch the pretrain .bin files.
It runs over the FILTERED corpora (output of filter_quality.py) and
produces a structured pool of seed documents that a teacher-LLM (Qwen 3.6
35B A3B Apex via LocalAI) will later turn into SFT training pairs via:

  OSS-Instruct : one seed passage → 3-5 instructions answerable by it
  Evol-Instruct: start simple, iteratively add constraints / reasoning / breadth

Two output streams are designed for downstream SFT:

  content-sft   : "Q about X → A grounded in the seed passage"
                  teaches the model factual domain knowledge
  structural-sft: "raw passage → structured decomposition (Q&A, bullets,
                   causal chain)"
                  teaches the model the PROCESS of distilling content — the
                  "learn-to-learn" stream Michael asked for

This collector only stages the SEEDS. Teacher-generation, judge-filtering,
and deduplication happen in a later script (qwen_synth_sft.py).

CPU-only, read-only against the NAS. Does not compete with the GPU for the
concurrent canary run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from scripts.data._common import atomic_text_writer, load_paths  # noqa: E402


# ---------------------------------------------------------------------------
# Heuristic type detection. Deliberately coarse — the teacher LLM makes the
# final call. We just pre-sort so later sampling can target specific mixes.
# ---------------------------------------------------------------------------
_RE_CODE = re.compile(r"<\|code\|>|def \w+\(|function \w+\(|class \w+[:\(]|import \w+|```")
_RE_DATES = re.compile(r"\b(19|20)\d{2}\b|\b\d{1,2}\.\s?(Jan|Feb|März|April|Mai|Juni|Juli|Aug|Sep|Okt|Nov|Dez|January|February|March|April|May|June|July|August|September|October|November|December)\b")
_RE_CITATION = re.compile(r"\[\d+\]|\(\w+,?\s*\d{4}\)|et al\.|doi:|ISBN")
_RE_IMPERATIVE_DE = re.compile(r"\b(Schritt|Zuerst|Dann|Anschließend|Nun|Als erstes|Folge|Gib|Schreib|Führe aus)\b", re.IGNORECASE)
_RE_IMPERATIVE_EN = re.compile(r"\b(Step \d|First,|Then,|Next,|Finally,|Run|Execute|Type|Enter|Click)\b", re.IGNORECASE)
_RE_OPINION = re.compile(r"\b(I think|I believe|in my opinion|meiner Meinung nach|ich glaube|ich denke|arguably|I'd say)\b", re.IGNORECASE)
_RE_NUMERIC = re.compile(r"\b\d+(\.\d+)?\s?(%|km|kg|€|\$|°C|W/m)")


@dataclass
class SeedSignals:
    char_count: int
    word_count: int
    has_code: bool = False
    date_hits: int = 0
    citation_hits: int = 0
    imperative_hits: int = 0
    opinion_hits: int = 0
    numeric_hits: int = 0
    uppercase_ratio: float = 0.0
    symbol_ratio: float = 0.0


@dataclass
class SeedRecord:
    id: str
    source_file: str
    language: str
    detected_type: str
    content: str
    signals: SeedSignals = field(default_factory=lambda: SeedSignals(0, 0))


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def _signals(text: str) -> SeedSignals:
    char_count = len(text)
    words = text.split()
    word_count = len(words)
    up = sum(1 for c in text if c.isupper())
    sy = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return SeedSignals(
        char_count=char_count,
        word_count=word_count,
        has_code=bool(_RE_CODE.search(text)),
        date_hits=len(_RE_DATES.findall(text)),
        citation_hits=len(_RE_CITATION.findall(text)),
        imperative_hits=len(_RE_IMPERATIVE_DE.findall(text)) + len(_RE_IMPERATIVE_EN.findall(text)),
        opinion_hits=len(_RE_OPINION.findall(text)),
        numeric_hits=len(_RE_NUMERIC.findall(text)),
        uppercase_ratio=up / max(char_count, 1),
        symbol_ratio=sy / max(char_count, 1),
    )


def _classify(sig: SeedSignals) -> str:
    """Very coarse bucketing — the teacher-LLM relabels anyway.

    Priority order matters (technical beats factual for code-heavy docs, etc.).
    """
    if sig.has_code or sig.symbol_ratio > 0.15:
        return "technical"
    if sig.imperative_hits >= 2:
        return "procedural"
    if sig.citation_hits >= 2 or (sig.date_hits >= 3 and sig.numeric_hits >= 2):
        return "factual"
    if sig.opinion_hits >= 1:
        return "opinion"
    if sig.date_hits >= 1 and sig.word_count > 200:
        return "narrative"
    return "general"


# ---------------------------------------------------------------------------
# Reservoir-sampling per source → uniform pick over the source file.
# ---------------------------------------------------------------------------
def _sample_from_file(
    path: Path,
    n_target: int,
    min_chars: int,
    max_chars: int,
    language: str,
    rng: random.Random,
) -> list[SeedRecord]:
    reservoir: list[SeedRecord] = []
    n_seen = 0
    source_name = path.name
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            text = line.rstrip("\n")
            L = len(text)
            if L < min_chars or L > max_chars:
                continue
            n_seen += 1
            sig = _signals(text)
            # Skip if the line is mostly symbols/uppercase — likely menu / nav
            if sig.uppercase_ratio > 0.35 or sig.symbol_ratio > 0.40:
                continue
            record = SeedRecord(
                id=hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest(),
                source_file=source_name,
                language=language,
                detected_type=_classify(sig),
                content=text,
                signals=sig,
            )
            if len(reservoir) < n_target:
                reservoir.append(record)
            else:
                # Algorithm R (Vitter)
                j = rng.randint(0, n_seen - 1)
                if j < n_target:
                    reservoir[j] = record
    return reservoir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _load_plan(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-config", type=Path, default=None,
                        help="Path resolution config (defaults to configs/data_paths.yaml).")
    parser.add_argument("--plan", type=Path,
                        default=REPO_ROOT / "configs" / "data" / "seed_collection.yaml")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: <data_root>/seeds/<YYYY-MM-DD>")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_paths(args.data_config) if args.data_config else load_paths()
    data_root = Path(cfg["_data_root"])
    plan = _load_plan(args.plan)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_root = args.output_dir or (data_root / "seeds" / today)
    out_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    # Per-source: path, language, target count, char bounds
    sources: list[dict] = plan["sources"]
    summary: dict[str, dict] = {}

    # Group output into per-category JSONL so downstream teacher runs can
    # draw balanced batches without re-scanning.
    category_streams: dict[str, list[SeedRecord]] = {}

    for src in sources:
        path = data_root / src["path"]
        if not path.is_file():
            print(f"skip missing: {path}", file=sys.stderr)
            continue
        print(f"sampling {src['name']} ({src['language']}, target {src['target_count']})")
        n = int(src["target_count"])
        records = _sample_from_file(
            path=path,
            n_target=n,
            min_chars=int(src.get("min_chars", plan["defaults"]["min_chars"])),
            max_chars=int(src.get("max_chars", plan["defaults"]["max_chars"])),
            language=src["language"],
            rng=rng,
        )
        summary[src["name"]] = {
            "sampled": len(records),
            "types": {},
        }
        for r in records:
            summary[src["name"]]["types"].setdefault(r.detected_type, 0)
            summary[src["name"]]["types"][r.detected_type] += 1
            category_streams.setdefault(r.detected_type, []).append(r)

    if args.dry_run:
        print("\n=== DRY RUN SUMMARY ===")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    # Shuffle each category globally (prevents source-order bias in downstream batches)
    for cat in category_streams:
        rng.shuffle(category_streams[cat])

    # Write per-category JSONL
    for cat, records in category_streams.items():
        out_path = out_root / f"{cat}.jsonl"
        with atomic_text_writer(out_path) as fh:
            for r in tqdm(records, desc=f"write {cat}", unit="rec"):
                # Flatten signals into the top-level dict for downstream consumers
                obj = {
                    "id": r.id,
                    "source_file": r.source_file,
                    "language": r.language,
                    "detected_type": r.detected_type,
                    "content": r.content,
                    **{f"sig_{k}": v for k, v in asdict(r.signals).items()},
                }
                fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # Manifest
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan": str(args.plan),
        "seed": args.seed,
        "output_dir": str(out_root),
        "summary_per_source": summary,
        "totals_per_category": {cat: len(recs) for cat, recs in category_streams.items()},
        "total_seeds": sum(len(recs) for recs in category_streams.values()),
    }
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nwrote manifest {out_root / 'manifest.json'}")
    print(f"total seeds : {manifest['total_seeds']:,}")
    print(f"per category: {manifest['totals_per_category']}")


if __name__ == "__main__":
    main()
