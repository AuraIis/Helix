"""Download Q&A datasets (SQuAD v2 + MS MARCO v2.1) for SFT.

Output:
  raw/sft/qa/squad_v2.jsonl       (~150k Q-A on Wikipedia paragraphs)
  raw/sft/qa/ms_marco_v2_1.jsonl  (~1M Q-A from Bing logs, well-formed answers)

Each line is a flat JSON record. Will be lifted into chat-style SFT in a
follow-up step (process_qa_seeds.py — TODO once we see the data shape).

Pure-CPU HF streaming + disk write. Does not touch GPU.

Run (from inside the container):
    python scripts/data/download_qa_seeds.py --source squad
    python scripts/data/download_qa_seeds.py --source msmarco
    python scripts/data/download_qa_seeds.py --source all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.data._common import atomic_text_writer, now_iso  # noqa: E402

DEFAULT_OUT_DIR = Path("raw/sft/qa")


@dataclass
class DownloadManifest:
    source: str
    hf_dataset: str
    output_file: str
    started_at: str
    finished_at: str = ""
    elapsed_seconds: float = 0.0
    records_written: int = 0
    records_skipped: int = 0
    bytes_written: int = 0
    notes: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-source record converters: flatten the HF row into our jsonl schema.
# ---------------------------------------------------------------------------


def _squad_records(ex: dict) -> Iterable[dict]:
    """SQuAD v2: {id, title, context, question, answers:{text,answer_start}}."""
    answers = ex.get("answers", {}) or {}
    answer_texts = list(answers.get("text") or [])
    has_answer = len(answer_texts) > 0
    yield {
        "source": "squad_v2",
        "id": ex.get("id", ""),
        "title": ex.get("title", ""),
        "context": ex.get("context", ""),
        "question": ex.get("question", ""),
        "answer": answer_texts[0] if has_answer else "",
        "all_answers": answer_texts,
        "is_impossible": not has_answer,           # SQuAD v2 has unanswerable Qs
    }


def _msmarco_records(ex: dict) -> Iterable[dict]:
    """MS MARCO v2.1: {query, query_type, answers, wellFormedAnswers, passages}.

    We keep the well-formed answer when present (cleanest for SFT), fall back
    to the short answers otherwise. We also keep the SELECTED passage text
    so the SFT prompt can use it as grounding context.
    """
    well_formed = list(ex.get("wellFormedAnswers") or [])
    short = list(ex.get("answers") or [])
    answer = well_formed[0] if well_formed else (short[0] if short else "")
    if not answer or answer == "No Answer Present.":
        # Skip records without a useful answer — nothing to learn from.
        return

    passages = ex.get("passages", {}) or {}
    p_texts = list(passages.get("passage_text") or [])
    p_selected = list(passages.get("is_selected") or [])
    selected_text = ""
    for txt, sel in zip(p_texts, p_selected):
        if sel == 1:
            selected_text = txt
            break
    if not selected_text and p_texts:
        selected_text = p_texts[0]

    yield {
        "source": "ms_marco_v2_1",
        "query_id": int(ex.get("query_id", 0) or 0),
        "query": ex.get("query", ""),
        "query_type": ex.get("query_type", ""),
        "answer": answer,
        "all_answers_short": short,
        "all_answers_wellformed": well_formed,
        "selected_passage": selected_text,
        "all_passages": p_texts,
    }


# ---------------------------------------------------------------------------
# Source registry. Each entry mirrors the structure of download_sft_seeds.py
# but lives in its own file so we don't churn the existing pipeline.
# ---------------------------------------------------------------------------


SOURCES = {
    "squad": {
        "source": "squad_v2",
        "hf_dataset": "rajpurkar/squad_v2",
        "config": None,
        "split": "train",
        "fn": _squad_records,
        "max_records": None,                         # ~130k train, all of it
        "streaming": False,                          # SQuAD is small, fits in RAM
        "notes": "SQuAD v2 train split. Includes ~50k unanswerable (is_impossible).",
    },
    "msmarco": {
        "source": "ms_marco_v2_1",
        "hf_dataset": "ms_marco",
        "config": "v2.1",
        "split": "train",
        "fn": _msmarco_records,
        "max_records": 200_000,                      # cap at 200k for speed
        "streaming": True,                           # MS MARCO train is ~10 GB
        "notes": "MS MARCO v2.1 train, capped to 200k records with answers.",
    },
}


def _open_hf(dataset: str, config, split: str, streaming: bool):
    from datasets import load_dataset

    return load_dataset(dataset, config, split=split, streaming=streaming, trust_remote_code=True)


def _download_one(spec: dict, out_dir: Path) -> DownloadManifest:
    out_file = out_dir / f"{spec['source']}.jsonl"
    manifest = DownloadManifest(
        source=spec["source"],
        hf_dataset=spec["hf_dataset"],
        output_file=str(out_file),
        started_at=now_iso(),
        notes=[spec.get("notes", "")],
    )
    t0 = time.time()
    max_records = spec.get("max_records")

    print(f"\n=== [{spec['source']}] -> {out_file} ===", flush=True)
    print(f"  hf: {spec['hf_dataset']} (config={spec['config']}, split={spec['split']})", flush=True)
    print(f"  max: {max_records or 'all'}  streaming: {spec.get('streaming', True)}", flush=True)

    try:
        ds = _open_hf(spec["hf_dataset"], spec["config"], spec["split"],
                      streaming=spec.get("streaming", True))
    except Exception as e:  # noqa: BLE001
        manifest.notes.append(f"LOAD FAILED: {type(e).__name__}: {e}")
        manifest.finished_at = now_iso()
        manifest.elapsed_seconds = round(time.time() - t0, 1)
        print(f"  FAILED to load: {e}", file=sys.stderr)
        return manifest

    with atomic_text_writer(out_file) as fh:
        for ex in tqdm(ds, desc=spec["source"], unit="rec"):
            for rec in spec["fn"](ex):
                if rec is None:
                    manifest.records_skipped += 1
                    continue
                line = json.dumps(rec, ensure_ascii=False) + "\n"
                fh.write(line)
                manifest.records_written += 1
                manifest.bytes_written += len(line.encode("utf-8"))
                if max_records and manifest.records_written >= max_records:
                    break
            if max_records and manifest.records_written >= max_records:
                break

    manifest.finished_at = now_iso()
    manifest.elapsed_seconds = round(time.time() - t0, 1)
    manifest_path = out_file.with_suffix(out_file.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(asdict(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  done: {manifest.records_written} records, "
          f"{manifest.bytes_written / 1e6:.1f} MB, "
          f"{manifest.elapsed_seconds:.0f}s", flush=True)
    print(f"  manifest: {manifest_path}", flush=True)
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True,
                   choices=list(SOURCES) + ["all"])
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "all":
        keys = list(SOURCES)
    else:
        keys = [args.source]

    for key in keys:
        spec = SOURCES[key]
        try:
            _download_one(spec, args.output_dir)
        except KeyboardInterrupt:
            print("\ninterrupted by user — partial output kept", file=sys.stderr)
            sys.exit(130)


if __name__ == "__main__":
    main()
