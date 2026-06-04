"""Lift raw SQuAD v2 + MS MARCO records into chat-style SFT format.

Reads:
    raw/sft/qa/squad_v2.jsonl
    raw/sft/qa/ms_marco_v2_1.jsonl

Writes:
    seeds/sft/qa/squad_v2.sft.jsonl
    seeds/sft/qa/ms_marco_v2_1.sft.jsonl
    seeds/sft/qa/qa_combined.sft.jsonl    (concatenation, shuffled by --seed)

Output format per line (standard chat-SFT):
    {
        "id":       str,
        "source":   str,        # squad_v2 / ms_marco_v2_1
        "messages": [
            {"role": "user",      "content": "..."},
            {"role": "assistant", "content": "..."}
        ],
        "meta": {
            "kind": "extractive_qa" | "generative_qa" | "unanswerable",
            "domain": str,      # e.g. wiki article title for SQuAD
            ...
        }
    }

Design choices:

- For SQuAD with `is_impossible: True`, the assistant explicitly says the
  question cannot be answered from the context. This trains the "I don't
  know" pattern, which is critical to fight hallucination.

- For SQuAD answerable items, the assistant returns the answer plus a
  short justification stitched from the surrounding sentence. This avoids
  training the model to spit out single-token replies that look unhelpful.

- For MS MARCO we always include the selected passage as user-side context.
  The assistant returns the well-formed answer (or short answer if missing).

- Records with empty user / empty assistant content are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.data._common import atomic_text_writer, now_iso  # noqa: E402

DEFAULT_IN_DIR = Path("raw/sft/qa")
DEFAULT_OUT_DIR = Path("seeds/sft/qa")


@dataclass
class Stats:
    started_at: str
    finished_at: str = ""
    in_records: int = 0
    out_records: int = 0
    dropped: dict = field(default_factory=dict)
    per_kind: dict = field(default_factory=dict)


def _bump(d: dict, k: str) -> None:
    d[k] = d.get(k, 0) + 1


# ---------------------------------------------------------------------------
# SQuAD v2 → SFT
# ---------------------------------------------------------------------------


_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _surrounding_sentence(context: str, answer: str) -> str:
    """Find the sentence in `context` that contains `answer`.
    Used so the assistant can echo a short justification."""
    if not answer or not context:
        return ""
    sentences = _SENT_BOUNDARY.split(context)
    for s in sentences:
        if answer in s:
            return s.strip()
    return ""


def _squad_to_sft(rec: dict, stats: Stats) -> dict | None:
    context = (rec.get("context") or "").strip()
    question = (rec.get("question") or "").strip()
    answer = (rec.get("answer") or "").strip()
    title = (rec.get("title") or "").strip()
    impossible = bool(rec.get("is_impossible"))

    if not context or not question:
        _bump(stats.dropped, "squad_empty_field")
        return None

    user_msg = f"Context:\n{context}\n\nQuestion: {question}"

    if impossible:
        assistant_msg = (
            "Based on the provided context, this question cannot be answered. "
            "The text does not contain the information needed to answer it."
        )
        kind = "unanswerable"
    else:
        if not answer:
            _bump(stats.dropped, "squad_no_answer_text")
            return None
        sentence = _surrounding_sentence(context, answer)
        if sentence and sentence.lower() != answer.lower() and len(sentence) > len(answer) + 20:
            assistant_msg = f"{answer}\n\n(From the context: \"{sentence}\")"
        else:
            assistant_msg = answer
        kind = "extractive_qa"

    _bump(stats.per_kind, kind)
    return {
        "id": f"squad_v2-{rec.get('id', '')}",
        "source": "squad_v2",
        "messages": [
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
        "meta": {
            "kind": kind,
            "domain": title,
            "is_impossible": impossible,
        },
    }


# ---------------------------------------------------------------------------
# MS MARCO v2.1 → SFT
# ---------------------------------------------------------------------------


def _msmarco_to_sft(rec: dict, stats: Stats) -> dict | None:
    query = (rec.get("query") or "").strip()
    answer = (rec.get("answer") or "").strip()
    passage = (rec.get("selected_passage") or "").strip()
    qtype = (rec.get("query_type") or "unknown").strip()

    if not query or not answer:
        _bump(stats.dropped, "msmarco_empty_field")
        return None
    if answer.lower() in {"no answer present.", "no answer present"}:
        _bump(stats.dropped, "msmarco_no_answer")
        return None

    if passage:
        user_msg = f"Reference passage:\n{passage}\n\nQuestion: {query}"
    else:
        user_msg = query

    _bump(stats.per_kind, "generative_qa")
    return {
        "id": f"ms_marco_v2_1-{rec.get('query_id', '')}",
        "source": "ms_marco_v2_1",
        "messages": [
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": answer},
        ],
        "meta": {
            "kind": "generative_qa",
            "domain": qtype,            # description / entity / location / numeric / person
        },
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _process_file(in_path: Path, out_path: Path, fn, stats: Stats) -> None:
    print(f"\n--- {in_path.name} -> {out_path.name} ---", flush=True)
    n_in = 0
    n_out = 0
    with in_path.open("r", encoding="utf-8") as fin, atomic_text_writer(out_path) as fout:
        for line in fin:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                _bump(stats.dropped, "bad_json")
                continue
            n_in += 1
            stats.in_records += 1
            out = fn(rec, stats)
            if out is None:
                continue
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1
            stats.out_records += 1
            if n_in % 25_000 == 0:
                print(f"    {n_in} in -> {n_out} out", flush=True)
    print(f"  {in_path.stem}: {n_in} in -> {n_out} out", flush=True)


def _shuffle_key(line: str, seed: int) -> str:
    h = hashlib.blake2b(digest_size=16, person=b"auralis-qa-sft")
    h.update(str(seed).encode("ascii"))
    h.update(b"\0")
    h.update(line.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def _shuffle_combine(out_paths: list, combined_path: Path, seed: int, bucket_count: int) -> int:
    """Concatenate and deterministically shuffle without loading all lines."""
    print(f"\n--- combining {len(out_paths)} files into {combined_path.name} (seed={seed}) ---",
          flush=True)
    bucket_count = max(16, bucket_count)
    count = 0
    with tempfile.TemporaryDirectory(prefix="qa_shuffle_", dir=combined_path.parent) as tmp:
        tmp_dir = Path(tmp)
        buckets = [
            (tmp_dir / f"bucket_{idx:04d}.tmp").open("w", encoding="utf-8", newline="\n")
            for idx in range(bucket_count)
        ]
        try:
            for p in out_paths:
                if not p.exists():
                    continue
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        key = _shuffle_key(line, seed)
                        buckets[int(key[:8], 16) % bucket_count].write(key + "\t" + line)
                        count += 1
        finally:
            for fh in buckets:
                fh.close()
        with atomic_text_writer(combined_path) as fout:
            for bucket in sorted(tmp_dir.glob("bucket_*.tmp")):
                with bucket.open("r", encoding="utf-8") as fh:
                    keyed_lines = fh.readlines()
                keyed_lines.sort()
                for keyed in keyed_lines:
                    _, line = keyed.split("\t", 1)
                    fout.write(line)
    return count


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--seed", type=int, default=42,
                   help="Shuffle seed for the combined file.")
    p.add_argument("--no-combine", action="store_true",
                   help="Skip the combined-and-shuffled output.")
    p.add_argument("--shuffle-buckets", type=int, default=256,
                   help="Temporary buckets for memory-safe deterministic combine.")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats = Stats(started_at=now_iso())
    t0 = time.time()
    out_paths = []

    squad_in = args.in_dir / "squad_v2.jsonl"
    if squad_in.exists():
        squad_out = args.out_dir / "squad_v2.sft.jsonl"
        _process_file(squad_in, squad_out, _squad_to_sft, stats)
        out_paths.append(squad_out)
    else:
        print(f"  skip: {squad_in} not found")

    msmarco_in = args.in_dir / "ms_marco_v2_1.jsonl"
    if msmarco_in.exists():
        msmarco_out = args.out_dir / "ms_marco_v2_1.sft.jsonl"
        _process_file(msmarco_in, msmarco_out, _msmarco_to_sft, stats)
        out_paths.append(msmarco_out)
    else:
        print(f"  skip: {msmarco_in} not found")

    if not args.no_combine and out_paths:
        combined = args.out_dir / "qa_combined.sft.jsonl"
        n = _shuffle_combine(out_paths, combined, args.seed, args.shuffle_buckets)
        print(f"  combined: {n} records -> {combined}", flush=True)

    stats.finished_at = now_iso()
    elapsed = time.time() - t0
    manifest = args.out_dir / "process_qa_manifest.json"
    manifest.write_text(json.dumps(asdict(stats), indent=2, ensure_ascii=False),
                        encoding="utf-8")

    print(f"\n=== SUMMARY ===")
    print(f"elapsed: {elapsed:.1f}s")
    print(f"in:      {stats.in_records}")
    print(f"out:     {stats.out_records}")
    print(f"dropped: {stats.dropped}")
    print(f"kinds:   {stats.per_kind}")
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
