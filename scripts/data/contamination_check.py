"""Baseline-question contamination check.

Scans the cleaned training-text files for literal appearances of our
``eval/baseline_questions.yaml`` questions (and their expected-keyword
combinations). Any hit means: the model could trivially pass that baseline
by memorising training text, which inflates later baseline scores.

Cheap to run — single pass, substring match on a pre-built set. Reports:

- total hits per question
- first-3 source filenames + line numbers where hit
- summary: fraction of baseline items hit by any source

Writes ``data/eval/contamination_report.md``. Non-zero exit if ANY hit is
found (so CI can gate launch).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.data._common import load_paths                    # noqa: E402


def _normalise(s: str) -> str:
    # Normalise whitespace and lowercase so "Was ist 15 × 7?" matches even if
    # reproduced with extra spaces / a different quote style.
    return re.sub(r"\s+", " ", s.strip().lower())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--questions", type=Path,
                   default=REPO / "eval" / "baseline_questions.yaml")
    p.add_argument("--data-config", type=Path, default=None)
    p.add_argument("--output-md", type=Path,
                   default=REPO / "data" / "eval" / "contamination_report.md")
    p.add_argument("--min-question-length", type=int, default=20,
                   help="Skip questions shorter than this (would over-match trivially).")
    args = p.parse_args()

    qdoc = yaml.safe_load(args.questions.read_text(encoding="utf-8"))
    questions = qdoc["questions"]
    targets: list[tuple[str, str]] = []
    for q in questions:
        norm = _normalise(q["question"])
        if len(norm) >= args.min_question_length:
            targets.append((q["id"], norm))

    cfg = load_paths(args.data_config) if args.data_config else load_paths()
    data_root = Path(cfg["_data_root"])

    # Collect all cleaned txt files for all languages
    sources: list[Path] = []
    for lang in ("english", "german", "code"):
        entries = cfg["cleaned"][lang]
        if isinstance(entries, str):
            entries = [entries]
        for entry in entries:
            if any(c in entry for c in "*?["):
                sources.extend(sorted(data_root.glob(entry)))
            else:
                p_ = data_root / entry
                if p_.is_file():
                    sources.append(p_)

    hits: dict[str, list[tuple[str, int, str]]] = {qid: [] for qid, _ in targets}
    print(f"scanning {len(sources)} files against {len(targets)} questions...")

    for src in sources:
        try:
            with src.open("r", encoding="utf-8", errors="replace") as fh:
                for line_no, line in enumerate(fh, 1):
                    norm_line = _normalise(line)
                    # Cheap substring containment; good enough for the "did my
                    # literal question show up verbatim?" check.
                    for qid, qtext in targets:
                        if qtext in norm_line:
                            if len(hits[qid]) < 3:
                                hits[qid].append((src.name, line_no, line.strip()[:200]))
        except OSError as e:
            print(f"  skip {src}: {e}", file=sys.stderr)

    contaminated = [qid for qid, h in hits.items() if h]
    fraction = len(contaminated) / max(len(targets), 1)

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    md = ["# Contamination Report\n",
          f"Questions scanned  : {len(targets)}",
          f"Sources scanned    : {len(sources)}",
          f"Contaminated IDs   : {len(contaminated)}",
          f"Contamination rate : {fraction*100:.2f}%"]
    if contaminated:
        md.append("\n## Hits\n")
        for qid in contaminated:
            md.append(f"\n### `{qid}`")
            for src, ln, text in hits[qid]:
                md.append(f"- `{src}:{ln}` → {text!r}")
    args.output_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {args.output_md}")
    print(f"contaminated: {len(contaminated)}/{len(targets)}")

    if contaminated:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
