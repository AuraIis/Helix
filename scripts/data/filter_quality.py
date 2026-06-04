"""Apply lightweight quality filters to a text corpus file.

This is a practical local-quality pass for already-downloaded corpora.
It is intentionally simple and CPU-friendly:

- normalise encoding / whitespace (for non-code)
- reject very short / very long lines
- reject URL-dense / symbol-dense garbage
- reject obvious mojibake and boilerplate-heavy lines
- reject extremely repetitive lines

The script is useful before tokenisation and before final mixing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Let the script run via `python scripts/data/filter_quality.py` without an
# editable install: make the repo root importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.data._common import atomic_text_writer, clean_text, now_iso


BOILERPLATE_PATTERNS = (
    # English web chrome
    "cookie policy",
    "privacy policy",
    "accept all cookies",
    "subscribe to our newsletter",
    "all rights reserved",
    "sign in to continue",
    "javascript is disabled",
    # German web chrome — the corpus is German-heavy but the list used to be
    # English-only, so German cookie banners / shop / login boilerplate slipped
    # through unfiltered. Multi-word phrases only, to avoid nuking legit prose.
    "diese website verwendet cookies",
    "wir verwenden cookies",
    "cookies akzeptieren",
    "alle cookies akzeptieren",
    "alle rechte vorbehalten",
    "newsletter abonnieren",
    "javascript ist deaktiviert",
    "bitte aktivieren sie javascript",
    "in den warenkorb",
    "zur kasse",
    "passwort vergessen",
    "anmelden oder registrieren",
)


@dataclass
class FilterManifest:
    input_file: str
    output_file: str
    language: str
    preserve_newlines: bool
    started_at: str
    finished_at: str = ""
    lines_in: int = 0
    lines_written: int = 0
    bytes_written: int = 0
    dropped: dict[str, int] = field(default_factory=dict)


def _drop(manifest: FilterManifest, reason: str) -> None:
    manifest.dropped[reason] = manifest.dropped.get(reason, 0) + 1


def _url_density(line: str) -> float:
    tokens = line.split()
    if not tokens:
        return 0.0
    url_like = sum(1 for token in tokens if token.startswith(("http://", "https://", "www.")))
    return url_like / len(tokens)


def _symbol_density(line: str) -> float:
    if not line:
        return 0.0
    symbol_like = sum(1 for ch in line if not ch.isalnum() and not ch.isspace())
    return symbol_like / len(line)


def _repetition_score(line: str) -> float:
    tokens = line.split()
    if not tokens:
        return 0.0
    unique = len(set(tokens))
    return 1.0 - (unique / len(tokens))


def _looks_mojibake(line: str) -> bool:
    return any(marker in line for marker in ("â€™", "â€œ", "â€", "Ã¼", "Ã¶", "Ã¤", "Â "))


def _normalise(line: str, preserve_newlines: bool) -> str:
    line = line.replace("\x00", "").replace("\r", " ")
    return line.rstrip("\n") if preserve_newlines else clean_text(line)


# Markers we never drop, regardless of length — they are file/code boundaries
# that the tokenizer/model needs to keep intact. Adding to this list is cheap.
PROTECTED_PREFIXES = (
    "<|code|>",
    "<|endcode|>",
    "<filename>",
    "<file_sep>",
    "<|file|>",
)


def _passes(
    line: str,
    *,
    min_length: int,
    max_length: int,
    preserve_newlines: bool,
    allow_mojibake: bool,
    max_repetition: float = 0.60,
) -> str | None:
    normalized = _normalise(line, preserve_newlines=preserve_newlines)

    # Boundary markers: only check too_long, otherwise let through.
    if any(normalized.startswith(p) for p in PROTECTED_PREFIXES):
        if len(normalized) > max_length:
            return "too_long"
        return None

    if len(normalized) < min_length:
        return "too_short"
    if len(normalized) > max_length:
        return "too_long"
    lower = normalized.lower()
    if any(pattern in lower for pattern in BOILERPLATE_PATTERNS):
        return "boilerplate"
    if _url_density(normalized) > 0.20:
        return "url_dense"
    if _symbol_density(normalized) > 0.35:
        return "symbol_dense"
    if _repetition_score(normalized) > max_repetition:
        return "repetitive"
    if not allow_mojibake and _looks_mojibake(normalized):
        return "mojibake"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", choices=["english", "german", "code"], required=True)
    parser.add_argument("--min-length", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--allow-mojibake", action="store_true")
    parser.add_argument("--max-repetition", type=float, default=0.60,
                        help="Reject lines where this fraction of tokens are duplicates. "
                             "Default 0.60 (strict). Math/code with formal-repetitive content "
                             "benefits from 0.80-0.85.")
    args = parser.parse_args()

    # Per-language length defaults: (min, max, preserve_newlines).
    # Code: lowered min from 50 to 10 so imports / short statements / closing
    # braces stay in. Pre-fix the filter wiped 78.8 percent of starcoderdata
    # because a typical code line is much shorter than 50 chars.
    defaults = {
        "english": (200, 100_000, False),
        "german": (300, 100_000, False),
        "code": (10, 30_000, True),
    }
    min_length, max_length, preserve_newlines = defaults[args.language]
    if args.min_length is not None:
        min_length = args.min_length
    if args.max_length is not None:
        max_length = args.max_length

    manifest = FilterManifest(
        input_file=str(args.input),
        output_file=str(args.output),
        language=args.language,
        preserve_newlines=preserve_newlines,
        started_at=now_iso(),
    )

    with atomic_text_writer(args.output) as out_fh, args.input.open(
        "r", encoding="utf-8", errors="replace"
    ) as in_fh:
        for line in in_fh:
            manifest.lines_in += 1
            reason = _passes(
                line,
                min_length=min_length,
                max_length=max_length,
                preserve_newlines=preserve_newlines,
                allow_mojibake=args.allow_mojibake,
                max_repetition=args.max_repetition,
            )
            if reason is not None:
                _drop(manifest, reason)
                continue
            normalized = _normalise(line, preserve_newlines=preserve_newlines)
            out_fh.write(normalized + "\n")
            manifest.lines_written += 1
            manifest.bytes_written += len((normalized + "\n").encode("utf-8"))

    manifest.finished_at = now_iso()
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
