"""Download Code pretraining data for Phase 1.

Sources (see ``Doc/SPEC_DATASETS.md`` §2.3):

- StarCoderData subset — 1B tokens across 9 languages with per-language shares
- Proof-Pile-2        — 250M tokens (math / formal reasoning)

StarCoderData is massive (>250B tokens); we stream and stop per language when
the per-language quota is filled.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterator

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.data._common import (
    DownloadStats,
    atomic_text_writer,
    check_free_space,
    clean_text,
    load_paths,
    now_iso,
    resolve,
)

# Code is token-dense (whitespace / symbols).
CODE_BYTES_PER_TOKEN = 3.5

LANG_SHARES: dict[str, float] = {
    "python":     0.30,
    "javascript": 0.20,
    "typescript": 0.10,
    "rust":       0.10,
    "cpp":        0.10,
    "go":         0.08,
    "java":       0.07,
    "shell":      0.03,
    "sql":        0.02,
}


def _parse_target_overrides(values: list[str] | None) -> dict[str, int]:
    """Parse CLI overrides of the form ``source=123456``."""
    overrides: dict[str, int] = {}
    for raw in values or []:
        if "=" not in raw:
            raise ValueError(f"Invalid target override {raw!r}; expected source=tokens.")
        name, value = raw.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Invalid target override {raw!r}; empty source name.")
        overrides[name] = int(value.replace("_", "").strip())
    return overrides


def _open_streaming(path: str, **kwargs: Any):
    from datasets import load_dataset
    return load_dataset(path, split="train", streaming=True, **kwargs)


def _stream_starcoder_language(lang: str, filters: dict[str, Any]) -> Iterator[tuple[str, dict[str, int]]]:
    ds = _open_streaming("bigcode/starcoderdata", data_dir=lang)
    min_len = filters["min_length"]
    max_len = filters["max_length"]
    min_stars = filters["min_stars"]
    for ex in ds:
        stars = ex.get("max_stars_count", 0) or 0
        if stars < min_stars:
            yield None, {"low_stars": 1}
            continue
        content = ex.get("content", "") or ""
        if len(content) < min_len:
            yield None, {"too_short": 1}
            continue
        if len(content) > max_len:
            yield None, {"too_long": 1}
            continue
        # Keep newlines for code — they are semantically meaningful.
        yield content.replace("\r\n", "\n"), {}


def _download_starcoder(
    out_dir: Path,
    target_tokens: int,
    filters: dict[str, Any],
) -> DownloadStats:
    output_path = out_dir / "starcoderdata.txt"
    stats = DownloadStats(
        source="starcoderdata",
        output_file=str(output_path),
        target_tokens=target_tokens,
        estimated_bytes_per_token=CODE_BYTES_PER_TOKEN,
        started_at=now_iso(),
        filters_applied={**filters, "lang_shares": LANG_SHARES},
    )
    total_target_bytes = stats.target_bytes()

    with atomic_text_writer(output_path) as fh:
        for lang, share in LANG_SHARES.items():
            lang_target_bytes = int(total_target_bytes * share)
            lang_bytes = 0
            for line, reasons in tqdm(
                _stream_starcoder_language(lang, filters),
                desc=f"starcoder:{lang}",
                unit="file",
            ):
                if line is None:
                    for reason, count in reasons.items():
                        key = f"{lang}:{reason}"
                        stats.filtered_reasons[key] = stats.filtered_reasons.get(key, 0) + count
                        stats.filtered_total += count
                    continue
                # Wrap each file with a language marker so the tokenizer and
                # downstream model see a clear boundary.
                wrapped = f"<|code|>[{lang}]\n{line}\n<|endcode|>"
                fh.write(wrapped + "\n")
                sz = len(wrapped.encode("utf-8")) + 1
                stats.final_bytes += sz
                stats.final_docs += 1
                lang_bytes += sz
                if lang_bytes >= lang_target_bytes:
                    break

    stats.finished_at = now_iso()
    stats.write_manifest(output_path.with_suffix(".txt.manifest.json"))
    return stats


def _stream_the_stack_v2(filters: dict[str, Any]) -> Iterator[tuple[str, dict[str, int]]]:
    """bigcode/the-stack-v2 — successor to StarCoderData, license-clean.

    Note: this is the full v2 release; for Phase 1 we only want the
    language-filtered subset. The per-language proportions in LANG_SHARES
    above are honoured at the caller level (_download_the_stack_v2).
    """
    for lang in LANG_SHARES:
        # The Stack v2 uses data_dir per-language e.g. "python", "javascript"
        ds = _open_streaming("bigcode/the-stack-v2", data_dir=lang)
        min_len = filters["min_length"]
        max_len = filters["max_length"]
        min_stars = filters["min_stars"]
        for ex in ds:
            stars = ex.get("star_count", 0) or ex.get("max_stars_count", 0) or 0
            if stars < min_stars:
                yield None, {f"{lang}:low_stars": 1}
                continue
            content = ex.get("content", "") or ""
            if len(content) < min_len:
                yield None, {f"{lang}:too_short": 1}
                continue
            if len(content) > max_len:
                yield None, {f"{lang}:too_long": 1}
                continue
            wrapped = f"<|code|>[{lang}]\n{content}\n<|endcode|>"
            yield wrapped.replace("\r\n", "\n"), {}


def _download_the_stack_v2(
    out_dir: Path,
    target_tokens: int,
    filters: dict[str, Any],
) -> DownloadStats:
    """Wrapper that splits the token budget proportionally across LANG_SHARES
    via the same per-language streaming used in _download_starcoder."""
    output_path = out_dir / "the_stack_v2.txt"
    stats = DownloadStats(
        source="the_stack_v2",
        output_file=str(output_path),
        target_tokens=target_tokens,
        estimated_bytes_per_token=CODE_BYTES_PER_TOKEN,
        started_at=now_iso(),
        filters_applied={**filters, "lang_shares": LANG_SHARES},
    )
    total_target_bytes = stats.target_bytes()
    with atomic_text_writer(output_path) as fh:
        for lang, share in LANG_SHARES.items():
            lang_target_bytes = int(total_target_bytes * share)
            lang_bytes = 0
            ds = _open_streaming("bigcode/the-stack-v2", data_dir=lang)
            for ex in tqdm(ds, desc=f"the_stack_v2:{lang}", unit="file"):
                stars = ex.get("star_count", 0) or ex.get("max_stars_count", 0) or 0
                if stars < filters["min_stars"]:
                    stats.filtered_reasons[f"{lang}:low_stars"] = stats.filtered_reasons.get(f"{lang}:low_stars", 0) + 1
                    stats.filtered_total += 1
                    continue
                content = ex.get("content", "") or ""
                if len(content) < filters["min_length"] or len(content) > filters["max_length"]:
                    stats.filtered_reasons[f"{lang}:length"] = stats.filtered_reasons.get(f"{lang}:length", 0) + 1
                    stats.filtered_total += 1
                    continue
                wrapped = f"<|code|>[{lang}]\n{content}\n<|endcode|>"
                fh.write(wrapped + "\n")
                sz = len(wrapped.encode("utf-8")) + 1
                stats.final_bytes += sz
                stats.final_docs += 1
                lang_bytes += sz
                if lang_bytes >= lang_target_bytes:
                    break
    stats.finished_at = now_iso()
    stats.write_manifest(output_path.with_suffix(".txt.manifest.json"))
    return stats


def _stream_open_web_math(_filters: dict[str, Any]) -> Iterator[tuple[str, dict[str, int]]]:
    """Replaces Proof-Pile-2 (script-based, unsupported by datasets v4+).

    open-web-math/open-web-math is the standalone parquet release of the
    math-web-content subset that used to live inside Proof-Pile-2. Same
    math/formal-reasoning flavour.
    """
    ds = _open_streaming("open-web-math/open-web-math")
    for ex in ds:
        text = ex.get("text", "") or ex.get("content", "") or ""
        if len(text) < 200:
            yield None, {"too_short": 1}
            continue
        yield clean_text(text), {}


def _download_open_web_math(out_dir: Path, target_tokens: int, filters: dict[str, Any]) -> DownloadStats:
    output_path = out_dir / "open_web_math.txt"
    stats = DownloadStats(
        source="open_web_math",
        output_file=str(output_path),
        target_tokens=target_tokens,
        estimated_bytes_per_token=CODE_BYTES_PER_TOKEN,
        started_at=now_iso(),
        filters_applied=filters,
    )
    target_bytes = stats.target_bytes()
    with atomic_text_writer(output_path) as fh:
        for line, reasons in tqdm(_stream_open_web_math(filters), desc="open_web_math", unit="doc"):
            if line is None:
                for reason, count in reasons.items():
                    stats.filtered_reasons[reason] = stats.filtered_reasons.get(reason, 0) + count
                    stats.filtered_total += count
                continue
            fh.write(line + "\n")
            stats.final_bytes += len(line.encode("utf-8")) + 1
            stats.final_docs += 1
            if stats.final_bytes >= target_bytes:
                break
    stats.finished_at = now_iso()
    stats.write_manifest(output_path.with_suffix(".txt.manifest.json"))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Phase 1 Code pretraining data.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--sources", nargs="+",
                        choices=["starcoder", "the_stack_v2", "open_web_math"],
                        default=["starcoder", "open_web_math"])
    parser.add_argument(
        "--target-tokens-override",
        nargs="*",
        default=None,
        help="Per-source token override(s), e.g. the_stack_v2=1700000000 open_web_math=300000000",
    )
    parser.add_argument("--required-free-gb", type=float, default=8.0)
    args = parser.parse_args()

    cfg = load_paths(args.config) if args.config else load_paths()
    out_dir = resolve(cfg, "raw", "code")
    out_dir.mkdir(parents=True, exist_ok=True)
    check_free_space(out_dir, args.required_free_gb)

    targets = dict(cfg["phase1_targets"]["code"])
    targets.update(_parse_target_overrides(args.target_tokens_override))
    filters = cfg["filters"]["code"]

    summaries: list[DownloadStats] = []
    if "starcoder" in args.sources:
        summaries.append(_download_starcoder(out_dir, int(targets["starcoder"]), filters))
    if "the_stack_v2" in args.sources:
        # Reuse the starcoder budget unless a dedicated target is set in config.
        tgt = int(targets.get("the_stack_v2", targets.get("starcoder", 1_000_000_000)))
        summaries.append(_download_the_stack_v2(out_dir, tgt, filters))
    if "open_web_math" in args.sources:
        summaries.append(_download_open_web_math(out_dir, int(targets["open_web_math"]), filters))

    print("\n=== Summary ===")
    for s in summaries:
        print(f"  {s.source:15s} {s.final_docs:>10,} docs  {s.final_bytes/1e9:>6.2f} GB")


if __name__ == "__main__":
    main()
