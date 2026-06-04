"""Download curated SFT-seed datasets from HuggingFace.

Single entry point for three separate SFT-seed domains, selectable via
``--domain``:

- ``coding``        : code + tests + competitive programming (APPS, MBPP,
                      HumanEval, CodeContests). Used to build the
                      "code + verify" stream.
- ``prompting``     : prompt engineering examples (awesome-chatgpt-prompts,
                      LIMA). Used to teach task / instruction framing.
- ``troubleshoot``  : Stack Exchange Q&A preferences (Michael's W1.6).
                      Used to build structured diagnose → verify chains.

Design:
- Everything goes into ``<data_root>/raw/sft/<domain>/<source>.jsonl``.
- Each line is a small flat JSON record with source-specific keys.
- Uses HF's ``datasets`` streaming where available; falls back to a
  regular load if streaming is not supported by the dataset.
- Writes atomically (``.tmp`` → rename) + a ``*.manifest.json`` sidecar.

Does NOT touch the pretrain .bin files. Does NOT compete with the GPU —
pure-CPU HF streaming + disk write.

Run (from inside the container)::

    python scripts/data/download_sft_seeds.py --domain coding
    python scripts/data/download_sft_seeds.py --domain prompting
    python scripts/data/download_sft_seeds.py --domain troubleshoot
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.data._common import atomic_text_writer, load_paths, now_iso  # noqa: E402


@dataclass
class DownloadManifest:
    domain: str
    source: str
    hf_dataset: str
    hf_config: str | None
    split: str
    output_file: str
    records_written: int = 0
    records_skipped: int = 0
    bytes_written: int = 0
    started_at: str = ""
    finished_at: str = ""
    elapsed_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)


def _open_hf(dataset: str, config: str | None, split: str, streaming: bool = True):
    from datasets import load_dataset
    kwargs: dict[str, Any] = {"split": split}
    if streaming:
        kwargs["streaming"] = True
    if config:
        return load_dataset(dataset, name=config, **kwargs)
    return load_dataset(dataset, **kwargs)


# ---------------------------------------------------------------------------
# Per-source record shaping. Each returns an iterable of dicts; each dict is
# one JSONL line. We try to preserve the *useful* fields from the source and
# drop noise (hashes, huge ids, redundant wrappers).
# ---------------------------------------------------------------------------

def _apps_records(ex: dict) -> Iterable[dict]:
    # codeparrot/apps: {problem_id, question, solutions, input_output,
    #                   difficulty, starter_code}
    yield {
        "source": "apps",
        "problem_id": ex.get("problem_id"),
        "difficulty": ex.get("difficulty"),
        "question": ex.get("question", ""),
        "starter_code": ex.get("starter_code", "") or "",
        # solutions is a JSON-encoded list of strings; keep raw for downstream
        "solutions_raw": ex.get("solutions", ""),
        "input_output_raw": ex.get("input_output", ""),  # the test cases!
    }


def _mbpp_records(ex: dict) -> Iterable[dict]:
    # mbpp: {task_id, text (problem), code (solution), test_list (tests),
    #        test_setup_code, challenge_test_list}
    yield {
        "source": "mbpp",
        "task_id": ex.get("task_id"),
        "prompt": ex.get("text", ""),
        "solution": ex.get("code", ""),
        "test_list": ex.get("test_list", []),
        "test_setup_code": ex.get("test_setup_code", "") or "",
    }


def _humaneval_records(ex: dict) -> Iterable[dict]:
    # openai_humaneval: {task_id, prompt, canonical_solution, test,
    #                    entry_point}
    yield {
        "source": "humaneval",
        "task_id": ex.get("task_id"),
        "prompt": ex.get("prompt", ""),
        "canonical_solution": ex.get("canonical_solution", ""),
        "test": ex.get("test", ""),
        "entry_point": ex.get("entry_point", ""),
    }


def _codecontests_records(ex: dict) -> Iterable[dict]:
    # deepmind/code_contests has huge records; keep the useful slice.
    yield {
        "source": "code_contests",
        "name": ex.get("name"),
        "description": ex.get("description", ""),
        "difficulty": ex.get("difficulty"),
        "cf_rating": ex.get("cf_rating"),
        # Test inputs/outputs (first 3 only — full set is huge)
        "public_tests": (ex.get("public_tests", {}) or {}),
        "private_tests_first3": {
            "input": (ex.get("private_tests", {}) or {}).get("input", [])[:3],
            "output": (ex.get("private_tests", {}) or {}).get("output", [])[:3],
        },
        # correct solutions snippet (first 2)
        "solutions_first2": {
            "language": (ex.get("solutions", {}) or {}).get("language", [])[:2],
            "solution": (ex.get("solutions", {}) or {}).get("solution", [])[:2],
        },
    }


def _awesome_prompts_records(ex: dict) -> Iterable[dict]:
    # fka/awesome-chatgpt-prompts: {act, prompt}
    yield {
        "source": "awesome_chatgpt_prompts",
        "act": ex.get("act", ""),
        "prompt": ex.get("prompt", ""),
    }


def _lima_records(ex: dict) -> Iterable[dict]:
    # GAIR/lima: {conversations (list of strings), source}
    yield {
        "source": "lima",
        "origin": ex.get("source", ""),
        "conversations": ex.get("conversations", []),
    }


def _stackex_preferences_records(ex: dict) -> Iterable[dict]:
    # HuggingFaceH4/stack-exchange-preferences:
    #   {question, answers (list of {text, pm_score, selected}), ...}
    q = ex.get("question", "") or ""
    answers = ex.get("answers", []) or []
    if not q or not answers:
        return
    # Sort by pm_score descending; keep top + bottom for preference pair
    answers_sorted = sorted(
        answers, key=lambda a: a.get("pm_score", 0) or 0, reverse=True,
    )
    yield {
        "source": "stack_exchange_preferences",
        "question": q,
        "answer_accepted": answers_sorted[0].get("text", ""),
        "answer_accepted_score": answers_sorted[0].get("pm_score", 0),
        "answer_rejected": answers_sorted[-1].get("text", "") if len(answers_sorted) > 1 else "",
        "answer_rejected_score": answers_sorted[-1].get("pm_score", 0) if len(answers_sorted) > 1 else None,
        "metadata_url": ex.get("metadata", [""])[0] if ex.get("metadata") else "",
    }


# ---------------------------------------------------------------------------
# Dataset registry per domain. Each entry: (source_key, hf_dataset, hf_config,
# split, record_fn, max_records_or_None).
# ---------------------------------------------------------------------------

DOMAINS: dict[str, list[dict[str, Any]]] = {
    "coding": [
        {"source": "apps",         "hf_dataset": "codeparrot/apps",
         "config": "all",          "split": "train",
         "fn": _apps_records,      "max_records": 10_000,
         "streaming": True,
         "notes": "Problem + tests + solutions. Gold for code-verify SFT."},
        {"source": "mbpp",         "hf_dataset": "mbpp",
         "config": "full",         "split": "train",
         "fn": _mbpp_records,      "max_records": None,
         "streaming": False,
         "notes": "974 Python problems with pytest tests."},
        {"source": "humaneval",    "hf_dataset": "openai_humaneval",
         "config": None,           "split": "test",
         "fn": _humaneval_records, "max_records": None,
         "streaming": False,
         "notes": "164 hand-written tasks, classic benchmark."},
        {"source": "code_contests","hf_dataset": "deepmind/code_contests",
         "config": None,           "split": "train",
         "fn": _codecontests_records, "max_records": 5_000,
         "streaming": True,
         "notes": "Competitive-programming problems with I/O tests."},
    ],
    "prompting": [
        {"source": "awesome_chatgpt_prompts", "hf_dataset": "fka/awesome-chatgpt-prompts",
         "config": None,           "split": "train",
         "fn": _awesome_prompts_records, "max_records": None,
         "streaming": False,
         "notes": "Curated role prompts, tiny."},
        {"source": "lima",         "hf_dataset": "GAIR/lima",
         "config": None,           "split": "train",
         "fn": _lima_records,      "max_records": None,
         "streaming": False,
         "notes": "1k super-high-quality SFT samples."},
    ],
    "troubleshoot": [
        {"source": "stackex_preferences",
         "hf_dataset": "HuggingFaceH4/stack-exchange-preferences",
         "config": None,           "split": "train",
         "fn": _stackex_preferences_records, "max_records": 100_000,
         "streaming": True,
         "notes": "Q&A across SE network incl. superuser/serverfault/askubuntu. "
                  "Perfect for Problem → Diagnose → Solution chains."},
    ],
}


def _download_one(spec: dict, out_dir: Path) -> DownloadManifest:
    out_file = out_dir / f"{spec['source']}.jsonl"
    manifest = DownloadManifest(
        domain="", source=spec["source"],
        hf_dataset=spec["hf_dataset"], hf_config=spec["config"],
        split=spec["split"], output_file=str(out_file),
        started_at=now_iso(),
        notes=[spec.get("notes", "")],
    )
    t0 = time.time()
    tmp_file = out_file.with_suffix(out_file.suffix + ".tmp")
    max_records = spec.get("max_records")

    print(f"\n=== [{spec['source']}] → {out_file} ===")
    print(f"  hf: {spec['hf_dataset']} (config={spec['config']}, split={spec['split']})")
    print(f"  max: {max_records or 'all'}  streaming: {spec.get('streaming', True)}")

    try:
        ds = _open_hf(spec["hf_dataset"], spec["config"], spec["split"],
                      streaming=spec.get("streaming", True))
    except Exception as e:                                     # noqa: BLE001
        manifest.notes.append(f"LOAD FAILED: {type(e).__name__}: {e}")
        manifest.finished_at = now_iso()
        manifest.elapsed_seconds = round(time.time() - t0, 1)
        print(f"  FAILED to load: {e}", file=sys.stderr)
        return manifest

    fn: Callable[[dict], Iterable[dict]] = spec["fn"]
    with atomic_text_writer(out_file) as fh:
        try:
            for ex in tqdm(ds, desc=spec["source"], unit="rec"):
                for rec in fn(ex):
                    if rec is None:
                        manifest.records_skipped += 1
                        continue
                    line = json.dumps(rec, ensure_ascii=False)
                    fh.write(line + "\n")
                    manifest.records_written += 1
                    manifest.bytes_written += len(line) + 1
                if max_records and manifest.records_written >= max_records:
                    break
        except KeyboardInterrupt:
            manifest.notes.append("INTERRUPTED")
            raise

    manifest.finished_at = now_iso()
    manifest.elapsed_seconds = round(time.time() - t0, 1)
    # Clean up stray tmp file if atomic_text_writer didn't
    if tmp_file.exists():
        tmp_file.unlink(missing_ok=True)
    out_file.with_suffix(out_file.suffix + ".manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  wrote {manifest.records_written:,} records, "
          f"{manifest.bytes_written/1e6:.1f} MB, "
          f"{manifest.elapsed_seconds:.1f}s")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", required=True, choices=list(DOMAINS))
    p.add_argument("--data-config", type=Path, default=None)
    p.add_argument("--sources", nargs="+", default=None,
                   help="Pick specific sources in the domain. Default: all.")
    args = p.parse_args()

    cfg = load_paths(args.data_config) if args.data_config else load_paths()
    data_root = Path(cfg["_data_root"])
    out_dir = data_root / "raw" / "sft" / args.domain
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = DOMAINS[args.domain]
    if args.sources:
        wanted = set(args.sources)
        specs = [s for s in specs if s["source"] in wanted]
        if not specs:
            sys.exit(f"no matching sources in domain {args.domain}: {wanted}")

    summaries: list[DownloadManifest] = []
    for spec in specs:
        m = _download_one(spec, out_dir)
        m.domain = args.domain
        summaries.append(m)

    print("\n=== Summary ===")
    for s in summaries:
        print(f"  {s.source:30s} {s.records_written:>10,} rec  "
              f"{s.bytes_written/1e6:>7.1f} MB  "
              f"{s.elapsed_seconds:>6.1f}s")


if __name__ == "__main__":
    main()
