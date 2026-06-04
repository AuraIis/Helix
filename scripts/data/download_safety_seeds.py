"""Download curated safety + jailbreak-resistance + behavior SFT seeds from HF.

This is a license-aware, audit-trail-producing companion to download_sft_seeds.py
that focuses specifically on the SAFETY / refusal / jailbreak-recognition
domain. Every dataset in the registry has been pre-checked against the
license-class filter (default: 'commercial' — only MIT, Apache, CC0, CC-BY,
ODC-BY).

Output:
    raw/sft/safety/<source>.jsonl
    raw/sft/safety/<source>.jsonl.manifest.json

Each downloaded dataset gets a manifest file documenting the license literal
that HuggingFace reports for it, the row count, the timestamp. This is the
audit evidence you want to keep around proving diligent sourcing.

Usage (from inside the container):
    # All commercial-safe sources at once:
    python scripts/data/download_safety_seeds.py --source all

    # Just one specific source:
    python scripts/data/download_safety_seeds.py --source hh_rlhf

    # See what's in the registry without downloading:
    python scripts/data/download_safety_seeds.py --list
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.data._common import atomic_text_writer, now_iso  # noqa: E402

DEFAULT_OUT_DIR = Path("raw/sft/safety")


@dataclass
class DownloadManifest:
    source: str
    hf_dataset: str
    hf_config: str | None
    split: str
    license_documented: str
    license_class: str
    output_file: str
    started_at: str
    finished_at: str = ""
    elapsed_seconds: float = 0.0
    records_written: int = 0
    records_skipped: int = 0
    bytes_written: int = 0
    notes: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-source record converters: flatten raw HF row into our shared shape.
#
# Common output schema (per line):
#   {
#     "source":      str,
#     "kind":        "preference" | "chat" | "harmful_request" | "behavior",
#     "user":        str,           # what the user said / would say
#     "preferred":   str,           # the response we want the model to give
#     "rejected":    str,           # optional: a worse / harmful response to avoid
#     "category":    str,           # source-specific tag (e.g. "harmful", "helpful")
#     "raw":         dict,          # the original row, for full auditability
#   }
# Only "source" is required; other fields filled when meaningful.
# ---------------------------------------------------------------------------


def _hh_rlhf_records(ex: dict) -> Iterable[dict]:
    """Anthropic HH-RLHF: {chosen, rejected} as full dialogue strings.
    'chosen' = the helpful or harmless reply, 'rejected' = the worse one."""
    chosen = ex.get("chosen", "") or ""
    rejected = ex.get("rejected", "") or ""
    # Heuristic: last "Human:" turn = the request; everything after = response.
    last_human = chosen.rfind("\n\nHuman:")
    if last_human < 0:
        last_human = chosen.rfind("Human:")
    user_text = ""
    chosen_response = chosen
    rejected_response = rejected
    if last_human >= 0:
        # last user turn lives until next "Assistant:" inside chosen
        tail = chosen[last_human:]
        a_idx = tail.find("Assistant:")
        if a_idx > 0:
            user_text = tail[:a_idx].replace("Human:", "").strip()
            chosen_response = tail[a_idx + len("Assistant:"):].strip()
            rej_a = rejected.rfind("Assistant:")
            if rej_a >= 0:
                rejected_response = rejected[rej_a + len("Assistant:"):].strip()
    yield {
        "source": "hh_rlhf",
        "kind": "preference",
        "user": user_text,
        "preferred": chosen_response,
        "rejected": rejected_response,
        "category": "helpful_or_harmless",
        "raw": ex,
    }


def _oasst_records(ex: dict) -> Iterable[dict]:
    """OpenAssistant: tree-structured messages. Emit message-level rows.
    One row per message with role + parent. Threading happens at consumer side."""
    yield {
        "source": "oasst",
        "kind": "chat",
        "user": ex.get("text", "") if ex.get("role") == "prompter" else "",
        "preferred": ex.get("text", "") if ex.get("role") == "assistant" else "",
        "rejected": "",
        "category": ex.get("role", "unknown"),
        "raw": ex,
    }


def _wildchat_records(ex: dict) -> Iterable[dict]:
    """WildChat: real user-ChatGPT conversations + toxicity flags + jailbreak
    flags. Emit one row per (turn) with the toxicity/jailbreak metadata
    preserved so the SFT consumer can filter by it."""
    conv = ex.get("conversation") or []
    toxic = ex.get("toxic", False)
    redacted = ex.get("redacted", False)
    for i, turn in enumerate(conv):
        if turn.get("role") == "user":
            yield {
                "source": "wildchat",
                "kind": "chat",
                "user": turn.get("content", ""),
                "preferred": (conv[i + 1].get("content", "")
                              if i + 1 < len(conv) and conv[i + 1].get("role") == "assistant"
                              else ""),
                "rejected": "",
                "category": (
                    "jailbreak" if toxic else
                    "redacted" if redacted else
                    "normal"
                ),
                "raw": {
                    "conversation_id": ex.get("conversation_id"),
                    "turn_index": i,
                    "toxic": toxic,
                    "redacted": redacted,
                },
            }


def _advbench_records(ex: dict) -> Iterable[dict]:
    """AdvBench: just {goal, target} pairs of harmful behaviors + their
    'should refuse'-style targets. We mark these as harmful_request — the
    SFT consumer can use them to teach refusal."""
    yield {
        "source": "advbench",
        "kind": "harmful_request",
        "user": ex.get("goal", "") or ex.get("prompt", "") or "",
        "preferred": "",   # leave empty — we generate refusals via Qwen later
        "rejected": ex.get("target", "") or "",
        "category": "explicit_harmful",
        "raw": ex,
    }


def _harmbench_records(ex: dict) -> Iterable[dict]:
    """HarmBench: red-team behaviors with category tags."""
    yield {
        "source": "harmbench",
        "kind": "harmful_request",
        "user": ex.get("Behavior", "") or ex.get("behavior", "") or "",
        "preferred": "",
        "rejected": "",
        "category": ex.get("FunctionalCategory", "") or ex.get("category", "") or "harmful",
        "raw": ex,
    }


def _jbb_records(ex: dict) -> Iterable[dict]:
    """JailbreakBench Behaviors: standardised harmful + benign pairs."""
    yield {
        "source": "jailbreakbench",
        "kind": "harmful_request",
        "user": ex.get("Goal", "") or ex.get("goal", "") or ex.get("prompt", "") or "",
        "preferred": "",
        "rejected": ex.get("Target", "") or ex.get("target", "") or "",
        "category": ex.get("Category", "") or ex.get("category", "") or "harmful",
        "raw": ex,
    }


# ---------------------------------------------------------------------------
# Source registry. License values reflect what HF reports in 2026-04 — verify
# yourself before trusting blindly.
# ---------------------------------------------------------------------------

SOURCES = {
    "hh_rlhf": {
        "source": "hh_rlhf",
        "hf_dataset": "Anthropic/hh-rlhf",
        "config": None,
        "split": "train",
        "fn": _hh_rlhf_records,
        "max_records": None,
        "streaming": True,
        "license_documented": "MIT",
        "license_class": "commercial",
        "notes": "Anthropic HH-RLHF preference pairs. Gold standard for "
                 "helpful + harmless training.",
    },
    "oasst1": {
        "source": "oasst1",
        "hf_dataset": "OpenAssistant/oasst1",
        "config": None,
        "split": "train",
        "fn": _oasst_records,
        "max_records": None,
        "streaming": True,
        "license_documented": "Apache-2.0",
        "license_class": "commercial",
        "notes": "OpenAssistant Conversations Dataset 1. High-quality multi-"
                 "turn chat with crowd-sourced quality ratings.",
    },
    "oasst2": {
        "source": "oasst2",
        "hf_dataset": "OpenAssistant/oasst2",
        "config": None,
        "split": "train",
        "fn": _oasst_records,
        "max_records": None,
        "streaming": True,
        "license_documented": "Apache-2.0",
        "license_class": "commercial",
        "notes": "Extended OASST dataset (oasst1 + new conversations).",
    },
    "wildchat": {
        "source": "wildchat",
        "hf_dataset": "allenai/WildChat-1M",
        "config": None,
        "split": "train",
        "fn": _wildchat_records,
        "max_records": 200_000,
        "streaming": True,
        "license_documented": "ODC-BY-1.0",
        "license_class": "commercial",
        "notes": "Real user/ChatGPT conversations from a public chat playground."
                 " Includes labeled toxicity + jailbreak attempts. Capped to "
                 "200k turns.",
    },
    "advbench": {
        "source": "advbench",
        "hf_dataset": "walledai/AdvBench",
        "config": None,
        "split": "train",
        "fn": _advbench_records,
        "max_records": None,
        "streaming": False,
        "license_documented": "MIT",
        "license_class": "commercial",
        "notes": "520 explicit harmful prompts from the GCG paper. Use as "
                 "refusal training signal.",
    },
    "harmbench": {
        "source": "harmbench",
        "hf_dataset": "walledai/HarmBench",
        "config": "standard",
        "split": "train",
        "fn": _harmbench_records,
        "max_records": None,
        "streaming": False,
        "license_documented": "MIT",
        "license_class": "commercial",
        "notes": "510 red-team behaviors with functional + semantic categories.",
    },
    "jailbreakbench": {
        "source": "jailbreakbench",
        "hf_dataset": "JailbreakBench/JBB-Behaviors",
        "config": "behaviors",
        "split": "harmful",
        "fn": _jbb_records,
        "max_records": None,
        "streaming": False,
        "license_documented": "MIT",
        "license_class": "commercial",
        "notes": "100 standardized harmful behaviors across 10 categories.",
    },
}


def _open_hf(dataset: str, config, split: str, streaming: bool):
    from datasets import load_dataset
    return load_dataset(dataset, config, split=split, streaming=streaming)


def _download_one(spec: dict, out_dir: Path, license_class: str) -> DownloadManifest:
    if spec.get("license_class") not in {"commercial", "permissive", "research"}:
        raise ValueError(f"spec missing valid license_class: {spec['source']!r}")

    # Hard gate: don't proceed if user-requested class doesn't include this source.
    rank = {"commercial": 3, "permissive": 2, "research": 1}
    if rank[spec["license_class"]] < rank[license_class]:
        print(f"  SKIP [{spec['source']}]: documented as {spec['license_class']!r}, "
              f"caller requires {license_class!r}", flush=True)
        return DownloadManifest(
            source=spec["source"], hf_dataset=spec["hf_dataset"],
            hf_config=spec.get("config"), split=spec["split"],
            license_documented=spec["license_documented"],
            license_class=spec["license_class"],
            output_file="(skipped)",
            started_at=now_iso(),
            notes=[f"SKIPPED: license-class mismatch (caller wants {license_class!r})"],
        )

    out_file = out_dir / f"{spec['source']}.jsonl"
    manifest = DownloadManifest(
        source=spec["source"], hf_dataset=spec["hf_dataset"],
        hf_config=spec.get("config"), split=spec["split"],
        license_documented=spec["license_documented"],
        license_class=spec["license_class"],
        output_file=str(out_file),
        started_at=now_iso(),
        notes=[spec.get("notes", "")],
    )
    t0 = time.time()
    max_records = spec.get("max_records")

    print(f"\n=== [{spec['source']}] -> {out_file} ===", flush=True)
    print(f"  hf:      {spec['hf_dataset']}  config={spec.get('config')}  split={spec['split']}", flush=True)
    print(f"  license: {spec['license_documented']}  (class={spec['license_class']})", flush=True)
    print(f"  max:     {max_records or 'all'}   streaming: {spec.get('streaming', True)}", flush=True)

    try:
        ds = _open_hf(spec["hf_dataset"], spec.get("config"), spec["split"],
                      streaming=spec.get("streaming", True))
    except Exception as e:                     # noqa: BLE001
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
                    line = json.dumps(rec, ensure_ascii=False) + "\n"
                    fh.write(line)
                    manifest.records_written += 1
                    manifest.bytes_written += len(line.encode("utf-8"))
                    if max_records and manifest.records_written >= max_records:
                        break
                if max_records and manifest.records_written >= max_records:
                    break
        except Exception as e:                 # noqa: BLE001
            manifest.notes.append(f"ITER FAILED: {type(e).__name__}: {e}")
            print(f"  ! iter failed mid-way: {e}", file=sys.stderr)

    manifest.finished_at = now_iso()
    manifest.elapsed_seconds = round(time.time() - t0, 1)
    manifest_path = out_file.with_suffix(out_file.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(asdict(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  done: {manifest.records_written} rec, "
          f"{manifest.bytes_written / 1e6:.1f} MB, "
          f"{manifest.elapsed_seconds:.0f}s", flush=True)
    return manifest


def cmd_list(args) -> None:
    rank = {"commercial": 3, "permissive": 2, "research": 1}
    print(f"\nRegistry — {len(SOURCES)} sources, current class filter: {args.license_class!r}\n")
    for key, spec in SOURCES.items():
        ok = rank[spec["license_class"]] >= rank[args.license_class]
        mark = "OK" if ok else "skip"
        print(f"  [{mark:4}] {key:20s} {spec['hf_dataset']:35s} "
              f"{spec['license_documented']:15s} (class={spec['license_class']})")
        print(f"           {spec.get('notes', '')[:90]}")


def cmd_download(args) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.source == "all":
        keys = list(SOURCES)
    else:
        if args.source not in SOURCES:
            sys.exit(f"unknown source {args.source!r}; "
                     f"known: {list(SOURCES)}")
        keys = [args.source]

    summary = []
    for key in keys:
        try:
            mf = _download_one(SOURCES[key], args.output_dir, args.license_class)
            summary.append(mf)
        except KeyboardInterrupt:
            print("\ninterrupted", file=sys.stderr)
            sys.exit(130)

    print("\n=== SUMMARY ===")
    for mf in summary:
        print(f"  {mf.source:20s} -> {mf.records_written} rec "
              f"({mf.bytes_written / 1e6:.1f} MB)  license={mf.license_documented}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--license-class",
                   choices=("commercial", "permissive", "research"),
                   default="commercial",
                   help="commercial = MIT/Apache/CC0/CC-BY/ODC-BY only (default).")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)

    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("list", help="Show the registry without downloading.")
    p_dl = sub.add_parser("download", help="Download one or all sources.")
    p_dl.add_argument("--source", required=True,
                      help=f"One of: all, {', '.join(SOURCES)}")

    # Back-compat: also accept --source / --list at top level for ergonomics.
    p.add_argument("--source", default=None, help="Same as 'download --source ...'.")
    p.add_argument("--list", action="store_true", help="Same as 'list' subcommand.")

    args = p.parse_args()

    if args.list or args.cmd == "list":
        cmd_list(args)
        return
    if args.cmd == "download":
        cmd_download(args)
        return
    if args.source:
        cmd_download(args)
        return
    p.print_help()


if __name__ == "__main__":
    main()
