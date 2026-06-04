"""Categorize collected safety seeds into Hard-No / Soft / Helpful bins.

Reads the raw safety jsonl files produced by download_safety_seeds.py and
splits them into three streams matching docs/AURALIS_SAFETY_POLICY.md:

    seeds/sft/safety/safety_hard.jsonl
        Records matching one of the 5 Hard-No categories. The model must
        refuse these in BOTH Default and Owner modes.

    seeds/sft/safety/safety_softable.jsonl
        Records that look "potentially-restricted" but don't match Hard-No.
        These will be sent to the Qwen rewriter (qwen_synth_sft.py) to
        produce two assistant responses each: one Default-hedged, one
        Owner-direct. Trained as conditional behaviour on the OWNER_MODE
        system flag.

    seeds/sft/safety/safety_normal.jsonl
        Records that look like ordinary helpful queries (most of WildChat,
        most of OASST). Pass-through to the standard SFT stream.

Plus a manifest with per-source category counts.

L1 (lexical) detection only — L2 semantic confirmation comes later via a
separate Qwen-judge pass when a live endpoint is available. L1 is
intentionally conservative on Hard-No: false positives there cost us
helpful coverage; false negatives are caught by the L2 follow-up pass.

Usage (from inside the container):
    python scripts/data/categorize_safety_seeds.py
    python scripts/data/categorize_safety_seeds.py --in-dir custom/raw/safety
    python scripts/data/categorize_safety_seeds.py --rebuild  # purge old output first
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.data._common import atomic_text_writer, now_iso  # noqa: E402

DEFAULT_IN_DIR = Path("raw/sft/safety")
DEFAULT_OUT_DIR = Path("seeds/sft/safety")


# ---------------------------------------------------------------------------
# Hard-No detection patterns. Conservative — match must be obviously
# operational, not merely topical. Each pattern is keyed by the Hard-No ID
# from the policy doc.
#
# These are deliberately short. The semantic L2 step will add nuance later.
# ---------------------------------------------------------------------------

HN_PATTERNS = {
    "HN-1_csam": [
        # Sexual + minor combinations. Catches "child"/"kid"/"minor"/"teen"
        # near sexual terms.
        r"\b(child|children|kid|kids|minor|minors|teen|teens|preteen|underage|loli|shota)\b[^.\n]{0,80}\b(sex|sexual|porn|nude|nudes|naked|fuck|rape|grooming|cp)\b",
        r"\b(sex|sexual|porn|nude|nudes|naked|fuck|rape|grooming|cp)\b[^.\n]{0,80}\b(child|children|kid|kids|minor|minors|teen|teens|preteen|underage|loli|shota)\b",
        r"\bcsam\b",
        r"\bchild\s+pornography\b",
    ],
    "HN-2_wmd_synthesis": [
        # Operational synthesis verbs near WMD nouns.
        r"\b(synthesi[sz]e|cook|make|produce|manufacture|brew|create|build|construct|prepare)\b[^.\n]{0,120}\b(sarin|vx|tabun|soman|nerve\s+agent|mustard\s+gas|chlorine\s+gas|phosgene|ricin|anthrax|smallpox|botulinum|nuclear\s+(bomb|weapon|device)|dirty\s+bomb|atomic\s+bomb|enriched\s+uranium|plutonium\s+core)\b",
        r"\b(step[- ]by[- ]step|recipe|procedure|exact\s+amounts?|precise\s+(quantities|measurements|temperatures))\b[^.\n]{0,120}\b(sarin|vx|tabun|nerve\s+agent|ricin|anthrax|botulinum|nuclear\s+device|fissile)\b",
        r"\b(weaponi[sz]e|aerosoli[sz]e)\b[^.\n]{0,80}\b(virus|pathogen|bacteria|spore)\b",
    ],
    "HN-3_concrete_attack_planning": [
        # Specific named target + violent verb. Vague hate is NOT this category.
        r"\b(kill|murder|assassinate|shoot|stab|bomb|attack|poison|harm|hurt)\b[^.\n]{0,40}\b(my\s+(neighbor|teacher|boss|coworker|wife|husband|partner|ex|family)|my\s+(\w+\s+){0,2}(teacher|boss|coworker))\b",
        r"\b(plan|planning|how\s+to|guide|instructions?)\b[^.\n]{0,60}\b(murder|assassinat|mass\s+shooting|school\s+shooting|stalking)\b[^.\n]{0,80}\b(specific|named|real|particular)\b",
        # Address-based targeting
        r"\b(kill|attack|harm|hurt|stalk)\b[^.\n]{0,80}\b\d{1,4}\s+\w+\s+(street|st\.|avenue|ave\.|road|rd\.|lane|ln\.|drive|dr\.)\b",
    ],
    "HN-4_doxxing": [
        # "Find me the home address of <real person>" pattern.
        r"\b(find|get|give\s+me|tell\s+me|look\s+up|locate)\b[^.\n]{0,60}\b(home\s+address|personal\s+phone|private\s+phone|home\s+phone|ssn|social\s+security|credit\s+card|password)\b[^.\n]{0,80}\b(of|for|belonging\s+to)\b",
        r"\bdox(x|xed|xing)\b",
        r"\b(home\s+address|residential\s+address|where\s+(does|do)\s+\w+\s+live)\b[^.\n]{0,80}\b(@\w+|twitter|youtuber|streamer|celebrity|actor|actress|politician|ceo)\b",
    ],
    "HN-5_deployment_ready_malware": [
        # Asks for a complete working exploit / payload / phishing kit.
        r"\b(write|generate|give\s+me|provide|create)\b[^.\n]{0,40}\b(working|complete|full|ready[- ]to[- ](deploy|use|run)|production[- ]ready|deployable)\b[^.\n]{0,60}\b(ransomware|trojan|rat|keylogger|stealer|backdoor|rootkit|botnet|c2|command\s+and\s+control|exploit\s+kit|phishing\s+(kit|page))\b",
        r"\b(build|create|set\s+up|deploy)\b[^.\n]{0,40}\b(phishing\s+(page|site|kit))\b[^.\n]{0,80}\b(target|impersonat|spoof)\b",
        r"\bzero[- ]?day\b[^.\n]{0,80}\b(working|functional|deployable|exploit|payload)\b",
    ],
}

# Compile once.
_COMPILED = {hn: [re.compile(p, re.IGNORECASE) for p in pats]
             for hn, pats in HN_PATTERNS.items()}


def detect_hard_no(text: str) -> str | None:
    """Return the Hard-No ID if any pattern matches, else None."""
    if not text:
        return None
    for hn_id, patterns in _COMPILED.items():
        for p in patterns:
            if p.search(text):
                return hn_id
    return None


# ---------------------------------------------------------------------------
# Soft-No heuristic: looks "sensitive" enough that we should send it
# through the rewrite step (so we get two responses — Default + Owner).
# Categories: drugs, hacking/sec, weapons (non-WMD), self-harm, controversial.
# ---------------------------------------------------------------------------

SOFT_HINTS = re.compile(
    r"\b("
    r"drug|drugs|cocaine|heroin|meth|mdma|lsd|psilocybin|fentanyl|"
    r"hack|hacking|exploit|vulnerability|bypass|crack|pirate|piracy|"
    r"weapon|gun|rifle|pistol|firearm|ammunition|"
    r"self[- ]harm|suicide|overdose|"
    r"abortion|euthanasia|assisted\s+suicide|"
    r"holocaust|nazi|stalin|gulag|"
    r"jailbreak|prompt\s+injection|ignore\s+(previous|prior)\s+(instructions|rules)|DAN\s+mode"
    r")\b",
    re.IGNORECASE,
)


@dataclass
class CategoryStats:
    started_at: str
    finished_at: str = ""
    in_records: int = 0
    hard_no: int = 0
    softable: int = 0
    normal: int = 0
    bad_records: int = 0
    per_hard_no: dict = field(default_factory=dict)
    per_source_in: dict = field(default_factory=dict)
    per_source_hard: dict = field(default_factory=dict)
    per_source_softable: dict = field(default_factory=dict)
    per_source_normal: dict = field(default_factory=dict)


def _bump(d: dict, key: str) -> None:
    d[key] = d.get(key, 0) + 1


def _process_one(in_path: Path, hard_fh, softable_fh, normal_fh,
                 stats: CategoryStats) -> None:
    print(f"\n--- {in_path.name} ---", flush=True)
    src_name = in_path.stem
    n_in = 0
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                stats.bad_records += 1
                continue
            n_in += 1
            stats.in_records += 1
            _bump(stats.per_source_in, src_name)

            user = rec.get("user", "") or ""
            # Some sources put text in 'preferred' or 'rejected' instead.
            extra = " ".join([rec.get("preferred", "") or "",
                              rec.get("rejected", "") or ""])

            hn = detect_hard_no(user) or detect_hard_no(extra)
            if hn:
                rec["_category"] = hn
                rec["_bin"] = "hard_no"
                hard_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats.hard_no += 1
                _bump(stats.per_hard_no, hn)
                _bump(stats.per_source_hard, src_name)
                continue

            if SOFT_HINTS.search(user) or SOFT_HINTS.search(extra):
                rec["_bin"] = "softable"
                softable_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats.softable += 1
                _bump(stats.per_source_softable, src_name)
                continue

            rec["_bin"] = "normal"
            normal_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats.normal += 1
            _bump(stats.per_source_normal, src_name)

    print(f"  {src_name}: {n_in} in", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--rebuild", action="store_true",
                   help="Delete existing output bins before running.")
    args = p.parse_args()

    if args.rebuild and args.out_dir.exists():
        for f in args.out_dir.glob("safety_*.jsonl*"):
            f.unlink()
        print(f"  rebuild: cleared previous output in {args.out_dir}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    inputs = sorted(args.in_dir.glob("*.jsonl"))
    if not inputs:
        sys.exit(f"no inputs found in {args.in_dir}")

    print(f"input dir : {args.in_dir}")
    print(f"output dir: {args.out_dir}")
    print(f"sources   : {len(inputs)}")
    for p in inputs:
        print(f"  - {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")

    stats = CategoryStats(started_at=now_iso())
    hard_path = args.out_dir / "safety_hard.jsonl"
    soft_path = args.out_dir / "safety_softable.jsonl"
    norm_path = args.out_dir / "safety_normal.jsonl"

    t0 = time.time()
    with atomic_text_writer(hard_path) as hh, \
         atomic_text_writer(soft_path) as sh, \
         atomic_text_writer(norm_path) as nh:
        for in_path in inputs:
            _process_one(in_path, hh, sh, nh, stats)

    stats.finished_at = now_iso()
    elapsed = time.time() - t0
    manifest_path = args.out_dir / "categorize_manifest.json"
    manifest_path.write_text(
        json.dumps(asdict(stats), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n=== SUMMARY ===")
    print(f"elapsed:  {elapsed:.1f}s")
    print(f"in:       {stats.in_records}")
    print(f"hard-no:  {stats.hard_no}  (per-rule: {stats.per_hard_no})")
    print(f"softable: {stats.softable}")
    print(f"normal:   {stats.normal}")
    print(f"bad:      {stats.bad_records}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
