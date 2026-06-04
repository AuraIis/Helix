"""License-aware Kaggle dataset downloader.

Two-stage license filter:
  1. Coarse: kaggle CLI's --license flag (cc/gpl/odb/other).
  2. Fine:   per-dataset metadata.json check against an allow-list.

Why both stages: the CLI filter is broad ("cc" matches CC0, CC-BY,
CC-BY-SA, CC-BY-NC, CC-BY-ND alike), and we explicitly need to weed out
the NoDerivatives + (depending on license-class) NonCommercial variants.

Output:
    raw/sft/kaggle/<dataset_slug>/<files>
    raw/sft/kaggle/<dataset_slug>/_audit.json

Each _audit.json captures: dataset slug, license literal as Kaggle reports
it, accepted-or-rejected decision, downloaded files + sizes, timestamp.
This is the audit trail you want when months later you need to prove the
training set is license-compliant.

Usage (from inside the container):
    # Search & list candidates without downloading:
    python scripts/data/download_kaggle_seeds.py search \\
        --query "german news" --license-class permissive --limit 10

    # Download a specific dataset (license check still applied):
    python scripts/data/download_kaggle_seeds.py fetch \\
        --slug owner/dataset-name --license-class permissive

    # Bulk: search + download all matching:
    python scripts/data/download_kaggle_seeds.py bulk \\
        --query "customer support" --license-class permissive --limit 5
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.data._common import now_iso  # noqa: E402

DEFAULT_OUT_DIR = Path("raw/sft/kaggle")


# ---------------------------------------------------------------------------
# License classes — Kaggle returns these literal strings in metadata.json.
# Map each user-facing class to the set of acceptable Kaggle license names.
# (Kaggle uses some quirky spellings; case-insensitive match in code.)
# ---------------------------------------------------------------------------

# Anything that names "noncommercial" or "noderivatives" is auto-rejected
# regardless of class.
ALWAYS_FORBIDDEN_KEYWORDS = (
    "noderivatives",
    "no-derivatives",
    "nd 4",
    "nd-4",
)

LICENSE_CLASSES = {
    # A: research-only — CC-BY-NC OK (still no ND though).
    "research": {
        "cc0",
        "cc0-1.0",
        "creative-commons-zero",
        "public-domain",
        "u.s. government works",
        "cc-by",
        "cc-by-4.0",
        "creative-commons-attribution-4-0",
        "cc-by-3.0",
        "cc-by-2.0",
        "cc-by-sa",
        "cc-by-sa-4.0",
        "creative-commons-attribution-share-alike-4-0",
        "cc-by-nc",
        "cc-by-nc-4.0",
        "creative-commons-attribution-noncommercial-4-0",
        "cc-by-nc-sa",
        "cc-by-nc-sa-4.0",
        "odbl-1.0",
        "open-database",
        "database-contents-license",
        "dbcl",
        "odc-by-1.0",
        "open-data-commons-by",
        "mit",
        "apache-2.0",
        "apache-license-2-0",
        "bsd",
        "gpl-2.0",
        "gpl-3.0",
    },
    # B: open-source-friendly — drop NC (would block any release).
    "permissive": {
        "cc0",
        "cc0-1.0",
        "creative-commons-zero",
        "public-domain",
        "u.s. government works",
        "cc-by",
        "cc-by-4.0",
        "creative-commons-attribution-4-0",
        "cc-by-3.0",
        "cc-by-2.0",
        "cc-by-sa",
        "cc-by-sa-4.0",
        "creative-commons-attribution-share-alike-4-0",
        "odbl-1.0",
        "open-database",
        "database-contents-license",
        "dbcl",
        "odc-by-1.0",
        "open-data-commons-by",
        "mit",
        "apache-2.0",
        "apache-license-2-0",
        "bsd",
    },
    # C: most-conservative, future-commercial-safe — only PD, CC0, attribution.
    "commercial": {
        "cc0",
        "cc0-1.0",
        "creative-commons-zero",
        "public-domain",
        "u.s. government works",
        "cc-by",
        "cc-by-4.0",
        "creative-commons-attribution-4-0",
        "cc-by-3.0",
        "cc-by-2.0",
        "odc-by-1.0",
        "open-data-commons-by",
        "mit",
        "apache-2.0",
        "apache-license-2-0",
        "bsd",
    },
}


@dataclass
class AuditEntry:
    slug: str
    license_literal: str
    license_class: str
    decision: str                # "accepted" | "rejected"
    reason: str = ""
    files: list = field(default_factory=list)
    bytes_total: int = 0
    fetched_at: str = ""


# ---------------------------------------------------------------------------
# License normalisation + decision logic
# ---------------------------------------------------------------------------


def _normalise_license(raw: str) -> str:
    """Lowercase, replace spaces and underscores with hyphens — gives a
    stable key suitable for set lookup against LICENSE_CLASSES entries."""
    if not raw:
        return ""
    s = raw.strip().lower()
    s = s.replace(" ", "-").replace("_", "-")
    s = s.replace("creativecommons", "creative-commons")
    return s


def is_acceptable(license_literal: str, class_name: str) -> tuple[bool, str]:
    if class_name not in LICENSE_CLASSES:
        return False, f"unknown class {class_name!r}"
    norm = _normalise_license(license_literal)
    if not norm:
        return False, "license missing"

    # Hard denylists per class — checked BEFORE allow-list so we never match
    # "cc-by" inside "cc-by-nc-4.0" by accident.
    raw_lower = license_literal.lower()
    has_nc = ("-nc" in norm) or ("noncommercial" in raw_lower) or ("non-commercial" in raw_lower)
    has_sa = ("-sa" in norm) or ("share-alike" in raw_lower) or ("sharealike" in raw_lower)
    has_nd = any(k in norm for k in ALWAYS_FORBIDDEN_KEYWORDS) or ("noderivatives" in raw_lower) or ("no-derivatives" in raw_lower)

    if has_nd:
        return False, "NoDerivatives forbidden (training is a derivative)"
    if class_name == "commercial" and has_nc:
        return False, "NonCommercial blocked under 'commercial' class"
    if class_name == "commercial" and has_sa:
        return False, "ShareAlike blocked under 'commercial' class"
    if class_name == "permissive" and has_nc:
        return False, "NonCommercial blocked under 'permissive' class"
    # 'permissive' allows SA (you stay open-source). 'research' allows NC + SA.

    if norm in LICENSE_CLASSES[class_name]:
        return True, "exact match"
    # Token-aware fallback: split on '-' and require all tokens of an allow
    # entry to be present in norm OR vice versa, BUT only when neither
    # contains forbidden tokens. Conservative: only accept if shortest is
    # a real prefix-or-equal of the longer when split on '-'.
    norm_tokens = set(norm.split("-"))
    for allow in LICENSE_CLASSES[class_name]:
        allow_tokens = set(allow.split("-"))
        if allow_tokens.issubset(norm_tokens) or norm_tokens.issubset(allow_tokens):
            # Defensive re-check (already filtered above, but explicit)
            if has_nc or has_nd:
                continue
            if class_name == "commercial" and has_sa:
                continue
            return True, f"token match: {allow}"
    return False, f"license {norm!r} not in {class_name} allow-list"


# ---------------------------------------------------------------------------
# Kaggle CLI wrappers
# ---------------------------------------------------------------------------


def _run(cmd: list, capture: bool = True) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def search_datasets(query: str, license_coarse: str = "cc",
                    max_size: int | None = None,
                    limit: int = 20) -> list:
    """Use kaggle CLI to search; returns list of dicts with at least
    {ref, title, size, lastUpdated}."""
    cmd = [
        "kaggle", "datasets", "list",
        "--search", query,
        "--license", license_coarse,
        "--csv",
        "-p", "1",
    ]
    if max_size is not None:
        cmd += ["--max-size", str(max_size)]
    rc, out, err = _run(cmd)
    if rc != 0:
        raise RuntimeError(f"kaggle search failed: {err}")
    # Parse CSV manually so we don't depend on csv module quirks for kaggle's
    # output (which uses commas inside title strings).
    import csv
    from io import StringIO
    rows = list(csv.DictReader(StringIO(out)))
    return rows[:limit]


def get_metadata(slug: str) -> dict:
    """Fetch dataset metadata, including the precise license name."""
    # `kaggle datasets metadata` writes a JSON file to a directory.
    tmp = Path("/tmp/_kaggle_meta_" + slug.replace("/", "_"))
    tmp.mkdir(parents=True, exist_ok=True)
    rc, out, err = _run(["kaggle", "datasets", "metadata", "-p", str(tmp), slug])
    if rc != 0:
        raise RuntimeError(f"metadata fetch failed for {slug}: {err}")
    meta_path = tmp / "dataset-metadata.json"
    if not meta_path.exists():
        raise RuntimeError(f"no metadata.json produced for {slug}")
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    return data


def download_dataset(slug: str, out_dir: Path, unzip: bool = True) -> list:
    """Download and (optionally) unzip a dataset. Returns list of files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-p", str(out_dir), slug]
    if unzip:
        cmd.append("--unzip")
    rc, out, err = _run(cmd)
    if rc != 0:
        raise RuntimeError(f"download failed for {slug}: {err}")
    return sorted(p for p in out_dir.iterdir() if p.is_file())


# ---------------------------------------------------------------------------
# High-level commands
# ---------------------------------------------------------------------------


def cmd_search(args) -> None:
    rows = search_datasets(args.query, license_coarse=args.kaggle_license,
                           max_size=args.max_size_mb * 1024 * 1024 if args.max_size_mb else None,
                           limit=args.limit)
    print(f"\nfound {len(rows)} candidate(s) for query={args.query!r} "
          f"under coarse={args.kaggle_license}, fine-class={args.license_class}\n")
    accepted = []
    for row in rows:
        slug = row.get("ref", "?")
        try:
            meta = get_metadata(slug)
            lic = meta.get("info", {}).get("licenses", [{}])[0].get("name", "")
        except Exception as e:                   # noqa: BLE001
            lic = ""
            print(f"  ! {slug}  (metadata error: {e})")
            continue
        ok, reason = is_acceptable(lic, args.license_class)
        mark = "OK" if ok else "REJ"
        print(f"  [{mark}] {slug}")
        print(f"       license: {lic!r}  ({reason})")
        print(f"       title:   {row.get('title', '')}")
        print(f"       size:    {row.get('size', '')}    updated: {row.get('lastUpdated', '')[:19]}")
        if ok:
            accepted.append(slug)
    print(f"\n{len(accepted)} dataset(s) pass {args.license_class!r} filter:")
    for s in accepted:
        print(f"  {s}")


def cmd_fetch(args) -> None:
    out_root = args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)
    slug = args.slug

    print(f"checking license for {slug} ...")
    meta = get_metadata(slug)
    lic = meta.get("info", {}).get("licenses", [{}])[0].get("name", "")
    ok, reason = is_acceptable(lic, args.license_class)

    audit = AuditEntry(
        slug=slug,
        license_literal=lic,
        license_class=args.license_class,
        decision="accepted" if ok else "rejected",
        reason=reason,
        fetched_at=now_iso(),
    )

    if not ok:
        print(f"  REJECTED: {reason!r}  (license={lic!r})")
        # Write rejection audit too — useful evidence later.
        rej_path = out_root / f"{slug.replace('/', '__')}_REJECTED.json"
        rej_path.write_text(json.dumps(asdict(audit), indent=2, ensure_ascii=False),
                            encoding="utf-8")
        return

    print(f"  ACCEPTED: license={lic!r}")
    dest = out_root / slug.replace("/", "__")
    dest.mkdir(exist_ok=True)
    files = download_dataset(slug, dest, unzip=not args.no_unzip)
    audit.files = [f.name for f in files]
    audit.bytes_total = sum(f.stat().st_size for f in files)
    (dest / "_audit.json").write_text(json.dumps(asdict(audit), indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    print(f"  downloaded {len(files)} file(s), "
          f"{audit.bytes_total / 1e6:.1f} MB to {dest}")


def cmd_bulk(args) -> None:
    rows = search_datasets(args.query, license_coarse=args.kaggle_license,
                           max_size=args.max_size_mb * 1024 * 1024 if args.max_size_mb else None,
                           limit=args.limit)
    out_root = args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    accepted_count = 0
    rejected_count = 0
    for row in rows:
        slug = row.get("ref", "?")
        sub_args = argparse.Namespace(
            slug=slug,
            license_class=args.license_class,
            output_dir=out_root,
            no_unzip=args.no_unzip,
        )
        try:
            cmd_fetch(sub_args)
            # Look up audit decision after fetch
            audit_path = out_root / slug.replace("/", "__") / "_audit.json"
            if audit_path.exists():
                accepted_count += 1
            else:
                rejected_count += 1
        except Exception as e:                  # noqa: BLE001
            print(f"  ! {slug} fetch error: {e}")
            rejected_count += 1
    print(f"\nbulk done: {accepted_count} fetched, {rejected_count} skipped/failed")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--license-class",
                        choices=list(LICENSE_CLASSES),
                        default="commercial",
                        help="A=research, B=permissive, C=commercial-safe (default).")
    common.add_argument("--kaggle-license", default="cc",
                        choices=("cc", "gpl", "odb", "other", "all"),
                        help="Coarse pre-filter passed to kaggle CLI.")
    common.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    common.add_argument("--max-size-mb", type=int, default=None,
                        help="Skip datasets larger than this many MB.")
    common.add_argument("--no-unzip", action="store_true",
                        help="Do not auto-unzip downloaded archives.")

    p_search = sub.add_parser("search", parents=[common],
                              help="List datasets matching query (no download).")
    p_search.add_argument("--query", required=True)
    p_search.add_argument("--limit", type=int, default=20)
    p_search.set_defaults(func=cmd_search)

    p_fetch = sub.add_parser("fetch", parents=[common],
                             help="Download one dataset by slug (license still checked).")
    p_fetch.add_argument("--slug", required=True,
                         help="owner/dataset-name (as shown on Kaggle URL).")
    p_fetch.set_defaults(func=cmd_fetch)

    p_bulk = sub.add_parser("bulk", parents=[common],
                            help="Search + download all matching datasets.")
    p_bulk.add_argument("--query", required=True)
    p_bulk.add_argument("--limit", type=int, default=10)
    p_bulk.set_defaults(func=cmd_bulk)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
