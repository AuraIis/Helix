#!/usr/bin/env python3
"""Consolidate Phase-3 SFT data into final corpus.

Sources:
- batch1.jsonl (980, all task_types)
- batch2.jsonl (5598 post-halluc-filter, EXCLUDE honest_refusal — superseded)
- halfilter_v2.jsonl (310 high-quality refusals with verschärftem System-Prompt)

Output:
- phase3_sft_final.jsonl (consolidated, deduped, ready for SFT trainer)
- phase3_sft_final.stats.json (audit metadata)

Usage:
    python scripts/data/synth/consolidate_phase3_final.py
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path('/workspace/v2data/raw/sft/synth/outputs')
SOURCES = [
    ('batch1', ROOT / 'phase3_batch1.jsonl', None),  # all task_types
    ('batch2', ROOT / 'phase3_batch2.jsonl', {'exclude_task_types': ['honest_refusal']}),
    ('halfilter_v2', ROOT / 'halfilter_v2.jsonl', None),  # only honest_refusal, all kept
]
OUT_PATH = ROOT / 'phase3_sft_final.jsonl'
STATS_PATH = ROOT / 'phase3_sft_final.stats.json'

records_by_id: dict[str, dict] = {}
source_counts: dict[str, int] = {}
excluded: dict[str, int] = {}

for label, path, opts in SOURCES:
    if not path.exists():
        print(f'  WARN: {path} does not exist, skipping')
        continue
    n_kept = n_excl = n_err = 0
    excl_types = set((opts or {}).get('exclude_task_types', []))
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            if d.get('error'):
                n_err += 1
                continue
            if d.get('task_type') in excl_types:
                n_excl += 1
                continue
            rec_id = d['id']
            # Strip non-essential fields for cleaner SFT format
            clean = {
                'id': d['id'],
                'task_type': d['task_type'],
                'source_batch': label,
                'model': d['model'],
                'messages': d['messages'],
            }
            if d.get('reasoning'):
                clean['reasoning'] = d['reasoning']
            records_by_id[rec_id] = clean
            n_kept += 1
    source_counts[label] = n_kept
    if n_excl > 0:
        excluded[label] = n_excl
    print(f'  {label}: kept={n_kept}  excluded={n_excl}  errors_skipped={n_err}')

print()
print(f'Total unique IDs: {len(records_by_id)}')

records = list(records_by_id.values())
by_task = Counter(r['task_type'] for r in records)
by_model = Counter(r['model'] for r in records)
by_source = Counter(r['source_batch'] for r in records)

print()
print('Task-Distribution:')
for tt, c in sorted(by_task.items(), key=lambda kv: -kv[1]):
    print(f'  {tt:25s} {c:>5d}')
print()
print('Model-Distribution:')
for m, c in by_model.items():
    print(f'  {m:35s} {c:>5d}')
print()
print('Source-Distribution:')
for s, c in by_source.items():
    print(f'  {s:25s} {c:>5d}')

# Write consolidated output
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with OUT_PATH.open('w', encoding='utf-8') as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print()
print(f'Wrote: {OUT_PATH}  ({OUT_PATH.stat().st_size / 1024 / 1024:.1f} MB)')

# Write stats
stats = {
    'consolidated_at': datetime.now(timezone.utc).isoformat(),
    'total_records': len(records),
    'sources': source_counts,
    'excluded': excluded,
    'task_distribution': dict(by_task),
    'model_distribution': {m: c for m, c in by_model.items()},
}
STATS_PATH.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding='utf-8')
print(f'Stats: {STATS_PATH}')
