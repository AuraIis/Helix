"""Process raw stack-exchange-preferences into a clean troubleshoot SFT seed.

Input:  raw/sft/troubleshoot/stackex_preferences.jsonl
Output: seeds/sft/coding_troubleshoot/clean.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.data._common import atomic_text_writer, now_iso

FENCE = chr(96) * 3
TICK = chr(96)
NL = chr(10)


@dataclass
class Stats:
    started_at: str
    finished_at: str = ''
    in_records: int = 0
    out_records: int = 0
    dropped: dict = field(default_factory=dict)
    domain_kept: dict = field(default_factory=dict)


def _html_to_text(html):
    if not html:
        return ''
    soup = BeautifulSoup(html, 'html.parser')
    for pre in soup.find_all('pre'):
        code_text = pre.get_text()
        wrapped = NL + NL + FENCE + NL + code_text.rstrip() + NL + FENCE + NL + NL
        pre.replace_with(soup.new_string(wrapped))
    for code in soup.find_all('code'):
        code_text = code.get_text()
        code.replace_with(soup.new_string(TICK + code_text + TICK))
    text = soup.get_text(separator=' ', strip=False)
    lines = [ln.rstrip() for ln in text.split(NL)]
    out = []
    blank = 0
    for ln in lines:
        if ln.strip() == '':
            blank += 1
            if blank <= 1:
                out.append('')
        else:
            blank = 0
            out.append(ln)
    return NL.join(out).strip()


def _domain(url):
    if not url:
        return 'unknown'
    netloc = urlparse(url).netloc.replace('meta.', '')
    if 'stackoverflow.com' in netloc:
        return 'stackoverflow'
    if '.stackexchange.com' in netloc:
        return netloc.replace('.stackexchange.com', '')
    return netloc


def _has_code(problem, answer):
    return FENCE in problem or FENCE in answer


def _drop(stats, reason):
    stats.dropped[reason] = stats.dropped.get(reason, 0) + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', type=Path,
                    default=Path('raw/sft/troubleshoot/stackex_preferences.jsonl'))
    ap.add_argument('--output-dir', type=Path,
                    default=Path('seeds/sft/coding_troubleshoot'))
    ap.add_argument('--min-score', type=int, default=5)
    ap.add_argument('--min-chars', type=int, default=80)
    ap.add_argument('--require-code', action='store_true', default=True)
    ap.add_argument('--no-require-code', dest='require_code', action='store_false')
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / 'clean.jsonl'
    manifest_path = args.output_dir / 'manifest.json'
    stats = Stats(started_at=now_iso())

    print('reading: ' + str(args.input), flush=True)
    print('writing: ' + str(out_path), flush=True)

    with args.input.open('r', encoding='utf-8') as fin, atomic_text_writer(out_path) as fout:
        for line_no, line in enumerate(fin):
            if args.limit and line_no >= args.limit:
                break
            stats.in_records += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                _drop(stats, 'bad_json')
                continue

            score = int(rec.get('answer_accepted_score', 0) or 0)
            if score < args.min_score:
                _drop(stats, 'low_score')
                continue

            problem = _html_to_text(rec.get('question', ''))
            accepted = _html_to_text(rec.get('answer_accepted', ''))
            rejected = _html_to_text(rec.get('answer_rejected', ''))

            if len(problem) < args.min_chars:
                _drop(stats, 'short_problem')
                continue
            if len(accepted) < args.min_chars:
                _drop(stats, 'short_accepted')
                continue

            has_code = _has_code(problem, accepted)
            if args.require_code and not has_code:
                _drop(stats, 'no_code')
                continue

            url = rec.get('metadata_url', '')
            domain = _domain(url)
            stats.domain_kept[domain] = stats.domain_kept.get(domain, 0) + 1

            out = {
                'id': stats.out_records,
                'source': domain,
                'score': score,
                'url': url,
                'problem': problem,
                'accepted_answer': accepted,
                'rejected_answer': rejected,
                'rejected_score': int(rec.get('answer_rejected_score', 0) or 0),
                'has_code': has_code,
                'chars': len(problem) + len(accepted),
            }
            fout.write(json.dumps(out, ensure_ascii=False) + NL)
            stats.out_records += 1

            if stats.in_records % 10000 == 0:
                print('  processed ' + str(stats.in_records) + ', kept ' + str(stats.out_records), flush=True)

    stats.finished_at = now_iso()
    manifest_path.write_text(
        json.dumps(asdict(stats), indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    print('done: ' + str(stats.in_records) + ' in -> ' + str(stats.out_records) + ' out')
    print('manifest: ' + str(manifest_path))


if __name__ == '__main__':
    main()
