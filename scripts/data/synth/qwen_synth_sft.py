"""Refactor cleaned stack-exchange records into structured Troubleshooting SFT.

Reads:  seeds/sft/coding_troubleshoot/clean.jsonl  (output of process_troubleshoot_seeds.py)
Writes: seeds/sft/coding_troubleshoot/sft.jsonl    (one chat-formatted SFT example per record)

For each clean record, we ask the teacher LLM to lift the raw question/answer
pair into our 6-step Troubleshooting schema:

    1. Problem        — what the user is reporting (verbatim, normalized)
    2. Clarification  — questions you'd ask back to disambiguate
    3. Diagnose chain — hypotheses ranked by likelihood, each with a check
    4. Solution       — the actual fix (explained step-by-step, with code)
    5. Verification   — how to confirm the fix worked
    6. Escalation     — when this approach fails, what's next

Output schema (per line):
    {
      "id": int,
      "source_id": int,                       # original clean.jsonl id
      "url": str,
      "messages": [                           # ready-to-train SFT chat
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": "..."}
      ],
      "meta": {
        "score": int,
        "domain": str,
        "teacher_model": str,
        "teacher_tokens": int,
        "rendered_at": iso8601
      }
    }

Resume support:
- The output JSONL is append-only. Re-running this skips IDs already present
  in the output file (read once at startup), so a long run that crashed
  half-way picks up where it left off.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from scripts.data.synth.qwen_client import GenRequest, QwenClient


SYSTEM_PROMPT = """You convert messy Stack Overflow Q/A into a structured \
troubleshooting answer. Stay grounded in the source material — do NOT invent \
fixes the original answer didn't suggest. Your job is to *restructure*, not to \
generate new technical claims.

Output exactly six sections, each marked with the heading on its own line:

# Problem
A one-paragraph summary of what the user is asking about. Include the relevant \
error messages and environment if mentioned. No fluff.

# Clarification
Two to three short questions you would ask the user back BEFORE diagnosing, to \
narrow down ambiguity. If the original question is already specific, write \
"None needed — the question already specifies X, Y, Z."

# Diagnose chain
A numbered list of 2–4 hypotheses, ordered by likelihood given the symptoms. \
For each hypothesis, give one concrete check the user can run to confirm or \
rule it out.

# Solution
The actual fix the accepted answer prescribes, rewritten step-by-step. Keep \
all code blocks intact (use markdown ```fenced``` style). If multiple steps \
are needed, number them. No editorializing.

# Verification
How the user knows the fix worked: the expected output, a test command, or a \
log line to look for.

# Escalation
What to try if this approach doesn't work — usually one or two backup paths \
(different library version, alternate API, file a bug report, etc.). If the \
original answer doesn't suggest any, write "If this doesn't work, the most \
likely cause is environment-specific; share `<diagnostic command output>` for \
follow-up."

Use plain prose. No marketing tone, no apologies. Code blocks stay verbatim."""


USER_TEMPLATE = """Restructure this Stack Overflow Q/A into the 6-section schema.

# Original question
{problem}

# Accepted answer
{accepted_answer}
"""


def _already_done(path: Path) -> set:
    """Return set of source_ids already present in an existing output file."""
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if "source_id" in rec:
                    done.add(rec["source_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _build_request(rec: dict, max_tokens: int) -> GenRequest:
    user = USER_TEMPLATE.format(
        problem=rec["problem"][:8000],            # cap input length
        accepted_answer=rec["accepted_answer"][:8000],
    )
    return GenRequest(
        request_id=f"sft-{rec['id']}",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user},
        ],
        max_tokens=max_tokens,
        temperature=0.3,                          # lower temp = more grounded
        top_p=0.9,
        extra={
            "source_id": rec["id"],
            "url": rec.get("url", ""),
            "score": rec.get("score", 0),
            "domain": rec.get("source", "unknown"),
        },
    )


def _format_output(res, teacher_model: str) -> dict:
    extra = res.extra or {}
    user_msg = (
        "I'm troubleshooting an issue. Please walk me through it step by step "
        "using the structured Problem / Clarification / Diagnose / Solution / "
        "Verification / Escalation format."
    )
    # The user-content wrapper is intentionally generic; the teacher's response
    # contains the original problem context inside the # Problem section, so
    # the SFT example trains the model to produce the structured answer
    # GIVEN any problem framing.
    return {
        "id": extra.get("source_id"),
        "source_id": extra.get("source_id"),
        "url": extra.get("url"),
        "messages": [
            {"role": "user",      "content": user_msg + "\n\n" + res.completion.split("# Problem", 1)[-1].split("# Clarification", 1)[0].strip()[:2000] if "# Problem" in res.completion else user_msg},
            {"role": "assistant", "content": res.completion},
        ],
        "meta": {
            "score": extra.get("score", 0),
            "domain": extra.get("domain", ""),
            "teacher_model": teacher_model,
            "teacher_tokens": res.total_tokens,
            "rendered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }


async def main_async(args) -> None:
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"reading: {in_path}", flush=True)
    print(f"writing: {out_path}", flush=True)

    done = _already_done(out_path)
    print(f"resume: {len(done)} records already in output, will skip", flush=True)

    # Load all clean records, filter out done + apply limit.
    requests = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["id"] in done:
                continue
            requests.append(_build_request(rec, max_tokens=args.max_tokens))
            if args.limit and len(requests) >= args.limit:
                break

    print(f"requests to send: {len(requests)}", flush=True)
    if not requests:
        print("nothing to do — exiting.")
        return

    client = QwenClient(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        max_concurrency=args.concurrency,
    )

    out_fh = out_path.open("a", encoding="utf-8")
    n_ok = n_err = 0
    t0 = time.monotonic()
    try:
        async for res in client.run_batch(requests):
            if res.error:
                n_err += 1
                print(f"  ERR {res.request_id}: {res.error[:120]}", flush=True)
                continue
            n_ok += 1
            out_rec = _format_output(res, teacher_model=args.model)
            out_fh.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            out_fh.flush()
            if n_ok % 50 == 0:
                rate = n_ok / max(0.001, time.monotonic() - t0)
                print(f"  ok={n_ok} err={n_err} rate={rate:.1f}/s", flush=True)
    finally:
        out_fh.close()

    print()
    print(f"done: ok={n_ok} err={n_err}", flush=True)
    print(f"client stats: {json.dumps(client.stats_summary(), indent=2)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input",  type=Path,
                    default=Path("seeds/sft/coding_troubleshoot/clean.jsonl"))
    ap.add_argument("--output", type=Path,
                    default=Path("seeds/sft/coding_troubleshoot/sft.jsonl"))
    ap.add_argument("--base-url",
                    default=os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1"))
    ap.add_argument("--api-key",
                    default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--model",
                    default=os.environ.get("OPENAI_MODEL", "Qwen2.5-32B-Instruct"))
    ap.add_argument("--concurrency", type=int, default=4,
                    help="Parallel in-flight requests against the endpoint.")
    ap.add_argument("--max-tokens",  type=int, default=2048)
    ap.add_argument("--limit",       type=int, default=None,
                    help="Stop after this many NEW requests (for testing).")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
