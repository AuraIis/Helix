"""DeepSeek V4 (Pro/Flash) async generation client for Auralis Phase-3 SFT data.

Routes Pro vs Flash based on the task_type per record. Pro for code-engineering
patterns where idiomatic concision matters (try/finally, edge-cases). Flash
for everything else — explainer / tutorial / step-by-step style fits the
Auralis target persona (preference-confirmed via A/B test 2026-04-28).

Endpoint: OpenRouter (single key, automatic provider fallback).

Input format (JSONL — one record per line):
    {"id": "task_001",
     "task_type": "code_implementation",
     "system_prompt": "Du bist ...",
     "user_prompt":   "Schreib mir eine Funktion die ..."}

    Optional fields per record:
        "model_override": "pro" | "flash"
        "temperature":    0.3
        "max_tokens":     4096

Output: JSONL appended, resume-safe (re-running with same --output skips done IDs).

Usage:
    export OPENROUTER_API_KEY=sk-or-v1-...
    python scripts/data/synth/deepseek_v4_client.py \\
        --input  raw/sft/synth/inputs/code_explain.jsonl \\
        --output raw/sft/synth/outputs/code_explain.jsonl \\
        --workers 16
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"

# Routing per task_type — based on A/B test 2026-04-28 (Pro="senior-engineer
# concise", Flash="tutor-style explicit"). User preference: Flash dominant,
# Pro only for code-generation idiomatic patterns.
TASK_ROUTING: dict[str, str] = {
    # Code engineering: Pro for idiomatic generation
    "code_implementation":  "pro",
    "code_refactoring":     "pro",
    "code_debug_fix":       "pro",
    "code_review_fix":      "pro",
    # Code teaching: Flash tutorial-style
    "code_explain":         "flash",
    "code_walkthrough":     "flash",
    "code_review_comment":  "flash",
    # Reasoning: Flash for explicit step-by-step
    "math_word_problem":    "flash",
    "logic_puzzle":         "flash",
    "step_by_step_reason":  "flash",
    # Knowledge: Flash for didactic depth
    "concept_explain":      "flash",
    "factual_qa":           "flash",
    "cultural_qa":          "flash",
    # Writing: mixed
    "translation":          "flash",
    "honest_refusal":       "flash",
    "rewrite_text":         "pro",
    "creative_writing":     "pro",
}

MODEL_IDS: dict[str, str] = {
    "flash": "deepseek/deepseek-v4-flash",
    "pro":   "deepseek/deepseek-v4-pro",
}

# Per-million-token prices (USD) — cheapest provider on OpenRouter 2026-04-28.
# Used only when OpenRouter response doesn't include `cost` directly.
DEFAULT_PRICE_PER_M: dict[str, dict[str, float]] = {
    "flash": {"in": 0.14, "out": 0.28},
    "pro":   {"in": 1.39, "out": 2.78},
}

DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS  = 4096
DEFAULT_TIMEOUT_S   = 180.0
DEFAULT_RETRIES     = 3


@dataclass
class TaskInput:
    id: str
    task_type: str
    system_prompt: str
    user_prompt: str
    model_override: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


@dataclass
class GenerationResult:
    id: str
    task_type: str
    model: str
    messages: list[dict[str, str]]
    reasoning: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0
    generated_at: str = ""
    error: str | None = None


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_RETRIES,
    ):
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout_s),
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=16),
        )
        return self

    async def __aexit__(self, *exc):
        if self._client is not None:
            await self._client.aclose()

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://<host>.local/auralis-v2",
            "X-Title":       "Auralis v2 SFT Generation",
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                assert self._client is not None
                resp = await self._client.post(OPENROUTER_BASE, json=payload, headers=headers)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code == 429:  # rate limit
                    await asyncio.sleep(5.0 * (attempt + 1))
                    last_error = RuntimeError(f"429 rate limit: {resp.text[:200]}")
                    continue
                if resp.status_code in (502, 503, 504):  # transient
                    await asyncio.sleep(2 ** attempt)
                    last_error = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                    continue
                # 4xx other → not retryable
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                last_error = e
                await asyncio.sleep(2 ** attempt)
        raise RuntimeError(f"Max retries exhausted: {last_error}")


def estimate_cost(model_key: str, in_tokens: int, out_tokens: int) -> float:
    p = DEFAULT_PRICE_PER_M[model_key]
    return (in_tokens * p["in"] + out_tokens * p["out"]) / 1_000_000


def _utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


async def generate_one(
    client: OpenRouterClient,
    task: TaskInput,
    semaphore: asyncio.Semaphore,
) -> GenerationResult:
    async with semaphore:
        model_key = task.model_override or TASK_ROUTING.get(task.task_type, "flash")
        if model_key not in MODEL_IDS:
            return GenerationResult(
                id=task.id, task_type=task.task_type, model=model_key,
                messages=[], generated_at=_utc_iso(),
                error=f"Unknown model_key {model_key!r}",
            )
        model_id = MODEL_IDS[model_key]
        messages = [
            {"role": "system", "content": task.system_prompt},
            {"role": "user",   "content": task.user_prompt},
        ]
        t0 = time.time()
        try:
            resp = await client.chat(
                model=model_id, messages=messages,
                temperature=task.temperature if task.temperature is not None else DEFAULT_TEMPERATURE,
                max_tokens=task.max_tokens if task.max_tokens is not None else DEFAULT_MAX_TOKENS,
            )
        except Exception as e:
            return GenerationResult(
                id=task.id, task_type=task.task_type, model=model_id,
                messages=messages, latency_s=time.time() - t0,
                generated_at=_utc_iso(), error=str(e),
            )
        try:
            choice = resp["choices"][0]["message"]
        except (KeyError, IndexError) as e:
            return GenerationResult(
                id=task.id, task_type=task.task_type, model=model_id,
                messages=messages, latency_s=time.time() - t0,
                generated_at=_utc_iso(),
                error=f"unexpected response shape: {e!r}: {str(resp)[:300]}",
            )

        content   = choice.get("content") or ""
        # Pro often emits internal CoT under `reasoning`/`reasoning_content`
        reasoning = choice.get("reasoning") or choice.get("reasoning_content")

        usage_raw = resp.get("usage") or {}
        in_t  = usage_raw.get("prompt_tokens", 0) or 0
        out_t = usage_raw.get("completion_tokens", 0) or 0
        cost  = usage_raw.get("cost")
        if cost is None:
            cost = estimate_cost(model_key, in_t, out_t)

        return GenerationResult(
            id=task.id,
            task_type=task.task_type,
            model=model_id,
            messages=messages + [{"role": "assistant", "content": content}],
            reasoning=reasoning,
            usage={"in_tokens": in_t, "out_tokens": out_t, "cost_usd": cost},
            latency_s=round(time.time() - t0, 3),
            generated_at=_utc_iso(),
        )


def load_tasks(input_path: Path) -> list[TaskInput]:
    tasks: list[TaskInput] = []
    with input_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  WARN skipping line {line_no}: {e}", file=sys.stderr)
                continue
            tasks.append(TaskInput(
                id=d["id"],
                task_type=d["task_type"],
                system_prompt=d["system_prompt"],
                user_prompt=d["user_prompt"],
                model_override=d.get("model_override"),
                temperature=d.get("temperature"),
                max_tokens=d.get("max_tokens"),
            ))
    return tasks


def load_existing_ids(output_path: Path) -> set[str]:
    """Return IDs of successfully-generated records (skip on resume)."""
    done: set[str] = set()
    if not output_path.exists():
        return done
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if d.get("error") is None and d.get("id"):
                    done.add(d["id"])
            except json.JSONDecodeError:
                continue
    return done


async def run_pipeline(
    input_path: Path,
    output_path: Path,
    workers: int,
    api_key: str,
    max_tasks: int | None = None,
    progress_every: int = 25,
) -> None:
    tasks = load_tasks(input_path)
    done = load_existing_ids(output_path)
    pending = [t for t in tasks if t.id not in done]
    if max_tasks is not None:
        pending = pending[:max_tasks]
    print(f"[load] total={len(tasks)} done={len(done)} pending={len(pending)}", flush=True)
    if not pending:
        print("[load] nothing to do", flush=True)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(workers)

    n_done = n_err = 0
    total_in = total_out = 0
    total_cost = 0.0
    by_model = {"flash": 0, "pro": 0}

    async with OpenRouterClient(api_key) as client:
        with output_path.open("a", encoding="utf-8") as out_f:
            futures = [asyncio.create_task(generate_one(client, t, sem)) for t in pending]
            for fut in asyncio.as_completed(futures):
                res = await fut
                out_f.write(json.dumps({
                    "id":           res.id,
                    "task_type":    res.task_type,
                    "model":        res.model,
                    "messages":     res.messages,
                    "reasoning":    res.reasoning,
                    "usage":        res.usage,
                    "latency_s":    res.latency_s,
                    "generated_at": res.generated_at,
                    "error":        res.error,
                }, ensure_ascii=False) + "\n")
                out_f.flush()

                if res.error:
                    n_err += 1
                else:
                    n_done += 1
                    total_in   += res.usage.get("in_tokens", 0)
                    total_out  += res.usage.get("out_tokens", 0)
                    total_cost += res.usage.get("cost_usd", 0.0) or 0.0
                    if "flash" in res.model:
                        by_model["flash"] += 1
                    elif "pro" in res.model:
                        by_model["pro"] += 1

                progress = n_done + n_err
                if progress % progress_every == 0:
                    pct = 100 * progress / len(pending)
                    print(
                        f"[gen] {progress}/{len(pending)} ({pct:.1f}%) err={n_err} "
                        f"flash={by_model['flash']} pro={by_model['pro']} "
                        f"cost=${total_cost:.3f} "
                        f"in={total_in/1e6:.2f}M out={total_out/1e6:.2f}M",
                        flush=True,
                    )

    print("\n=== DONE ===", flush=True)
    print(f"  generated:  {n_done}", flush=True)
    print(f"  errors:     {n_err}", flush=True)
    print(f"  flash:      {by_model['flash']}", flush=True)
    print(f"  pro:        {by_model['pro']}", flush=True)
    print(f"  cost (USD): ${total_cost:.4f}", flush=True)
    print(f"  in_tokens:  {total_in:,}", flush=True)
    print(f"  out_tokens: {total_out:,}", flush=True)
    print(f"  output:     {output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input",   type=Path, required=True,
                        help="JSONL with task records")
    parser.add_argument("--output",  type=Path, required=True,
                        help="JSONL for generated examples (appended; resume-safe)")
    parser.add_argument("--workers", type=int, default=16,
                        help="concurrent in-flight requests (default 16)")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="cap number of tasks (for testing)")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit(
            "ERROR: OPENROUTER_API_KEY env var not set. "
            "Get one at https://openrouter.ai/keys"
        )

    asyncio.run(run_pipeline(
        input_path=args.input,
        output_path=args.output,
        workers=args.workers,
        api_key=api_key,
        max_tasks=args.max_tasks,
        progress_every=args.progress_every,
    ))


if __name__ == "__main__":
    main()
