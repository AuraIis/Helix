"""OpenAI-compatible chat-completion client for synth-SFT generation.

Designed to talk to either:
  - a local Qwen served via vLLM (OPENAI_API_BASE=http://localhost:8000/v1)
  - DeepSeek's API (OPENAI_API_BASE=https://api.deepseek.com/v1)
  - any other OpenAI-API-compatible endpoint

Features:
  - async batch generation with a concurrency semaphore
  - exponential backoff on transient errors (429, 5xx, network)
  - per-call token + cost accounting
  - JSONL append-only output so a crashed run can resume
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import AsyncIterator, Optional

try:
    import httpx
except ImportError:
    print("install httpx first: pip install httpx", file=sys.stderr)
    raise


@dataclass
class GenRequest:
    request_id: str
    messages: list                              # [{'role','content'}, ...]
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95
    extra: dict = field(default_factory=dict)        # arbitrary metadata to round-trip
    extra_body: dict = field(default_factory=dict)   # extra request fields (e.g. reasoning_effort)


@dataclass
class GenResult:
    request_id: str
    completion: str
    finish_reason: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    error: Optional[str] = None
    elapsed_ms: int = 0
    extra: dict = field(default_factory=dict)


@dataclass
class CostStats:
    requests_attempted: int = 0
    requests_succeeded: int = 0
    requests_failed: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    elapsed_ms_total: int = 0


class QwenClient:
    def __init__(
        self,
        base_url: str,
        api_key: str = "EMPTY",
        model: str = "Qwen2.5-32B-Instruct",
        max_concurrency: int = 4,
        max_retries: int = 5,
        timeout_s: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self._sem = asyncio.Semaphore(max_concurrency)
        self.stats = CostStats()

    async def _post_one(self, client: httpx.AsyncClient, req: GenRequest) -> GenResult:
        url = self.base_url + "/chat/completions"
        body = {
            "model": self.model,
            "messages": req.messages,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "stream": False,
        }
        if req.extra_body:
            body.update(req.extra_body)
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key,
        }
        t0 = time.monotonic()
        last_error = ""
        for attempt in range(self.max_retries):
            try:
                resp = await client.post(url, json=body, headers=headers, timeout=self.timeout_s)
                if resp.status_code == 200:
                    data = resp.json()
                    choice = data["choices"][0]
                    usage = data.get("usage", {}) or {}
                    elapsed_ms = int((time.monotonic() - t0) * 1000)
                    return GenResult(
                        request_id=req.request_id,
                        completion=choice["message"].get("content", "") or "",
                        finish_reason=choice.get("finish_reason", "stop"),
                        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                        total_tokens=int(usage.get("total_tokens", 0) or 0),
                        elapsed_ms=elapsed_ms,
                        extra=req.extra,
                    )
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_error = f"http {resp.status_code}: {resp.text[:200]}"
                    backoff = min(2 ** attempt + random.random(), 60.0)
                    await asyncio.sleep(backoff)
                    continue
                last_error = f"http {resp.status_code}: {resp.text[:200]}"
                break
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                last_error = f"{type(e).__name__}: {str(e)[:200]}"
                backoff = min(2 ** attempt + random.random(), 60.0)
                await asyncio.sleep(backoff)
            except Exception as e:
                last_error = f"{type(e).__name__}: {str(e)[:200]}"
                break

        return GenResult(
            request_id=req.request_id,
            completion="",
            finish_reason="error",
            error=last_error,
            elapsed_ms=int((time.monotonic() - t0) * 1000),
            extra=req.extra,
        )

    async def _run_one(self, client: httpx.AsyncClient, req: GenRequest) -> GenResult:
        async with self._sem:
            self.stats.requests_attempted += 1
            res = await self._post_one(client, req)
            if res.error:
                self.stats.requests_failed += 1
            else:
                self.stats.requests_succeeded += 1
                self.stats.prompt_tokens += res.prompt_tokens
                self.stats.completion_tokens += res.completion_tokens
                self.stats.total_tokens += res.total_tokens
            self.stats.elapsed_ms_total += res.elapsed_ms
            return res

    async def run_batch(
        self,
        requests: list,
        output_jsonl: Optional[Path] = None,
    ) -> AsyncIterator[GenResult]:
        """Run all requests concurrently. If output_jsonl is set, each result
        is appended as it completes (one JSON per line) — robust against
        interrupts: a crashed run can be resumed by skipping IDs already in
        the file."""
        out_fh = None
        if output_jsonl is not None:
            output_jsonl = Path(output_jsonl)
            output_jsonl.parent.mkdir(parents=True, exist_ok=True)
            out_fh = output_jsonl.open("a", encoding="utf-8")

        try:
            async with httpx.AsyncClient() as client:
                tasks = [asyncio.create_task(self._run_one(client, r)) for r in requests]
                for fut in asyncio.as_completed(tasks):
                    res = await fut
                    if out_fh:
                        out_fh.write(json.dumps(asdict(res), ensure_ascii=False) + "\n")
                        out_fh.flush()
                    yield res
        finally:
            if out_fh:
                out_fh.close()

    def stats_summary(self) -> dict:
        return asdict(self.stats)


# ---------------------------------------------------------------------------
# Smoke-test CLI: send N hello-world requests, print stats.
# ---------------------------------------------------------------------------
def _smoke_main() -> None:
    ap = argparse.ArgumentParser(description="Smoke-test the QwenClient against a live endpoint.")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "Qwen2.5-32B-Instruct"))
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=2)
    args = ap.parse_args()

    client = QwenClient(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        max_concurrency=args.concurrency,
    )

    reqs = [
        GenRequest(
            request_id=f"smoke-{i}",
            messages=[{"role": "user", "content": "Say hello in one short sentence."}],
            max_tokens=64,
            temperature=0.7,
            extra={"idx": i},
        )
        for i in range(args.n)
    ]

    async def go():
        async for res in client.run_batch(reqs):
            if res.error:
                print(res.request_id, "ERROR:", res.error)
            else:
                print(res.request_id, "->", res.completion[:80].replace("\n", " "))
        print()
        print("stats:", json.dumps(client.stats_summary(), indent=2))

    asyncio.run(go())


if __name__ == "__main__":
    _smoke_main()
