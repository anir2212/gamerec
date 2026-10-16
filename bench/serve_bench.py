"""Bounded open-loop localhost request benchmark; record achieved rate and tails."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path

import httpx


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty latency series")
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


async def benchmark(base: str, rate: int, seconds: int) -> dict:
    durations: list[float] = []
    errors = 0
    semaphore = asyncio.Semaphore(100)
    async with httpx.AsyncClient(timeout=10, limits=httpx.Limits(max_connections=100)) as client:
        async def one(index: int, start: float):
            nonlocal errors
            await asyncio.sleep(max(0, start + index / rate - time.perf_counter()))
            async with semaphore:
                sent = time.perf_counter()
                try:
                    response = await client.get(base + "/recommend", params={
                        "user_id": f"u{index % 120:03d}", "k": 10, "method": "itemknn"})
                    if response.status_code != 200 or len(response.json()["items"]) != 10:
                        errors += 1
                except (httpx.HTTPError, KeyError, ValueError):
                    errors += 1
                durations.append((time.perf_counter() - sent) * 1000)
        count = rate * seconds
        started = time.perf_counter()
        await asyncio.gather(*(one(i, started) for i in range(count)))
        elapsed = time.perf_counter() - started
    return {"target_rps": rate, "duration_target_seconds": seconds,
            "requests": count, "errors": errors, "elapsed_seconds": elapsed,
            "achieved_rps": count / elapsed, "p50_ms": percentile(durations, 0.5),
            "p95_ms": percentile(durations, 0.95), "p99_ms": percentile(durations, 0.99)}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:18081")
    parser.add_argument("--seconds", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("runs/serve_bench.json"))
    args = parser.parse_args()
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(args.base + "/recommend", params={"user_id": "u000"})
        response.raise_for_status()  # Warm model loading before measuring requests.
    rows = [await benchmark(args.base, rate, args.seconds) for rate in (50, 200)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
