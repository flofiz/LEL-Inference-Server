"""Measure concurrent HTML uploads; stop at the first failed or too-slow wave."""

import argparse
import asyncio
import json
import math
import mimetypes
from pathlib import Path
from time import perf_counter

import httpx

LOCAL = "http://127.0.0.1:8000"
PUBLIC = "https://lel.cotedor.fr"


async def measure(client, image, filename, mime, max_ttft, max_duration):
    started = perf_counter()
    first = None
    pages = None
    finished = {}
    try:
        async with asyncio.timeout(max_duration):
            async with asyncio.timeout(max_ttft) as first_token_deadline:
                async with client.stream(
                    "POST", "/transcribe/html/stream",
                    files={"image": (filename, image, mime)},
                ) as response:
                    response.raise_for_status()
                    if not response.headers.get("content-type", "").startswith("application/x-ndjson"):
                        raise RuntimeError("unexpected response (maintenance or proxy error)")
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        event = json.loads(line)
                        kind = event["type"]
                        if kind == "metadata":
                            pages = {page["pageIndex"] for page in event["pages"]}
                        elif kind == "delta" and first is None and event.get("html"):
                            first = perf_counter() - started
                            first_token_deadline.reschedule(None)
                        elif kind == "error":
                            raise RuntimeError(f"generation failed on fragment {event.get('pageIndex')}")
                        elif kind == "done":
                            finished[event["pageIndex"]] = event["finishReason"]
                        elif kind == "complete":
                            if first is None or not pages or finished.keys() != pages:
                                raise RuntimeError("incomplete or empty transcription")
                            return {
                                "ttft_seconds": first, "duration_seconds": perf_counter() - started,
                                "fragments": len(pages), "truncated": sum(r == "length" for r in finished.values()),
                            }
    except TimeoutError as exc:
        phase = "first delta" if first is None else "complete response"
        raise RuntimeError(f"deadline exceeded waiting for {phase}") from exc
    raise RuntimeError("connection closed before complete event")


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


async def wave(run_request, size):
    started = perf_counter()
    tasks = [asyncio.create_task(run_request()) for _ in range(size)]
    try:
        _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for task in pending:
            task.cancel()
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    rows = [row for row in outcomes if isinstance(row, dict)]
    failures = [f"{type(exc).__name__}: {exc}" for exc in outcomes if isinstance(exc, Exception)]
    elapsed = perf_counter() - started
    return {
        "concurrent_images": size, "completed": len(rows),
        "cancelled": sum(isinstance(row, asyncio.CancelledError) for row in outcomes),
        "errors": failures, "wall_seconds": elapsed, "images_per_second": len(rows) / elapsed,
        "completed_fragments": sum(row["fragments"] for row in rows),
        "truncated_fragments": sum(row["truncated"] for row in rows),
        "ttft_p50_seconds": percentile([row["ttft_seconds"] for row in rows], 0.5),
        "ttft_p95_seconds": percentile([row["ttft_seconds"] for row in rows], 0.95),
        "duration_p95_seconds": percentile([row["duration_seconds"] for row in rows], 0.95),
    }


async def healthy(client):
    try:
        response = await client.get("/health", timeout=5)
        payload = response.json()
        return response.is_success and isinstance(payload, dict) and payload.get("status") == "ok"
    except (httpx.HTTPError, ValueError):
        return False


async def run(args):
    image = args.image.read_bytes()
    mime = mimetypes.guess_type(args.image)[0] or "application/octet-stream"
    address = PUBLIC if args.public else LOCAL
    report = {
        "base_url": address, "image": args.image.name, "bytes_per_upload": len(image),
        "max_ttft_seconds": args.max_ttft, "max_duration_seconds": args.max_duration, "waves": [],
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    async with httpx.AsyncClient(
        base_url=address, http2=args.public, timeout=None,
        limits=httpx.Limits(max_connections=max(args.levels), max_keepalive_connections=max(args.levels)),
    ) as client:
        async def request():
            return await measure(client, image, args.image.name, mime, args.max_ttft, args.max_duration)

        for index, size in enumerate(args.levels):
            if not await healthy(client):
                report["stopped"] = "health check failed before wave"
                break
            print(f"Starting {size} concurrent uploads", flush=True)
            result = await wave(request, size)
            result["healthy_after"] = await healthy(client)
            report["waves"].append(result)
            save()
            print(f"{result['completed']}/{size} complete; TTFT p95={result['ttft_p95_seconds']}s; "
                  f"errors={len(result['errors'])}", flush=True)
            if result["errors"] or result["completed"] != size or not result["healthy_after"]:
                report["stopped"] = "wave failed or exceeded the latency budget"
                break
            if index + 1 < len(args.levels):
                await asyncio.sleep(args.cooldown)
    save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 4, 8, 16, 32, 48, 64, 96, 128])
    parser.add_argument("--max-ttft", type=float, default=30.0)
    parser.add_argument("--max-duration", type=float, default=600.0)
    parser.add_argument("--cooldown", type=float, default=10.0)
    parser.add_argument("--public", action="store_true", help="Use lel.cotedor.fr; only during an agreed test window")
    args = parser.parse_args()
    if args.levels != sorted(set(args.levels)) or not all(1 <= n <= 128 for n in args.levels):
        parser.error("levels must be increasing, distinct and between 1 and 128")
    if not all(math.isfinite(t) for t in (args.max_ttft, args.max_duration, args.cooldown)) or not (
        0 < args.max_ttft <= args.max_duration and args.cooldown >= 0
    ):
        parser.error("invalid timeout or cooldown")
    asyncio.run(run(args))
