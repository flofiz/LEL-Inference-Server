"""Measure concurrent HTML uploads; stop at the first failed or too-slow wave."""

import argparse
import asyncio
import json
import math
import mimetypes
from io import BytesIO
from itertools import cycle
from pathlib import Path
from time import perf_counter

import httpx

LOCAL = "http://127.0.0.1:8000"
PUBLIC = "https://lel.cotedor.fr"
QWEN_PAGE_PIXELS = 2000 * 32 * 32


def prepare_image(path, resize):
    source = path.read_bytes()
    info = {"name": path.name, "original_bytes": len(source)}
    if resize:
        from PIL import Image

        with Image.open(BytesIO(source)) as image:
            width, height = image.size
            # A landscape crop includes ~3% overlap; keep each half near the Qwen budget.
            limit = QWEN_PAGE_PIXELS * (2 / 1.03 if width > height else 1)
            scale = min(1, math.sqrt(limit / (width * height)))
            size = max(1, int(width * scale)), max(1, int(height * scale))
            if scale < 1:
                payload = BytesIO()
                image.convert("RGB").resize(size, Image.Resampling.LANCZOS).save(
                    payload, "JPEG", quality=90, optimize=True,
                )
                info.update(original_size=[width, height], upload_size=list(size))
                result = payload.getvalue()
                info["bytes_per_upload"] = len(result)
                return (result, path.stem + ".jpg", "image/jpeg"), info
    info["bytes_per_upload"] = len(source)
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    return (source, path.name, mime), info


async def measure(client, image, filename, mime, max_ttft, max_duration):
    started = perf_counter()
    first = None
    body_sent = None
    pages = None
    finished = {}

    async def trace(event, info):
        nonlocal body_sent
        if event.endswith(".send_request_body.complete"):
            body_sent = perf_counter() - started

    try:
        async with asyncio.timeout(max_duration):
            async with asyncio.timeout(max_ttft) as first_token_deadline:
                async with client.stream(
                    "POST", "/transcribe/html/stream",
                    files={"image": (filename, image, mime)},
                    extensions={"trace": trace},
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
                                "body_sent_seconds": body_sent,
                                "post_upload_seconds": first - body_sent if body_sent is not None else None,
                                "fragments": len(pages), "truncated": sum(r == "length" for r in finished.values()),
                            }
    except TimeoutError as exc:
        phase = "first delta" if first is None else "complete response"
        detail = f"; request body sent at {body_sent:.2f}s" if body_sent is not None else "; upload not completed"
        raise RuntimeError(f"deadline exceeded waiting for {phase}{detail}") from exc
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
        "body_sent_p95_seconds": percentile([row["body_sent_seconds"] for row in rows if row.get("body_sent_seconds") is not None], 0.95),
        "post_upload_p95_seconds": percentile([row["post_upload_seconds"] for row in rows if row.get("post_upload_seconds") is not None], 0.95),
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
    images, descriptions = zip(*(prepare_image(path, args.resize) for path in args.image))
    address = PUBLIC if args.public else LOCAL
    report = {
        "base_url": address, "images": descriptions, "resized_for_upload": args.resize,
        "http2_enabled": args.public and not args.http1,
        "max_ttft_seconds": args.max_ttft, "max_duration_seconds": args.max_duration, "waves": [],
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    async with httpx.AsyncClient(
        base_url=address, http2=args.public and not args.http1, timeout=None,
        limits=httpx.Limits(max_connections=max(args.levels), max_keepalive_connections=max(args.levels)),
    ) as client:
        for index, size in enumerate(args.levels):
            if not await healthy(client):
                report["stopped"] = "health check failed before wave"
                break
            print(f"Starting {size} concurrent uploads", flush=True)
            workload = cycle(images)

            async def request():
                return await measure(client, *next(workload), args.max_ttft, args.max_duration)

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
    parser.add_argument("--image", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 4, 8, 16, 32, 48, 64, 96, 128])
    parser.add_argument("--max-ttft", type=float, default=60.0)
    parser.add_argument("--max-duration", type=float, default=600.0)
    parser.add_argument("--cooldown", type=float, default=10.0)
    parser.add_argument("--public", action="store_true", help="Use lel.cotedor.fr; only during an agreed test window")
    parser.add_argument("--http1", action="store_true", help="Disable HTTP/2 negotiation on the public route")
    parser.add_argument("--resize", action="store_true", help="Resize uploads to the Qwen image budget (requires Pillow)")
    args = parser.parse_args()
    if args.levels != sorted(set(args.levels)) or not all(1 <= n <= 128 for n in args.levels):
        parser.error("levels must be increasing, distinct and between 1 and 128")
    if not all(math.isfinite(t) for t in (args.max_ttft, args.max_duration, args.cooldown)) or not (
        0 < args.max_ttft <= args.max_duration and args.cooldown >= 0
    ):
        parser.error("invalid timeout or cooldown")
    asyncio.run(run(args))
