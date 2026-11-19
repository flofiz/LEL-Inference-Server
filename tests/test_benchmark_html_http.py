import asyncio
import json
from pathlib import Path

import httpx
import pytest


@pytest.fixture
def benchmark(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import benchmark_html_http

    return benchmark_html_http


def test_http_benchmark_counts_crops_not_uploads(benchmark):
    events = [
        {"type": "metadata", "pages": [{"pageIndex": 0}, {"pageIndex": 1}]},
        {"type": "delta", "pageIndex": 1, "html": "<p>Texte</p>"},
        {"type": "done", "pageIndex": 1, "finishReason": "stop"},
        {"type": "done", "pageIndex": 0, "finishReason": "length"},
        {"type": "complete"},
    ]

    def endpoint(request):
        assert request.url.path == "/transcribe/html/stream"
        return httpx.Response(200, headers={"Content-Type": "application/x-ndjson"},
                              text="\n".join(json.dumps(event) for event in events))

    async def run():
        async with httpx.AsyncClient(base_url=benchmark.LOCAL, transport=httpx.MockTransport(endpoint)) as client:
            return await benchmark.wave(lambda: benchmark.measure(client, b"image", "page.jpg", "image/jpeg", 5, 10), 3)

    result = asyncio.run(run())
    assert result["completed"] == 3
    assert result["completed_fragments"] == 6 and result["truncated_fragments"] == 3
    assert result["ttft_p95_seconds"] >= result["ttft_p50_seconds"]


def test_http_benchmark_cancels_remaining_requests_on_first_failure(benchmark):
    called = 0

    async def request():
        nonlocal called
        called += 1
        if called == 1:
            raise RuntimeError("deadline exceeded")
        await asyncio.Future()

    result = asyncio.run(asyncio.wait_for(benchmark.wave(request, 4), timeout=1))
    assert result["cancelled"] == 3 and result["completed"] == 0
    assert result["errors"] == ["RuntimeError: deadline exceeded"]
