"""CPU tests for the benchmark bookkeeping (no vLLM or GPU required)."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def benchmark(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from scripts import benchmark_rswa

    return benchmark_rswa


def test_benchmark_measures_both_crops_at_requested_concurrency(benchmark):
    cache = benchmark.CachePeak()

    class Engine:
        running = peak = 0

        async def generate(self, payload, params, request_id):
            self.running += 1
            self.peak = max(self.peak, self.running)
            cache.record(SimpleNamespace(kv_cache_usage=0.1 * self.running), None)
            try:
                await asyncio.sleep(0)
                yield SimpleNamespace(prompt_token_ids=[1, 2], outputs=[
                    SimpleNamespace(token_ids=[3], finish_reason=None),
                ])
                await asyncio.sleep(0)
                yield SimpleNamespace(prompt_token_ids=None, outputs=[
                    SimpleNamespace(token_ids=[4, 5], finish_reason="length"),
                ])
            finally:
                self.running -= 1

    engine = Engine()
    params = SimpleNamespace(max_tokens=3)
    report = asyncio.run(benchmark.measure_case(
        engine, cache, "long", [{}, {}], params, True, 2, 1,
    ))
    assert [row["crop"] for row in report["requests"]] == [1, 2, 1, 2]
    assert all(row["generated_tokens"] == 3 for row in report["requests"])
    assert all(row["prompt_tokens"] == 2 for row in report["requests"])
    assert engine.peak == 2
    assert report["peak_observed_kv_cache_fraction"] == pytest.approx(0.2)
    assert report["median_decode_tokens_per_second"] > 0


def test_benchmark_fails_if_a_forced_generation_ends_early(benchmark):
    async def generate(*args, **kwargs):
        yield SimpleNamespace(prompt_token_ids=[1], outputs=[
            SimpleNamespace(token_ids=[2], finish_reason="stop"),
        ])

    with pytest.raises(RuntimeError, match="got 1, budget 16384"):
        asyncio.run(benchmark.measure_request(
            SimpleNamespace(generate=generate), {}, SimpleNamespace(max_tokens=16384), 1, True,
        ))
