"""Regression tests for token logprobs delivered in vLLM delta batches."""

import asyncio
import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_transcription_metrics_are_independent_of_delta_batching(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "src"))
    from inference.engine import HTREngine

    htr = HTREngine.__new__(HTREngine)
    htr.logger = logging.getLogger(__name__)
    htr.engine = SimpleNamespace(get_tokenizer=lambda: SimpleNamespace(all_special_ids=[6]))

    tokens = [
        ("Con", -1.0),
        ("seil\t", -2.0),
        ("1\t2\t3\t4\t0", -3.0),
        ("\nAutre", -6.0),
        ("\t1\t2\t3\t4\t0\n", -4.0),
        ("<|im_end|>", -0.5),
    ]

    async def read_batches(batches):
        async def outputs():
            for ids in batches:
                yield SimpleNamespace(outputs=[SimpleNamespace(
                    text="".join(tokens[i - 1][0] for i in ids if i != 6),
                    token_ids=ids,
                    logprobs=[
                        {99: SimpleNamespace(logprob=-99.0), i: SimpleNamespace(
                            decoded_token=tokens[i - 1][0], logprob=tokens[i - 1][1],
                        )}
                        for i in ids
                    ],
                )])

        return [json.loads(chunk) async for chunk in htr.stream_results(outputs(), ((1, 1), 0))]

    single = asyncio.run(read_batches([[i] for i in range(1, 7)]))
    grouped = asyncio.run(read_batches([[1, 2], [3, 4, 5, 6], []]))
    assert grouped == single
    assert [line["label"] for line in grouped[:-1]] == ["Conseil", "Autre"]
    assert grouped[0]["line_perplexity"] == round(math.exp(7 / (3 + 1 / 6)), 3)
    assert grouped[0]["char_perplexity"] == round(math.exp(2.6 / 1.8), 3)
    metrics = grouped[-1]["metrics"]
    assert metrics["file_length"] == 6
    assert metrics["file_perplexity"] == -16.5
    assert metrics["char_ppl_count"] == pytest.approx(1.8 + 5 / 6)
    assert metrics["char_perplexity"] == pytest.approx(-2.6 - 5)
