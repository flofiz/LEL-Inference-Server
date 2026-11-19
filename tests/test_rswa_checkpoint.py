"""Opt-in test with the real multimodal checkpoint and no running Ray server."""

import asyncio
import math
import os
import sys
from pathlib import Path

import pytest


def test_rswa_checkpoint_long_multimodal_decode(monkeypatch, request):
    model = os.environ.get("HTR_RSWA_TEST_MODEL")
    if not model or sys.platform != "linux":
        pytest.skip("Set HTR_RSWA_TEST_MODEL on the Linux GPU host")
    image_bytes = request.getfixturevalue("image_bytes")
    output_path = os.environ.get("HTR_RSWA_TEST_OUTPUT")

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "src"))
    from inference.engine import HTREngine, PROMPT, SYSTEM

    async def run():
        htr = HTREngine(
            name=Path(model).name,
            model=model,
            require_rswa=True,
            max_model_len=8192,
            max_num_seqs=1,
            max_num_batched_tokens=2048,
            limit_mm_per_prompt={"image": 1},
        )
        try:
            from processing.preprocessing import get_inputs
            from vllm.sampling_params import RequestOutputKind, SamplingParams

            evidence = await htr.verify_ready()
            count = evidence["window"] + 3 * evidence["block_size"]
            inputs, _, _ = next(get_inputs(
                image_bytes, htr.engine.get_tokenizer(), PROMPT, SYSTEM, qwen_version="qwen35"
            ))
            params = SamplingParams(
                temperature=0.0,
                min_tokens=count,
                max_tokens=count,
                ignore_eos=True,
                logprobs=1,
                output_kind=RequestOutputKind.DELTA,
            )
            generated = 0
            text_chunks = []
            async for result in htr.engine.generate(inputs, params, request_id="rswa-smoke"):
                output = result.outputs[0]
                if output_path:
                    text_chunks.append(output.text)
                if not output.token_ids:
                    continue
                assert len(output.token_ids) == len(output.logprobs)
                for token, logprobs in zip(output.token_ids, output.logprobs):
                    assert math.isfinite(logprobs[token].logprob), f"Nonfinite logprob at {generated}"
                    generated += 1
            assert generated == count
            if output_path:
                Path(output_path).write_text("".join(text_chunks), encoding="utf-8")
        finally:
            htr.shutdown()

    asyncio.run(run())
