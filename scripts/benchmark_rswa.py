"""Benchmark the RSWA engine directly on the CUDA host, with Ray Serve stopped."""

import argparse
import asyncio
from importlib.metadata import distribution, version
import json
import logging
from pathlib import Path
from statistics import median
import sys
from time import perf_counter
from uuid import uuid4


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("model", "normal-image", "long-image", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--concurrency", type=positive_int, nargs="+", default=[1, 4])
    parser.add_argument("--repeats", type=positive_int, default=1)
    parser.add_argument("--normal-max-tokens", type=positive_int, default=4096)
    parser.add_argument("--long-lengths", type=positive_int, nargs="+", default=[8192, 16384])
    parser.add_argument("--max-model-len", type=positive_int, default=24576)
    parser.add_argument("--max-num-batched-tokens", type=positive_int, default=4096)
    args = parser.parse_args()
    if max(args.normal_max_tokens, *args.long_lengths) >= args.max_model_len:
        parser.error("--max-model-len must leave space for the prompt in addition to the output")
    return args


class CachePeak:
    """vLLM stat-logger factory; report peak observed cache occupancy per run."""

    def __init__(self):
        self.reset()

    def __call__(self, vllm_config, engine_index=0):
        return self

    def reset(self):
        self.peak = None

    def record(self, scheduler_stats, iteration_stats, mm_cache_stats=None, engine_idx=0):
        if scheduler_stats is not None:
            usage = scheduler_stats.kv_cache_usage
            self.peak = usage if self.peak is None else max(self.peak, usage)

    def log_engine_initialized(self):
        pass

    def log(self):
        pass


async def measure_request(engine, inputs, params, crop, force):
    started = perf_counter()
    first = last = None
    prompt_tokens = first_tokens = generated = 0
    finish_reason = None
    async for result in engine.generate(inputs, params, request_id=uuid4().hex):
        output, = result.outputs
        if result.prompt_token_ids is not None:
            prompt_tokens = len(result.prompt_token_ids)
        tokens = len(output.token_ids)
        if tokens:
            last = perf_counter()
            if first is None:
                first, first_tokens = last, tokens
            generated += tokens
        finish_reason = output.finish_reason
    finished = perf_counter()
    if not generated or (force and generated != params.max_tokens):
        raise RuntimeError(
            f"Unexpected output length for crop {crop}: got {generated}, "
            f"budget {params.max_tokens}, prompt {prompt_tokens}, finish={finish_reason}"
        )
    decode_seconds = last - first
    return {
        "crop": crop,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated,
        "finish_reason": finish_reason,
        "ttft_seconds": first - started,
        "e2e_seconds": finished - started,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": (generated - first_tokens) / decode_seconds if decode_seconds else None,
    }


async def measure_case(engine, cache, name, inputs, params, force, concurrency, repeat):
    slots = asyncio.Semaphore(concurrency)

    async def measure(crop, payload):
        async with slots:
            return await measure_request(engine, payload, params, crop, force)

    cache.reset()
    started = perf_counter()
    # Exercise every crop at every concurrency; portrait -> one crop, double page -> two.
    requests = await asyncio.gather(*(
        measure(crop, payload)
        for _ in range(concurrency)
        for crop, payload in enumerate(inputs, start=1)
    ))
    elapsed = perf_counter() - started
    decode_rates = [r["decode_tokens_per_second"] for r in requests if r["decode_tokens_per_second"] is not None]
    return {
        "case": name,
        "max_output_tokens_per_crop": params.max_tokens,
        "forced_length": force,
        "concurrency": concurrency,
        "repeat": repeat,
        "wall_seconds": elapsed,
        "median_ttft_seconds": median(r["ttft_seconds"] for r in requests),
        "median_decode_tokens_per_second": median(decode_rates) if decode_rates else None,
        "aggregate_tokens_per_second": sum(r["generated_tokens"] for r in requests) / elapsed,
        "peak_observed_kv_cache_fraction": cache.peak,
        "requests": requests,
    }


async def run(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from inference.engine import HTREngine, PROMPT, SYSTEM, latin1_regex
    from inference.rswa_patch import EDITS, digest

    def save(report):
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    cache = CachePeak()
    htr = HTREngine(
        name=args.model.name,
        model=str(args.model.resolve()),
        require_rswa=True,
        enable_thinking=False,
        stat_loggers=[cache],
        max_model_len=args.max_model_len,
        max_num_seqs=max(args.concurrency),
        max_num_batched_tokens=args.max_num_batched_tokens,
        limit_mm_per_prompt={"image": 1},
    )
    try:
        import torch
        from processing.preprocessing import get_inputs
        from vllm.sampling_params import RequestOutputKind, SamplingParams, StructuredOutputsParams

        evidence = await htr.verify_ready()
        engine = htr.engine
        tokenizer = engine.get_tokenizer()

        def load_image(path):
            return [inputs for inputs, _, _ in get_inputs(
                path.read_bytes(), tokenizer, PROMPT, SYSTEM, qwen_version="qwen35",
                enable_thinking=False,
            )]

        def sampling(tokens, force=False):
            return SamplingParams(
                temperature=0.0,
                max_tokens=tokens,
                min_tokens=tokens if force else 0,
                ignore_eos=force,
                logprobs=1,
                output_kind=RequestOutputKind.DELTA,
                structured_outputs=StructuredOutputsParams(regex=latin1_regex),
            )

        normal = load_image(args.normal_image)
        long = load_image(args.long_image)
        installed = distribution("vllm")
        report = {
            "label": args.label,
            "model": str(args.model.resolve()),
            "images": {"normal": str(args.normal_image), "long": str(args.long_image)},
            "hardware": torch.cuda.get_device_name(),
            "versions": {name: version(name) for name in ("vllm", "torch", "transformers")},
            "source_sha256": {
                path: digest(Path(installed.locate_file(path)).read_bytes()) for path in EDITS
            },
            "engine": {
                "max_model_len": engine.vllm_config.model_config.max_model_len,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "max_num_seqs": max(args.concurrency),
                "thinking": False,
                "mtp_tokens": 1,
                "rswa": evidence,
            },
            "runs": [],
        }
        save(report)
        print(f"Inputs: {len(normal)} normal crop(s), {len(long)} long crop(s)", flush=True)
        for concurrency in args.concurrency:
            # Warm up both image shapes and this level of batching before timing.
            await measure_case(engine, cache, "warmup", normal + long,
                               sampling(32, force=True), True, concurrency, 0)
            for repeat in range(1, args.repeats + 1):
                cases = [("normal", normal, args.normal_max_tokens, False)]
                cases += [("long", long, length, True) for length in args.long_lengths]
                for name, inputs, tokens, force in cases:
                    print(f"{name}: concurrency={concurrency}, output budget={tokens}, repeat={repeat}", flush=True)
                    result = await measure_case(
                        engine, cache, name, inputs, sampling(tokens, force), force,
                        concurrency, repeat,
                    )
                    report["runs"].append(result)
                    save(report)
                    print(f"  {result['aggregate_tokens_per_second']:.1f} aggregate tok/s; "
                          f"TTFT median {result['median_ttft_seconds']:.3f}s", flush=True)
    finally:
        htr.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run(parse_args()))
