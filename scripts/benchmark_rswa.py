"""Benchmark the RSWA engine directly on the CUDA host, with Ray Serve stopped."""

import argparse
import asyncio
from contextlib import AsyncExitStack
from importlib.metadata import distribution, version
from itertools import cycle, islice
import json
import logging
import math
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
    for flag in ("model", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--long-image", type=Path)
    parser.add_argument("--normal-image", type=Path, nargs="+", default=[])
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--concurrency", type=positive_int, nargs="+", default=[1, 4])
    parser.add_argument("--repeats", type=positive_int, default=1)
    parser.add_argument("--normal-max-tokens", type=positive_int, default=4096)
    parser.add_argument("--long-lengths", type=positive_int, nargs="*", default=[8192, 16384])
    parser.add_argument("--long-max-tokens", type=positive_int, default=16384)
    parser.add_argument("--max-model-len", type=positive_int, default=24576)
    parser.add_argument("--max-num-batched-tokens", type=positive_int, default=4096)
    parser.add_argument("--max-num-seqs", type=positive_int)
    parser.add_argument("--mtp-tokens", type=int, choices=(0, 1, 2, 3), default=1)
    parser.add_argument("--format", choices=("html", "tsv"), default="tsv")
    parser.add_argument("--suite", choices=("normal", "long", "both"), default="both")
    parser.add_argument("--single-wave", action="store_true")
    parser.add_argument("--wave-timeout", type=positive_int, default=1800)
    parser.add_argument("--gpu-index", help="Physical nvidia-smi GPU index or UUID to sample")
    args = parser.parse_args()
    if (args.suite != "long" and not args.normal_image) or (args.suite != "normal" and not args.long_image):
        parser.error("provide the images required by --suite")
    budgets = [args.normal_max_tokens] if args.suite != "long" else []
    if args.suite != "normal":
        budgets += args.long_lengths or [args.long_max_tokens]
    if max(budgets) >= args.max_model_len:
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
        self.running = self.waiting = self.preemptions = 0
        self.drafts = self.accepted = 0

    def record(self, scheduler_stats, iteration_stats, mm_cache_stats=None, engine_idx=0):
        if scheduler_stats is not None:
            usage = scheduler_stats.kv_cache_usage
            self.peak = usage if self.peak is None else max(self.peak, usage)
            self.running = max(self.running, getattr(scheduler_stats, "num_running_reqs", 0))
            waiting = (getattr(scheduler_stats, "num_waiting_reqs", 0)
                       + getattr(scheduler_stats, "num_skipped_waiting_reqs", 0))
            self.waiting = max(self.waiting, waiting)
            if spec := getattr(scheduler_stats, "spec_decoding_stats", None):
                self.drafts += spec.num_draft_tokens
                self.accepted += spec.num_accepted_tokens
        if iteration_stats is not None:
            self.preemptions += getattr(iteration_stats, "num_preempted_reqs", 0)

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


async def measure_case(engine, cache, name, inputs, params, force, concurrency, repeat,
                       *, single_wave=False, telemetry=None):
    slots = asyncio.Semaphore(concurrency)

    async def measure(crop, payload):
        async with slots:
            return await measure_request(engine, payload, params, crop, force)

    cache.reset()
    if telemetry is not None:
        telemetry.reset()
    started = perf_counter()
    # Exercise every crop at every concurrency; portrait -> one crop, double page -> two.
    batch = list(enumerate(inputs, start=1))
    workload = list(islice(cycle(batch), concurrency)) if single_wave else batch * concurrency
    requests = await asyncio.gather(*(measure(crop, payload) for crop, payload in workload))
    finished = perf_counter()
    elapsed = finished - started
    decode_rates = [r["decode_tokens_per_second"] for r in requests if r["decode_tokens_per_second"] is not None]
    return {
        "case": name,
        "max_output_tokens_per_crop": params.max_tokens,
        "forced_length": force,
        "concurrency": concurrency,
        "fragments_submitted": len(requests),
        "repeat": repeat,
        "wall_seconds": elapsed,
        "median_ttft_seconds": median(r["ttft_seconds"] for r in requests),
        "p95_ttft_seconds": sorted(r["ttft_seconds"] for r in requests)[math.ceil(0.95 * len(requests)) - 1],
        "median_decode_tokens_per_second": median(decode_rates) if decode_rates else None,
        "aggregate_tokens_per_second": sum(r["generated_tokens"] for r in requests) / elapsed,
        "fragments_per_minute": 60 * len(requests) / elapsed,
        "truncated_fragments": sum(r["finish_reason"] == "length" for r in requests),
        "peak_observed_kv_cache_fraction": cache.peak,
        "scheduler": {"peak_running": cache.running, "peak_waiting": cache.waiting, "preemptions": cache.preemptions},
        "mtp": {"draft_tokens": cache.drafts, "accepted_tokens": cache.accepted,
                "acceptance_fraction": cache.accepted / cache.drafts if cache.drafts else None},
        "gpu": telemetry.summarize(started, finished) if telemetry is not None else None,
        "requests": requests,
    }


async def run(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from inference.engine import HTREngine, HTML_PROMPT, PROMPT, SYSTEM, latin1_regex
    from inference.rswa_patch import EDITS, MTP_EDITS, digest

    def save(report):
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    cache = CachePeak()
    max_seqs = args.max_num_seqs or max(args.concurrency)
    htr = HTREngine(
        name=args.model.name,
        model=str(args.model.resolve()),
        require_rswa=True,
        mtp_tokens=args.mtp_tokens,
        enable_thinking=False,
        stat_loggers=[cache],
        max_model_len=args.max_model_len,
        max_num_seqs=max_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        limit_mm_per_prompt={"image": 1},
    )
    stack = AsyncExitStack()
    try:
        import torch
        from processing.preprocessing import get_inputs
        from vllm.sampling_params import RequestOutputKind, SamplingParams, StructuredOutputsParams

        evidence = await htr.verify_ready()
        engine = htr.engine
        tokenizer = engine.get_tokenizer()
        telemetry = None
        if args.gpu_index is not None:
            from benchmark_gpu import GpuSampler

            telemetry = await stack.enter_async_context(GpuSampler(args.gpu_index))

        def load_image(path):
            return [inputs for inputs, _, _ in get_inputs(
                path.read_bytes(), tokenizer, HTML_PROMPT if args.format == "html" else PROMPT,
                SYSTEM, qwen_version="qwen35",
                enable_thinking=False,
            )]

        def sampling(tokens, force=False):
            options = {}
            if args.format == "tsv":
                options = {"logprobs": 1, "structured_outputs": StructuredOutputsParams(regex=latin1_regex)}
            return SamplingParams(
                temperature=0.0,
                max_tokens=tokens,
                min_tokens=tokens if force else 0,
                ignore_eos=force,
                output_kind=RequestOutputKind.DELTA,
                **options,
            )

        normal = [crop for path in args.normal_image for crop in load_image(path)] if args.suite != "long" else []
        long = load_image(args.long_image) if args.suite != "normal" else []
        installed = distribution("vllm")
        sources = EDITS | (MTP_EDITS if args.mtp_tokens > 1 else {})
        report = {
            "label": args.label,
            "model": str(args.model.resolve()),
            "images": {"normal": [str(path) for path in args.normal_image],
                       "long": str(args.long_image) if args.long_image else None},
            "format": args.format,
            "single_wave": args.single_wave,
            "gpu_index": args.gpu_index,
            "hardware": torch.cuda.get_device_name(),
            "versions": {name: version(name) for name in ("vllm", "torch", "transformers")},
            "source_sha256": {
                path: digest(Path(installed.locate_file(path)).read_bytes()) for path in sources
            },
            "engine": {
                "max_model_len": engine.vllm_config.model_config.max_model_len,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "max_num_seqs": max_seqs,
                "kv_pool_blocks": engine.vllm_config.cache_config.num_gpu_blocks,
                "thinking": False,
                "mtp_tokens": args.mtp_tokens,
                "rswa": evidence,
            },
            "runs": [],
        }
        save(report)
        print(f"Inputs: {len(normal)} normal crop(s), {len(long)} long crop(s)", flush=True)
        for concurrency in args.concurrency:
            for images in (normal, long):
                if images:
                    await asyncio.wait_for(measure_case(
                        engine, cache, "warmup", images, sampling(32, force=True),
                        True, concurrency, 0, single_wave=args.single_wave,
                    ), timeout=args.wave_timeout)
            for repeat in range(1, args.repeats + 1):
                offset = (repeat - 1) % len(normal) if normal else 0
                normal_batch = normal[offset:] + normal[:offset]
                cases = [("normal", normal_batch, args.normal_max_tokens, False)] if normal else []
                if long:
                    cases += [("long", long, length, True) for length in args.long_lengths]
                    if not args.long_lengths:
                        cases.append(("long-natural", long, args.long_max_tokens, False))
                for name, inputs, tokens, force in cases:
                    print(f"{name}: concurrency={concurrency}, output budget={tokens}, repeat={repeat}", flush=True)
                    result = await asyncio.wait_for(measure_case(
                        engine, cache, name, inputs, sampling(tokens, force), force,
                        concurrency, repeat, single_wave=args.single_wave, telemetry=telemetry,
                    ), timeout=args.wave_timeout)
                    report["runs"].append(result)
                    save(report)
                    print(f"  {result['aggregate_tokens_per_second']:.1f} aggregate tok/s; "
                          f"TTFT median {result['median_ttft_seconds']:.3f}s", flush=True)
    finally:
        try:
            await stack.aclose()
        finally:
            htr.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run(parse_args()))
