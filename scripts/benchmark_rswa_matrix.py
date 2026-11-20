"""Run separate RSWA engines sequentially for an HTML MTP/concurrency sweep."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

WORKER = Path(__file__).with_name("benchmark_rswa.py")
DEFAULT_LEVELS = (1, 2, 4, 8, 16, 32, 48, 64, 80, 90, 96, 128)
FIELDS = (
    "mtp_tokens", "batched_tokens", "case", "forced", "output_cap", "concurrency", "repeat",
    "tokens_s", "fragments_min", "ttft_p50_s", "ttft_p95_s", "kv_peak_fraction",
    "running_peak", "waiting_peak", "preemptions", "truncated", "acceptance_fraction",
    "gpu_active_mean_pct", "memory_controller_mean_pct", "vram_peak_mib", "gpu_samples",
)


def worker_command(args, k, budget, report):
    command = [
        sys.executable, str(WORKER), "--model", str(args.model), "--output", str(report),
        "--label", f"k{k}-t{budget}", "--format", "html", "--suite", args.suite,
        "--single-wave", "--mtp-tokens", str(k), "--gpu-index", args.gpu,
        "--max-num-seqs", str(args.max_num_seqs),
        "--max-num-batched-tokens", str(budget), "--repeats", str(args.repeats),
        "--normal-max-tokens", str(args.normal_max_tokens),
        "--wave-timeout", str(args.wave_timeout),
        "--concurrency", *map(str, args.levels), "--long-lengths", *map(str, args.long_lengths),
    ]
    if args.normal_image:
        command += ["--normal-image", *map(str, args.normal_image)]
    if args.long_image:
        command += ["--long-image", str(args.long_image)]
    return command


def summary_row(run, k, budget):
    gpu = run.get("gpu") or {}
    fields = gpu.get("fields", {})
    return {
        "mtp_tokens": k, "batched_tokens": budget, "case": run["case"],
        "forced": run["forced_length"], "output_cap": run["max_output_tokens_per_crop"],
        "concurrency": run["concurrency"], "repeat": run["repeat"],
        "tokens_s": run["aggregate_tokens_per_second"], "fragments_min": run["fragments_per_minute"],
        "ttft_p50_s": run["median_ttft_seconds"], "ttft_p95_s": run["p95_ttft_seconds"],
        "kv_peak_fraction": run["peak_observed_kv_cache_fraction"],
        "running_peak": run["scheduler"]["peak_running"], "waiting_peak": run["scheduler"]["peak_waiting"],
        "preemptions": run["scheduler"]["preemptions"], "truncated": run["truncated_fragments"],
        "acceptance_fraction": run["mtp"]["acceptance_fraction"],
        "gpu_active_mean_pct": fields.get("utilization.gpu", {}).get("mean"),
        "memory_controller_mean_pct": fields.get("utilization.memory", {}).get("mean"),
        "vram_peak_mib": fields.get("memory.used", {}).get("peak"), "gpu_samples": gpu.get("samples", 0),
    }


def run(args):
    args.out_dir.mkdir(parents=True, exist_ok=False)
    env = os.environ | {"CUDA_VISIBLE_DEVICES": args.gpu, "PYTHONUNBUFFERED": "1"}
    attempts, rows = [], []
    for budget in args.batched_tokens:
        for k in args.mtp:
            directory = args.out_dir / f"k{k}-t{budget}"
            directory.mkdir()
            report = directory / "report.json"
            log = directory / "engine.log"
            command = worker_command(args, k, budget, report)
            print(f"MTP={k}, max_num_batched_tokens={budget} → {log}", flush=True)
            with log.open("w", encoding="utf-8") as stream:
                process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
            attempts.append({
                "mtp_tokens": k, "batched_tokens": budget, "return_code": process.returncode,
                "command": command, "report": str(report) if report.exists() else None, "log": str(log),
            })
            if report.exists():
                result = json.loads(report.read_text(encoding="utf-8"))
                rows.extend(summary_row(run, k, budget) for run in result["runs"])
            (args.out_dir / "matrix.json").write_text(json.dumps(attempts, indent=2) + "\n", encoding="utf-8")
            with (args.out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            if process.returncode:
                print(f"Mode incomplete (exit {process.returncode}); inspect {log}.", flush=True)
    print(f"Summary: {args.out_dir / 'summary.csv'}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--normal-image", type=Path, nargs="+", default=[])
    parser.add_argument("--long-image", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True, help="New directory for results")
    parser.add_argument("--suite", choices=("normal", "long", "both"), default="both")
    parser.add_argument("--mtp", type=int, choices=(0, 1, 2, 3), nargs="+", default=(0, 1, 2, 3))
    parser.add_argument("--levels", type=int, nargs="+", default=DEFAULT_LEVELS, help="Offered vLLM fragments per wave")
    parser.add_argument("--batched-tokens", type=int, nargs="+", default=(16384,))
    parser.add_argument("--max-num-seqs", type=int, default=90)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--normal-max-tokens", type=int, default=16384)
    parser.add_argument("--wave-timeout", type=int, default=1800)
    parser.add_argument("--long-lengths", type=int, nargs="*", default=(), help="Force these output lengths; default: stop naturally")
    parser.add_argument("--gpu", default="0", help="Physical CUDA device index or GPU UUID")
    args = parser.parse_args()
    if (args.suite != "long" and not args.normal_image) or (args.suite != "normal" and not args.long_image):
        parser.error("provide the images required by --suite")
    for values in (args.mtp, args.levels, args.batched_tokens, args.long_lengths):
        if len(set(values)) != len(values):
            parser.error("duplicate levels are not supported")
    if min(*args.levels, *args.batched_tokens, *args.long_lengths,
           args.max_num_seqs, args.repeats, args.normal_max_tokens, args.wave_timeout) < 1:
        parser.error("counts must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
