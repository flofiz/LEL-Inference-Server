"""Best-effort NVML telemetry through nvidia-smi, without profiling CUDA."""

import asyncio
from statistics import mean
from time import perf_counter

FIELDS = (
    "utilization.gpu", "utilization.memory", "memory.used", "memory.total",
    "power.draw", "clocks.sm", "clocks.mem",
)


class GpuSampler:
    def __init__(self, device):
        self.device = device
        self.process = self.reader = self.error = None
        self.samples = []

    async def __aenter__(self):
        try:
            self.process = await asyncio.create_subprocess_exec(
                "nvidia-smi", "-i", self.device, "--query-gpu=" + ",".join(FIELDS),
                "--format=csv,noheader,nounits", "--loop-ms=1000",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            )
            self.reader = asyncio.create_task(self._collect())
        except OSError as exc:
            self.error = str(exc)
        return self

    async def _collect(self):
        async for line in self.process.stdout:
            if len(values := line.decode(errors="replace").strip().split(",")) != len(FIELDS):
                continue
            sample = {}
            for key, value in zip(FIELDS, values):
                try:
                    sample[key] = float(value)
                except ValueError:
                    continue
            if sample:
                self.samples.append((perf_counter(), sample))

    def reset(self):
        self.samples.clear()

    def summarize(self, started, finished):
        samples = [sample for instant, sample in self.samples if started <= instant <= finished]
        fields = {}
        for key in FIELDS:
            values = [sample[key] for sample in samples if key in sample]
            if values:
                fields[key] = {"mean": mean(values), "peak": max(values)}
        return {"control": "nvidia-smi", "samples": len(samples), "fields": fields,
                "error": self.error or ("no samples; check GPU index and nvidia-smi" if not samples else None)}

    async def __aexit__(self, *_):
        if self.process is not None:
            if self.process.returncode is None:
                try:
                    self.process.terminate()
                except ProcessLookupError:
                    pass
            await self.process.wait()
            await self.reader
