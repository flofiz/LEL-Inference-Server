"""HTML output stays a fragment; do not parse, rewrite or render model HTML."""

import asyncio
from contextlib import aclosing
import json
import logging
import re

HTML_MAX_TOKENS = 16_384
HTML_CONTEXT_LENGTH = 24_576

_OPEN_FENCE = re.compile(r"\A\s*```html[ \t]*\r?\n", re.IGNORECASE)
_CLOSE_FENCE = re.compile(r"(?:\A|\r?\n)[ \t]*```[ \t]*(?:\r?\n)*\Z")


def unwrap_html_fragment(text):
    opening = _OPEN_FENCE.match(text)
    if opening is None:
        return text
    return _CLOSE_FENCE.sub("", text[opening.end():])


async def stream_html_fragments(metadata, pages, outputs):
    """Multiplex one delta iterator per crop as independently ordered NDJSON."""
    def encode(event):
        return (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")

    queue = asyncio.Queue(maxsize=64)

    async def forward(page_index, iterator):
        try:
            finish_reason = None
            async with aclosing(iterator) as stream:
                async for result in stream:
                    output, = result.outputs
                    if output.text:
                        await queue.put({"type": "delta", "pageIndex": page_index, "html": output.text})
                    if output.finish_reason is not None:
                        finish_reason = output.finish_reason
            if finish_reason is None:
                raise RuntimeError("vLLM returned no final HTML status")
            await queue.put({"type": "done", "pageIndex": page_index, "finishReason": finish_reason})
        except Exception:
            logging.getLogger("ray.serve").exception("HTML generation failed for page %d", page_index)
            await queue.put({"type": "error", "pageIndex": page_index, "message": "HTML generation failed"})

    yield encode({"type": "metadata", **metadata,
                  "pages": [{"pageIndex": i, **page} for i, page in enumerate(pages)]})
    tasks = [asyncio.create_task(forward(i, iterator)) for i, iterator in enumerate(outputs)]
    try:
        remaining = len(tasks)
        while remaining:
            item = await queue.get()
            remaining -= item["type"] in ("done", "error")
            yield encode(item)
        yield encode({"type": "complete"})
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
