"""Tests for the HTML transcription path without a GPU or Ray Serve."""

import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from starlette.exceptions import HTTPException


@pytest.fixture
def html_output(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "src"))
    from processing import html_output

    return html_output


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ('```html\n<p><span coord="1,2,3,4,0">Un  texte</span></p>\n```',
         '<p><span coord="1,2,3,4,0">Un  texte</span></p>'),
        (' ```HTML\r\n<table><td rowspan="2"></td></table>\r\n```\r\n',
         '<table><td rowspan="2"></td></table>'),
        ("```html\n<p>Fragment tronqué", "<p>Fragment tronqué"),
        ("<p>Texte ``` conservé</p>", "<p>Texte ``` conservé</p>"),
    ],
)
def test_unwrap_html_only_removes_outer_fences(html_output, text, fragment):
    assert html_output.unwrap_html_fragment(text) == fragment


def test_html_transcription_preserves_crop_order_and_budget(html_output, monkeypatch):
    from inference.engine import HTREngine
    from templates.tasks.HTML import HTML_PROMPT

    def get_inputs(contents, tokenizer, prompt, system, **options):
        assert contents == b"image"
        assert prompt == HTML_PROMPT
        assert options == {"qwen_version": "qwen35", "enable_thinking": False}
        yield {"side": "left"}, (1000 / 600, 1000 / 900), 0
        yield {"side": "right"}, (1000 / 620, 1000 / 900), 550

    ids = iter(("left-id", "right-id"))
    monkeypatch.setitem(sys.modules, "processing.preprocessing", SimpleNamespace(get_inputs=get_inputs))
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", SimpleNamespace(
        RequestOutputKind=SimpleNamespace(FINAL_ONLY="final"),
        SamplingParams=lambda **kwargs: kwargs,
    ))
    monkeypatch.setitem(sys.modules, "vllm.utils", SimpleNamespace(random_uuid=lambda: next(ids)))
    completed = []

    async def generate(inputs, params, request_id):
        assert params == {"temperature": 0.0, "max_tokens": 16_384, "output_kind": "final"}
        side = inputs["side"]
        assert request_id == side + "-id"
        if side == "left":
            await asyncio.sleep(0)
        completed.append(side)
        yield SimpleNamespace(outputs=[SimpleNamespace(
            text=f'```html\n<p id="{side}">Écrit</p>\n```',
            finish_reason="length" if side == "right" else "stop",
        )])

    async def read_image(request):
        return b"image", {"imageWidth": 1170, "imageHeight": 900, "model": "qwen"}

    htr = HTREngine.__new__(HTREngine)
    htr.enable_thinking = False
    htr._read_image = read_image
    config = SimpleNamespace(max_model_len=html_output.HTML_CONTEXT_LENGTH)
    htr.engine = SimpleNamespace(
        vllm_config=SimpleNamespace(model_config=config),
        get_tokenizer=lambda: None,
        generate=generate,
    )

    response = asyncio.run(htr.transcribe_html(None))
    assert completed == ["right", "left"]
    assert response.media_type == "application/json"
    assert json.loads(response.body) == {
        "imageWidth": 1170,
        "imageHeight": 900,
        "model": "qwen",
        "pages": [
            {"html": '<p id="left">Écrit</p>', "offsetX": 0, "width": 600, "height": 900, "finishReason": "stop"},
            {"html": '<p id="right">Écrit</p>', "offsetX": 550, "width": 620, "height": 900, "finishReason": "length"},
        ],
    }
    config.max_model_len = 8192
    with pytest.raises(HTTPException) as error:
        asyncio.run(htr.transcribe_html(None))
    assert error.value.status_code == 503


def test_html_stream_keeps_each_crops_deltas_ordered(html_output):
    async def run():
        right_ready = asyncio.Event()

        def result(text, reason=None):
            return SimpleNamespace(outputs=[SimpleNamespace(text=text, finish_reason=reason)])

        async def left():
            yield result("<table>")
            await right_ready.wait()
            yield result("</table>", "stop")

        async def right():
            yield result("<p>Droite</p>", "length")
            right_ready.set()

        pages = [{"offsetX": 0}, {"offsetX": 100}]
        return [json.loads(line) async for line in html_output.stream_html_fragments(
            {"imageWidth": 200}, pages, [left(), right()]
        )]

    events = asyncio.run(run())
    assert events[0]["pages"] == [{"pageIndex": 0, "offsetX": 0}, {"pageIndex": 1, "offsetX": 100}]
    deltas = [(event["pageIndex"], event["html"]) for event in events if event["type"] == "delta"]
    assert deltas == [(0, "<table>"), (1, "<p>Droite</p>"), (0, "</table>")]
    assert {event["pageIndex"]: event["finishReason"] for event in events if event["type"] == "done"} == {
        0: "stop", 1: "length",
    }
    assert events[-1] == {"type": "complete"}


def test_closing_html_stream_cancels_active_generation(html_output):
    async def run():
        closed = asyncio.Event()

        async def output():
            try:
                yield SimpleNamespace(outputs=[SimpleNamespace(text="<table>", finish_reason=None)])
                await asyncio.Future()
            finally:
                closed.set()

        stream = html_output.stream_html_fragments({}, [{}], [output()])
        await anext(stream)  # metadata
        await anext(stream)  # first delta
        await stream.aclose()
        assert closed.is_set()

    asyncio.run(run())
