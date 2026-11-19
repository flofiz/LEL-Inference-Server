import json
from email.message import Message
from pathlib import Path

import pytest


@pytest.fixture
def demo(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import demo_transcription_html

    return demo_transcription_html


def test_demo_posts_image_and_renders_partial_tables(demo, monkeypatch):
    image = b"\x89PNG\r\n\x1a\n"
    assert demo.image_matches_mime(image, "image/png")
    assert not demo.image_matches_mime(image, "image/jpeg")

    class Response:
        def __enter__(self):
            self.headers = Message()
            self.headers["Content-Type"] = "application/x-ndjson"
            return self

        def __exit__(self, *args):
            pass

        def __iter__(self):
            events = [
                {"type": "metadata", "imageWidth": 100, "imageHeight": 100,
                 "pages": [{"pageIndex": 0, "offsetX": 0, "width": 100, "height": 100}]},
                {"type": "delta", "pageIndex": 0, "html": "```ht"},
                {"type": "delta", "pageIndex": 0,
                 "html": 'ml\n<table><tr><td coord="10,20,30,40,0" onclick="evil()">Texte'},
                {"type": "delta", "pageIndex": 0, "html": "</td></tr></table>\n```"},
                {"type": "done", "pageIndex": 0, "finishReason": "stop"},
                {"type": "complete"},
            ]
            yield from (json.dumps(event).encode() + b"\n" for event in events)

    def post(request, timeout):
        assert request.full_url == "https://lel.cotedor.fr/transcribe/html/stream"
        assert request.get_method() == "POST"
        assert timeout == 600
        assert b'name="image"; filename="page.png"' in request.data
        assert image in request.data
        return Response()

    monkeypatch.setattr(demo, "urlopen", post)
    times = iter((1.0, 1.2, 1.21))
    monkeypatch.setattr(demo, "monotonic", lambda: next(times))
    preview, empty, partial, final, complete = demo.transcribe(image, "image/png")
    assert "data:image/png;base64," in preview["html"]
    assert empty["fragment"] == ""
    assert partial["pageIndex"] == 0 and partial["finishReason"] is None
    assert partial["fragment"] == '<table><tr><td data-segment="1-1">Texte'
    assert 'x="1.00" y="2.00" width="2.00" height="2.00"' in partial["boxes"]
    assert final["fragment"].endswith("</td></tr></table>")
    assert "onclick" not in complete["html"]
    assert not complete["hasErrors"]
