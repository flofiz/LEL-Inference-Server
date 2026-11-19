import json
from email.message import Message
from pathlib import Path

import pytest


@pytest.fixture
def demo(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import demo_transcription_html

    return demo_transcription_html


def test_demo_posts_image_and_renders_server_fragments(demo, monkeypatch):
    image = b"\x89PNG\r\n\x1a\n"
    assert demo.image_matches_mime(image, "image/png")
    assert not demo.image_matches_mime(image, "image/jpeg")

    class Response:
        def __enter__(self):
            self.headers = Message()
            self.headers["Content-Type"] = "application/json"
            return self

        def __exit__(self, *args):
            pass

        def read(self, limit):
            return json.dumps({
                "imageWidth": 100, "imageHeight": 100,
                "pages": [{
                    "offsetX": 0, "width": 100, "height": 100,
                    "html": '<p coord="10,20,30,40,0" onclick="evil()">Texte</p>',
                }],
            }).encode()

    def post(request, timeout):
        assert request.full_url == "https://lel.cotedor.fr/transcribe/html"
        assert request.get_method() == "POST"
        assert timeout == 600
        assert b'name="image"; filename="page.png"' in request.data
        assert image in request.data
        return Response()

    monkeypatch.setattr(demo, "urlopen", post)
    html = demo.transcribe(image, "image/png")
    assert "data:image/png;base64," in html
    assert '<p data-segment="1-1">Texte</p>' in html
    assert 'x="1.00" y="2.00" width="2.00" height="2.00"' in html
    assert "onclick" not in html
