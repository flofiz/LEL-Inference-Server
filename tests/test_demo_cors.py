from pathlib import Path

from fastapi import FastAPI
from starlette.responses import StreamingResponse
from starlette.testclient import TestClient


def test_portable_cors_only_allows_the_html_stream_from_an_opaque_origin(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "src"))
    from api.demo_cors import PortableDemoCors

    app = FastAPI()
    app.add_middleware(PortableDemoCors)

    @app.post("/transcribe/html/stream")
    @app.post("/transcribe")
    def generate():
        return StreamingResponse(iter(['{"type":"complete"}\n']), media_type="application/x-ndjson")

    with TestClient(app) as client:
        reply = client.post("/transcribe/html/stream", headers={"Origin": "null"})
        assert reply.headers["access-control-allow-origin"] == "null"
        assert reply.headers["vary"] == "Origin"
        assert "access-control-allow-credentials" not in reply.headers
        assert reply.json() == {"type": "complete"}
        assert "access-control-allow-origin" not in client.post(
            "/transcribe", headers={"Origin": "null"},
        ).headers
        assert "access-control-allow-origin" not in client.post(
            "/transcribe/html/stream", headers={"Origin": "https://other.example"},
        ).headers
