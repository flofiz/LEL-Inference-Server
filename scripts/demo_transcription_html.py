"""Local browser demo for the public HTML transcription endpoint (stdlib only)."""

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from time import monotonic
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4
import webbrowser

from preview_transcription_html import render, render_page

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from processing.html_output import unwrap_html_fragment

UPSTREAM = "https://lel.cotedor.fr/transcribe/html/stream"
UPLOAD_LIMIT = 30 * 1024 * 1024
RESPONSE_LIMIT = 4 * 1024 * 1024
IMAGE_EXTENSIONS = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}
CLIENT_PAGE = Path(__file__).with_suffix(".html")


class DemoError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def image_matches_mime(image, mime):
    if mime == "image/jpeg":
        return image.startswith(b"\xff\xd8\xff")
    if mime == "image/png":
        return image.startswith(b"\x89PNG\r\n\x1a\n")
    return mime == "image/webp" and image[:4] == b"RIFF" and image[8:12] == b"WEBP"


def preview_text(raw):
    # Do not show a partially received opening Markdown fence.
    if "```html".startswith(raw.strip().lower()):
        return ""
    return unwrap_html_fragment(raw)


def transcribe(image, mime):
    """Read the server's NDJSON; emit safe, throttled updates for the demo."""
    boundary = uuid4().hex
    extension = IMAGE_EXTENSIONS[mime]
    header = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="page.{extension}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode("ascii")
    request = Request(
        UPSTREAM, data=header + image + f"\r\n--{boundary}--\r\n".encode("ascii"),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "Accept": "application/x-ndjson"},
    )
    try:
        with urlopen(request, timeout=600) as response:
            if response.headers.get_content_type() != "application/x-ndjson":
                if response.headers.get("Retry-After"):
                    raise DemoError(503, "Le serveur de transcription est en maintenance.")
                raise DemoError(502, "Le serveur n’a pas renvoyé de flux de transcription.")
            report = None
            received = 0
            for line in response:
                received += len(line)
                if received > RESPONSE_LIMIT:
                    raise DemoError(502, "La réponse du serveur est trop volumineuse.")
                event = json.loads(line)
                kind = event["type"]
                if kind == "metadata" and report is None:
                    pages = [dict(page, html="") for page in event["pages"]]
                    if not pages:
                        raise ValueError("empty transcription")
                    report = {**event, "pages": pages}
                    raw = [""] * len(pages)
                    last_update = [0.0] * len(pages)
                    yield {"type": "preview", "html": render(report, image, mime)}
                elif report is None:
                    raise ValueError("missing stream metadata")
                elif kind == "complete":
                    if any("finishReason" not in page for page in pages):
                        raise ValueError("missing page completion")
                    yield {"type": "complete", "html": render(report, image, mime),
                           "hasErrors": any(page["finishReason"] == "error" for page in pages)}
                    return
                elif kind in ("delta", "done", "error"):
                    index = event["pageIndex"]
                    if type(index) is not int or not 0 <= index < len(pages):
                        raise ValueError("invalid page index")
                    page = pages[index]
                    if kind == "delta":
                        raw[index] += event["html"]
                        now = monotonic()
                        if now - last_update[index] < 0.15:
                            continue
                        last_update[index] = now
                    else:
                        page["finishReason"] = event["finishReason"] if kind == "done" else "error"
                    page["html"] = preview_text(raw[index])
                    yield render_page(page, index, partial=kind == "delta")
            raise DemoError(502, "Le flux s’est interrompu avant la fin de la transcription.")
    except HTTPError as exc:
        if exc.code in (408, 504):
            raise DemoError(504, "Le serveur a dépassé son délai. Essayez une page plus courte.") from exc
        if exc.code == 413:
            raise DemoError(413, "Le serveur a refusé cette image : le fichier est trop volumineux.") from exc
        raise DemoError(502, f"Le serveur de transcription a répondu HTTP {exc.code}.") from exc
    except TimeoutError as exc:
        raise DemoError(504, "Aucune réponse du serveur après dix minutes.") from exc
    except (URLError, OSError) as exc:
        raise DemoError(502, "Impossible de joindre le serveur de transcription.") from exc
    except (TypeError, ValueError, KeyError) as exc:
        raise DemoError(502, "La transcription renvoyée est illisible.") from exc


class DemoHandler(BaseHTTPRequestHandler):
    def reply(self, status, content, mime="text/plain"):
        body = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{mime}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(body)
        except ConnectionError:
            pass

    def do_GET(self):
        if self.path == "/":
            self.reply(200, CLIENT_PAGE.read_text(encoding="utf-8"), "text/html")
        else:
            self.reply(404, "Page introuvable.")

    def write_event(self, item):
        self.wfile.write((json.dumps(item, ensure_ascii=False) + "\n").encode("utf-8"))
        self.wfile.flush()

    def do_POST(self):
        if self.path != "/transcribe":
            self.reply(404, "Page introuvable.")
            return
        origin = f"http://127.0.0.1:{self.server.server_port}"
        if self.headers.get("Origin") != origin:
            self.reply(403, "Cette route est réservée à la page locale de démo.")
            return
        try:
            mime = self.headers.get_content_type()
            if mime not in IMAGE_EXTENSIONS:
                raise DemoError(415, "Choisissez un fichier JPEG, PNG ou WebP.")
            try:
                size = int(self.headers.get("Content-Length", ""))
            except ValueError:
                size = 0
            if not 0 < size <= UPLOAD_LIMIT:
                raise DemoError(413, "Choisissez une image non vide de 30 Mio maximum.")
            image = self.rfile.read(size)
            if len(image) != size or not image_matches_mime(image, mime):
                raise DemoError(400, "Le contenu ne correspond pas au format d’image annoncé.")
        except DemoError as exc:
            self.reply(exc.status, str(exc))
            return

        with_events = transcribe(image, mime)
        try:
            first = next(with_events)
        except DemoError as exc:
            self.reply(exc.status, str(exc))
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.write_event(first)
            try:
                for item in with_events:
                    self.write_event(item)
            except DemoError as exc:
                self.write_event({"type": "error", "message": str(exc)})
        except ConnectionError:
            pass
        finally:
            with_events.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically")
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", args.port), DemoHandler) as server:
        address = f"http://127.0.0.1:{server.server_port}/"
        print(f"Démo de transcription : {address}", flush=True)
        if not args.no_browser:
            webbrowser.open(address)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
