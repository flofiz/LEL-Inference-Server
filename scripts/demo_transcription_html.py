"""Local browser demo for the public HTML transcription endpoint (stdlib only)."""

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4
import webbrowser

from preview_transcription_html import render

UPSTREAM = "https://lel.cotedor.fr/transcribe/html"
UPLOAD_LIMIT = 30 * 1024 * 1024
RESPONSE_LIMIT = 2 * 1024 * 1024
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


def transcribe(image, mime):
    boundary = uuid4().hex
    extension = IMAGE_EXTENSIONS[mime]
    header = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="page.{extension}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode("ascii")
    request = Request(
        UPSTREAM, data=header + image + f"\r\n--{boundary}--\r\n".encode("ascii"),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=600) as response:
            if response.headers.get_content_type() != "application/json":
                if response.headers.get("Retry-After"):
                    raise DemoError(503, "Le serveur de transcription est en maintenance.")
                raise DemoError(502, "Le serveur n’a pas renvoyé de transcription JSON.")
            payload = response.read(RESPONSE_LIMIT + 1)
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

    if len(payload) > RESPONSE_LIMIT:
        raise DemoError(502, "La réponse du serveur est trop volumineuse.")
    try:
        report = json.loads(payload)
        if not report["pages"]:
            raise ValueError("empty transcription")
        return render(report, image, mime)
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
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if self.path == "/":
            self.reply(200, CLIENT_PAGE.read_text(encoding="utf-8"), "text/html")
        else:
            self.reply(404, "Page introuvable.")

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
            output = transcribe(image, mime)
        except DemoError as exc:
            self.reply(exc.status, str(exc))
        else:
            self.reply(HTTPStatus.OK, output, "text/html")


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
