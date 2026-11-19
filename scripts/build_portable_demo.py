"""Build the single-file demo; recipients do not need Python."""

import json
from pathlib import Path

from preview_transcription_html import render_shell


def main():
    directory = Path(__file__).resolve().parent
    shell = render_shell("@@width@@", "@@height@@", "@@image@@", "@@fragments@@", "@@boxes@@", "@@total@@")
    shell_literal = json.dumps(shell, ensure_ascii=False).replace("<", "\\u003c")
    portable_js = (directory / "portable_transcription_html.js").read_text(encoding="utf-8")
    client = (directory / "demo_transcription_html.html").read_text(encoding="utf-8")
    client = client.replace("connect-src 'self';", "connect-src 'self' https://lel.cotedor.fr;")
    client = client.replace("<!-- portable client -->",
                            f"<script>const portableViewerShell = {shell_literal};\n{portable_js}</script>")
    (directory.parent / "demo-lettres-en-lumieres.html").write_text(client, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
