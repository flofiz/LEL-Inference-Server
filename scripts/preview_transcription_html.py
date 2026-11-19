"""Render a local, self-contained review page from the HTML transcription API."""

import argparse
from base64 import b64encode
from hashlib import sha256
from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path


class Fragment(HTMLParser):
    # Model output is untrusted. Rebuild only the tags and attributes we render.
    tags = set(
        "p h1 h2 h3 h4 h5 h6 span table caption thead tbody tfoot tr th td "
        "aside signature page_number del br em strong i b sub sup".split()
    )

    def __init__(self, page, number):
        super().__init__()
        self.page, self.number = page, number
        self.parts, self.boxes = [], []

    def handle_starttag(self, tag, attrs):
        if tag not in self.tags:
            return
        attrs = dict(attrs)
        safe = {}
        for key in ("colspan", "rowspan"):
            value = attrs.get(key)
            if tag in ("td", "th") and value and value.isascii() and value.isdecimal():
                safe[key] = value
        try:
            x1, y1, x2, y2, angle = map(int, (attrs.get("coord") or "").split(","))
        except ValueError:
            pass
        else:
            if 0 <= x1 <= x2 <= 1000 and 0 <= y1 <= y2 <= 1000:
                index = len(self.boxes) + 1
                segment = f"{self.number}-{index}"
                safe["data-segment"] = segment
                w, h = self.page["width"] / 1000, self.page["height"] / 1000
                x, y = self.page["offsetX"] + x1 * w, y1 * h
                self.boxes.append(
                    f'<rect class="box page-{self.number % 2}" data-segment="{segment}" '
                    f'x="{x:.2f}" y="{y:.2f}" width="{(x2 - x1) * w:.2f}" '
                    f'height="{(y2 - y1) * h:.2f}" vector-effect="non-scaling-stroke">'
                    f'<title>Fragment {self.number}, segment {index}, angle {angle}°</title></rect>'
                )
        attributes = "".join(f' {key}="{escape(value, quote=True)}"' for key, value in safe.items())
        self.parts.append(f"<{tag}{attributes}>")

    def handle_endtag(self, tag):
        if tag in self.tags and tag != "br":
            self.parts.append(f"</{tag}>")

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_data(self, data):
        self.parts.append(escape(data))


STYLE = """
* { box-sizing: border-box; }
body { margin: 0; color: #263447; background: #eef1f6; font: 15px system-ui, sans-serif; }
.bar { background: #fff; padding: 14px 20px; border-bottom: 1px solid #cbd5e0; }
h1 { margin: 0; font-size: 22px; }
.controls { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 22px; margin-top: 9px; }
label { display: inline-flex; align-items: center; gap: 7px; }
#status { color: #475a70; }
main { display: grid; grid-template-columns: 56% 44%; height: calc(100dvh - 96px); }
.scan { background: #293447; overflow: auto; }
.scan svg { display: block; width: 100%; height: auto; }
.box { fill: #28e0fc22; stroke: #05daf5; stroke-width: 1.1; cursor: pointer; }
.box.page-0 { fill: #ffba4b23; stroke: #ffb346; }
.box.active { fill: #ffed3990; stroke: #fff423; stroke-width: 3; }
.hide-boxes .box { visibility: hidden; }
.transcript { min-width: 0; overflow: auto; padding: 15px; }
.fragment { background: white; border: 1px solid #d5dde7; border-radius: 8px; margin-bottom: 16px; padding: 14px; }
h2 { margin: 0 0 9px; font-size: 17px; }
.fragment div { overflow: auto; }
page_number, signature, aside { display: block; margin: 0.5em 0; }
table { border-collapse: collapse; width: max-content; max-width: 100%; margin: 9px 0; }
td, th { border: 1px solid #cbd5e0; padding: 3px 5px; min-width: 12px; }
thead, tfoot { background: #edf4f9; }
[data-segment] { cursor: pointer; }
.transcript [data-segment].active { background: #ffea8a; outline: 1px solid #d18c0c; }
@media (max-width: 850px) { main { display: block; height: auto; } .scan { max-height: 61vh; } }
"""

SCRIPT = """
const overlay = document.querySelector("#overlay");
const scan = document.querySelector(".scan");
const zoom = document.querySelector("#zoom");
const pairs = new Map();
function indexSegments() {
    pairs.clear();
    for (const element of document.querySelectorAll("[data-segment]")) {
        const key = element.dataset.segment;
        if (!pairs.has(key)) pairs.set(key, []);
        pairs.get(key).push(element);
    }
}
indexSegments();
let active = [];
let activeKey = null;
function select(key) {
    for (const element of active) element.classList.remove("active");
    activeKey = key;
    active = pairs.get(key) || [];
    for (const element of active) element.classList.add("active");
    const [fragment, index] = key.split("-");
    const text = active.find(element => element.closest(".transcript"))?.textContent.trim() || "";
    document.querySelector("#status").textContent =
        `Fragment ${fragment} · segment ${index}${text ? " · " + text.slice(0, 100) : ""}`;
}
document.querySelector("main").addEventListener("pointerover", event => {
    const target = event.target.closest("[data-segment]");
    if (target) select(target.dataset.segment);
});
document.querySelector("main").addEventListener("click", event => {
    const target = event.target.closest("[data-segment]");
    if (!target) return;
    select(target.dataset.segment);
    for (const element of active)
        if (element !== target) element.scrollIntoView({block: "center", inline: "center"});
});
overlay.addEventListener("change", () => scan.classList.toggle("hide-boxes", !overlay.checked));
zoom.addEventListener("input", () => {
    scan.querySelector("svg").style.width = zoom.value + "%";
    document.querySelector("#zoom-value").textContent = zoom.value + "%";
});
window.addEventListener("message", event => {
    if (window.parent === window || event.source !== window.parent || event.data?.type !== "page") return;
    const {pageIndex, fragment, boxes, segmentCount, finishReason} = event.data;
    if (!Number.isInteger(pageIndex)) return;
    const page = document.querySelector(`.fragment[data-page-index="${pageIndex}"]`);
    const group = scan.querySelector(`g[data-page-index="${pageIndex}"]`);
    if (!page || !group) return;
    // Only sanitized markup from the local relay reaches these containers.
    // The browser closes open table/paragraph tags provisionally.
    const content = page.querySelector(".fragment-body");
    const left = content.scrollLeft;
    content.innerHTML = fragment;
    content.scrollLeft = left;
    group.innerHTML = boxes;
    page.querySelector("h2").textContent =
        `Fragment ${pageIndex + 1} · ${segmentCount} segments (${finishReason || "en cours"})`;
    document.querySelector("#segment-count").textContent = scan.querySelectorAll(".box").length;
    indexSegments();
    if (activeKey && pairs.has(activeKey)) select(activeKey);
});
"""


def render_page(page, index, *, partial=False):
    fragment = Fragment(page, index + 1)
    fragment.feed(page["html"])
    if not partial:
        fragment.close()
    return {
        "type": "page", "pageIndex": index,
        "fragment": "".join(fragment.parts), "boxes": "".join(fragment.boxes),
        "segmentCount": len(fragment.boxes), "finishReason": page.get("finishReason"),
    }


def render(report, image, image_mime="image/jpeg"):
    fragments, boxes = [], []
    total = 0
    for index, page in enumerate(report["pages"]):
        fragment = render_page(page, index)
        total += fragment["segmentCount"]
        boxes.append(f'<g data-page-index="{index}">{fragment["boxes"]}</g>')
        reason = escape(str(fragment["finishReason"] or "en cours"))
        fragments.append(
            f'<section class="fragment" data-page-index="{index}"><h2>Fragment {index + 1} · '
            f'{fragment["segmentCount"]} segments <small>({reason})</small></h2>'
            f'<div class="fragment-body">{fragment["fragment"]}</div></section>'
        )
    width, height = int(report["imageWidth"]), int(report["imageHeight"])
    script_hash = b64encode(sha256(SCRIPT.encode()).digest()).decode()
    image_url = f"data:{image_mime};base64," + b64encode(image).decode()
    return f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'sha256-{script_hash}'">
<title>Aperçu de la transcription</title><style>{STYLE}</style></head>
<body><header class="bar"><h1>Image et transcription · <span id="segment-count">{total}</span> segments</h1>
<div class="controls">
<label><input id="overlay" type="checkbox" checked>Afficher les segments</label>
<label>Zoom <input id="zoom" type="range" min="100" max="350" step="10" value="100"> <output id="zoom-value">100%</output></label>
<span id="status">Cliquez sur un segment pour le retrouver dans l'autre volet.</span>
</div></header>
<main><div class="scan"><svg viewBox="0 0 {width} {height}" aria-label="Image originale et segments">
<image href="{image_url}" width="{width}" height="{height}"/>
{"".join(boxes)}</svg></div><div class="transcript">{"".join(fragments)}</div></main>
<script>{SCRIPT}</script></body></html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("image", type=Path, help="Original JPEG sent to /transcribe/html")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    args.output.write_text(render(report, args.image.read_bytes()), encoding="utf-8", newline="\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
