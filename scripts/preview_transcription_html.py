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
    categories = {
        "aside": "aside", "signature": "signature", "page_number": "page-number",
        "table": "table", **{f"h{level}": "heading" for level in range(1, 7)},
    }

    def __init__(self, page, number):
        super().__init__()
        self.page, self.number = page, number
        self.parts, self.boxes = [], []
        self.context = []

    def handle_starttag(self, tag, attrs):
        if tag not in self.tags:
            return
        if tag in self.categories:
            self.context.append(tag)
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
                category = f" tag-{self.categories[self.context[-1]]}" if self.context else ""
                self.boxes.append(
                    f'<rect class="box page-{self.number % 2}{category}" data-segment="{segment}" '
                    f'x="{x:.2f}" y="{y:.2f}" width="{(x2 - x1) * w:.2f}" '
                    f'height="{(y2 - y1) * h:.2f}" vector-effect="non-scaling-stroke">'
                    f'<title>Fragment {self.number}, segment {index}, angle {angle}°</title></rect>'
                )
        attributes = "".join(f' {key}="{escape(value, quote=True)}"' for key, value in safe.items())
        self.parts.append(f"<{tag}{attributes}>")

    def handle_endtag(self, tag):
        if tag in self.context:
            last = len(self.context) - 1 - self.context[::-1].index(tag)
            del self.context[last:]
        if tag in self.tags and tag != "br":
            self.parts.append(f"</{tag}>")

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_data(self, data):
        self.parts.append(escape(data))


STYLE = """
* { box-sizing: border-box; }
body { margin: 0; height: 100dvh; display: flex; flex-direction: column;
    color: #242e38; background: #f1ede5; font: 14px system-ui, sans-serif; }
.bar, .controls { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 16px; }
.bar { background: #fffaf2; padding: 6px 12px; border-bottom: 1px solid #d8ccb7; }
.bar h1 { margin: 0; font-size: 14px; white-space: nowrap; }
.bar label { display: inline-flex; align-items: center; gap: 5px; }
#zoom { width: 85px; }
#status { flex: 1 1 140px; min-width: 0; color: #516074; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
main { display: grid; grid-template-columns: minmax(0, 55fr) minmax(0, 45fr); flex: 1; min-height: 0; }
.scan { background: #28323e; overflow: auto; min-height: 0; }
.scan svg { display: block; height: auto; margin-inline: auto; }
.box { fill: #28e0fc18; stroke: #05daf5; stroke-width: 1.1; cursor: pointer; }
.box.page-0 { fill: #ffba4b20; stroke: #ffb346; }
.box.tag-table { fill: #47be8c18; stroke: #2dcb96; }
.box.tag-aside { fill: #ad7bff29; stroke: #b78cff; }
.box.tag-signature { fill: #f57a7a22; stroke: #fa8a80; }
.box.tag-page-number { fill: #c4ccd629; stroke: #d9e1eb; }
.box.tag-heading { fill: #e9cc5a24; stroke: #e8c75d; }
.box.active { fill: #ffed3990; stroke: #fff423; stroke-width: 3; }
.hide-boxes .box { visibility: hidden; }
.transcript { min-width: 0; overflow: auto; padding: 10px; }
.fragment { background: #fffdf8; border: 1px solid #ded3be; border-radius: 8px; margin-bottom: 10px; padding: 12px; }
.fragment > h2 { margin: 0 0 9px; font-size: 14px; font-weight: 600; color: #4d5660; }
.fragment-body { overflow: auto; color: #27231d; font: 17px/1.5 Georgia, "Times New Roman", serif; }
.fragment-body p { margin: .6em 0; }
.fragment-body p > span[data-segment] { display: block; }
.fragment-body h1, .fragment-body h2, .fragment-body h3,
.fragment-body h4, .fragment-body h5, .fragment-body h6 { margin: .7em 0 .3em; color: #423725; }
.fragment-body aside { margin: .65em 0; padding: .4em .75em; border-left: 3px solid #8c63c6;
    background: #f3ecfc; color: #533875; font-style: italic; }
.fragment-body signature { display: block; margin: .65em 0; padding: .25em .6em;
    border-top: 1px dashed #d7a6a0; color: #772f2a; text-align: right; font-style: italic; font-weight: 600; }
.fragment-body page_number { display: inline-block; margin: .25em 0; padding: 1px 7px;
    border: 1px solid #ded7c9; border-radius: 5px; color: #515b69; background: #f2eee6;
    font: 12px/1.5 system-ui, sans-serif; font-variant-numeric: tabular-nums; }
.fragment-body del { color: #75665c; text-decoration-thickness: 1.5px; }
table { border-collapse: collapse; width: max-content; min-width: 100%; margin: 8px 0; font-size: 15px; line-height: 1.4; }
td, th { border: 1px solid #d5c7ad; padding: 3px 5px; min-width: 12px; vertical-align: top; }
thead, tfoot { background: #f0e7d5; }
[data-segment] { cursor: pointer; }
.transcript [data-segment].active { background: #ffea8a; outline: 1px solid #d18c0c; }
@media (max-width: 720px) {
    main { grid-template-columns: minmax(0, 1fr); grid-template-rows: minmax(0, 52fr) minmax(0, 48fr); }
}
"""

SCRIPT = """
const overlay = document.querySelector("#overlay");
const scan = document.querySelector(".scan");
const svg = scan.querySelector("svg");
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
function fitImage() {
    const {width, height} = scan.getBoundingClientRect();
    const image = svg.viewBox.baseVal;
    if (image.width && image.height) {
        const fittedWidth = Math.min(width, height * image.width / image.height);
        svg.style.width = fittedWidth * Number(zoom.value) / 100 + "px";
    }
    document.querySelector("#zoom-value").textContent = zoom.value + "%";
}
zoom.addEventListener("input", fitImage);
new ResizeObserver(fitImage).observe(scan);
fitImage();
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
    image_url = f"data:{image_mime};base64," + b64encode(image).decode()
    return render_shell(width, height, image_url, "".join(fragments), "".join(boxes), total)


def render_shell(width, height, image_url, fragments, boxes, total):
    script_hash = b64encode(sha256(SCRIPT.encode()).digest()).decode()
    return f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'sha256-{script_hash}'">
<title>Aperçu de la transcription</title><style>{STYLE}</style></head>
<body><header class="bar"><h1>Image et transcription · <span id="segment-count">{total}</span> segments</h1>
<div class="controls">
<label><input id="overlay" type="checkbox" checked>Afficher les segments</label>
<label title="100 % affiche l’image entière"><input id="zoom" type="range" min="100" max="350" step="10" value="100" aria-label="Zoom"> <output id="zoom-value">100%</output></label>
</div><span id="status">Cliquez sur un segment pour le retrouver.</span></header>
<main><div class="scan"><svg viewBox="0 0 {width} {height}" aria-label="Image originale et segments">
<image href="{image_url}" width="{width}" height="{height}"/>
{boxes}</svg></div><div class="transcript">{fragments}</div></main>
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
