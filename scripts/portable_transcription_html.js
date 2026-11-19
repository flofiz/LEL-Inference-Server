const publicStream = "https://lel.cotedor.fr/transcribe/html/stream";
const allowedTags = new Set("p h1 h2 h3 h4 h5 h6 span table caption thead tbody tfoot tr th td aside signature page_number del br em strong i b sub sup".split(" "));
const tagTypes = {aside: "aside", signature: "signature", page_number: "page-number", table: "table",
    h1: "heading", h2: "heading", h3: "heading", h4: "heading", h5: "heading", h6: "heading"};

async function openPortableTranscription(file) {
    const form = new FormData();
    form.append("image", file);
    let response;
    try {
        response = await fetch(publicStream, {method: "POST", body: form, credentials: "omit"});
    } catch {
        throw new Error("Impossible de lire le serveur public. Vérifiez la connexion et l’autorisation CORS de la démo.");
    }
    if (!response.ok) throw new Error(`Le serveur a répondu HTTP ${response.status}.`);
    if (!response.headers.get("Content-Type")?.includes("application/x-ndjson"))
        throw new Error("Le serveur n’a pas renvoyé de flux de transcription (maintenance possible).");
    return response;
}

function portablePage(page, pageIndex) {
    const source = document.createElement("template");
    const result = document.createElement("div");
    const boxes = [];
    const text = page.raw;
    if (!"```html".startsWith(text.trim().toLowerCase())) {
        const opening = /^\s*```html[ \t]*\r?\n/i;
        source.innerHTML = opening.test(text)
            ? text.replace(opening, "").replace(/(?:^|\r?\n)[ \t]*```[ \t]*(?:\r?\n)*$/, "") : text;
    }
    const number = pageIndex + 1;

    function copy(node, parent, context) {
        if (node.nodeType === Node.TEXT_NODE) {
            parent.append(document.createTextNode(node.textContent));
            return;
        }
        if (node.nodeType !== Node.ELEMENT_NODE) return;
        const tag = node.localName;
        if (["script", "style", "iframe", "object", "embed", "svg", "math"].includes(tag)) return;
        if (!allowedTags.has(tag)) {
            for (const child of node.childNodes) copy(child, parent, context);
            return;
        }
        const safe = document.createElement(tag);
        const kind = Object.hasOwn(tagTypes, tag) ? tagTypes[tag] : context;
        if (tag === "td" || tag === "th") {
            for (const key of ["colspan", "rowspan"]) {
                const span = node.getAttribute(key);
                if (span && /^\d+$/.test(span)) safe.setAttribute(key, span);
            }
        }
        const coord = node.getAttribute("coord");
        if (coord && /^\d+,\d+,\d+,\d+,-?\d+$/.test(coord)) {
            const [x1, y1, x2, y2, angle] = coord.split(",").map(Number);
            if (x1 <= x2 && y1 <= y2 && x2 <= 1000 && y2 <= 1000 && Number.isSafeInteger(angle)) {
                const key = `${number}-${boxes.length + 1}`;
                const w = page.width / 1000, h = page.height / 1000;
                safe.dataset.segment = key;
                boxes.push(`<rect class="box page-${number % 2}${kind ? " tag-" + kind : ""}" data-segment="${key}" ` +
                    `x="${page.offsetX + x1 * w}" y="${y1 * h}" width="${(x2 - x1) * w}" ` +
                    `height="${(y2 - y1) * h}" vector-effect="non-scaling-stroke">` +
                    `<title>Fragment ${number}, segment ${boxes.length + 1}, angle ${angle}°</title></rect>`);
            }
        }
        for (const child of node.childNodes) copy(child, safe, kind);
        parent.append(safe);
    }
    for (const child of source.content.childNodes) copy(child, result, "");
    return {type: "page", pageIndex, fragment: result.innerHTML, boxes: boxes.join(""),
        segmentCount: boxes.length, finishReason: page.finishReason};
}

function portableEvents(file, display) {
    let pages, imageUrl, width, height;
    const dimension = value => {
        if (!Number.isSafeInteger(value) || value < 1) throw new Error("Dimensions du document invalides.");
        return value;
    };

    function viewer() {
        let total = 0;
        const fragments = [], boxes = [];
        pages.forEach((page, i) => {
            const view = portablePage(page, i);
            total += view.segmentCount;
            const reason = {stop: "stop", length: "length", error: "error"}[page.finishReason];
            fragments.push(`<section class="fragment" data-page-index="${i}"><h2>Fragment ${i + 1} · ` +
                `${view.segmentCount} segments <small>(${typeof reason === "string" ? reason : "en cours"})</small></h2>` +
                `<div class="fragment-body">${view.fragment}</div></section>`);
            boxes.push(`<g data-page-index="${i}">${view.boxes}</g>`);
        });
        const values = {width, height, image: imageUrl, fragments: fragments.join(""), boxes: boxes.join(""), total};
        return portableViewerShell.replace(/@@(width|height|image|fragments|boxes|total)@@/g,
            (_, key) => values[key]);
    }

    return async event => {
        if (event.type === "metadata") {
            width = dimension(event.imageWidth);
            height = dimension(event.imageHeight);
            pages = event.pages.map((page, i) => {
                if (page.pageIndex !== i || !Number.isSafeInteger(page.offsetX) || page.offsetX < 0)
                    throw new Error("Index ou position de page invalide.");
                return {width: dimension(page.width), height: dimension(page.height),
                    offsetX: page.offsetX, raw: "", lastUpdate: -Infinity};
            });
            imageUrl = await new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => resolve(reader.result);
                reader.onerror = () => reject(new Error("Impossible de lire l’image."));
                reader.readAsDataURL(file);
            });
            await display({type: "preview", html: viewer()});
        } else if (event.type === "complete") {
            if (!pages || pages.some(page => !page.finishReason)) throw new Error("Flux incomplet.");
            await display({type: "complete", html: viewer(), hasErrors: pages.some(p => p.finishReason === "error")});
        } else if (["delta", "done", "error"].includes(event.type)) {
            const i = event.pageIndex;
            if (!Number.isInteger(i) || !pages?.[i]) throw new Error("Fragment inconnu dans le flux.");
            const page = pages[i];
            if (event.type === "delta") {
                page.raw += event.html;
                if (performance.now() - page.lastUpdate < 150) return;
                page.lastUpdate = performance.now();
            } else {
                page.finishReason = event.type === "error" ? "error" : event.finishReason;
            }
            await display(portablePage(page, i));
        }
    };
}
