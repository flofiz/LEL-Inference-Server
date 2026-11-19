"""HTML output stays a fragment; do not parse, rewrite or render model HTML."""

import re

HTML_MAX_TOKENS = 16_384
HTML_CONTEXT_LENGTH = 24_576

_OPEN_FENCE = re.compile(r"\A\s*```html[ \t]*\r?\n", re.IGNORECASE)
_CLOSE_FENCE = re.compile(r"(?:\A|\r?\n)[ \t]*```[ \t]*(?:\r?\n)*\Z")


def unwrap_html_fragment(text):
    opening = _OPEN_FENCE.match(text)
    if opening is None:
        return text
    return _CLOSE_FENCE.sub("", text[opening.end():])
