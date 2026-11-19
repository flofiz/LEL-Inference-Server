"""Token-based scoring of streamed transcription lines."""


def token_text_spans(text, selected, special_ids):
    """Map sampled logprobs to emitted text without changing the text itself."""
    tokens = [prob for token, prob in selected if token not in special_ids]
    spans = [(p.decoded_token, p.logprob, 1.0) for p in tokens if p.decoded_token]
    if "".join(piece for piece, _, _ in spans) == text:
        return spans
    # The incremental detokenizer may withhold or normalize a token's text.
    # Preserve the emitted text and use a chunk average if alignment fails.
    count = len(tokens)
    average = sum(token.logprob for token in tokens) / count if count else 0.0
    return [(text, average, count)] if text else []


class ScoredLines:
    def __init__(self):
        self.spans = []

    def push(self, text, logprob, weight):
        length = len(text)
        while text:
            head, newline, text = text.partition("\n")
            piece = head + newline
            self.spans.append((piece, logprob, weight * len(piece) / length))
            if newline:
                spans, self.spans = self.spans, []
                yield spans


def score_range(spans, start=0, end=None):
    """Return logprob sum and token weight for a character range in a line."""
    total = weight = 0.0
    offset = 0
    for piece, logprob, token_weight in spans:
        stop = offset + len(piece)
        overlap = (stop if end is None else min(stop, end)) - max(offset, start)
        if overlap > 0:
            fraction = token_weight * overlap / len(piece)
            total += logprob * fraction
            weight += fraction
        offset = stop
    return total, weight
