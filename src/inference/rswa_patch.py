"""Install or verify the audited Python-only RSWA patch for vLLM 0.34.0."""

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from tempfile import NamedTemporaryFile

VLLM_VERSION = "0.34.0"

# SHA256 of the source in the official Linux x86-64 wheel and v0.34.0 tag.
MODEL_PATH = "vllm/model_executor/models/qwen3_next.py"
TRITON_PATH = "vllm/v1/attention/ops/triton_unified_attention.py"
ORIGINAL_HASHES = {
    MODEL_PATH: "92ff95db994a56927008624de7e6fadcd0064efda6e11611e6057fb78af42884",
    TRITON_PATH: "4f376e7a028866d3bff3c28839d0f9ea2e09257d1bfb2e0e666e2fd623bc2ff7",
}
PATCHED_HASHES = {
    MODEL_PATH: "2ffc21ef858e6e8fdb0702129be45c7983e9f89541b0846e1321e2bcea7f8337",
    TRITON_PATH: "5512e7544d18c84bb4cacbebe5ec648265d889a118d56eb24c3b252dda456da8",
}

MODEL_EDITS = (
    (
        b"from vllm.model_executor.layers.attention import Attention\n",
        b"from vllm.model_executor.layers.attention import Attention\n"
        b"from vllm.model_executor.layers.attention.rswa_attention import RSWAAttention\n",
    ),
    (
        b"        self.attn = Attention(\n",
        b"""        rswa_window = model_config.rswa_window if model_config is not None else None
        attn_cls = Attention
        rswa_kwargs = {}
        if rswa_window is not None:
            if attn_type != AttentionType.DECODER or self.dual_chunk_attention_config:
                raise ValueError("RSWA requires causal decoder attention")
            attn_cls = RSWAAttention
            rswa_kwargs["rswa_window"] = rswa_window
        self.attn = attn_cls(
""",
    ),
    (
        b"            attn_type=attn_type,\n            **{\n",
        b"            attn_type=attn_type,\n            **rswa_kwargs,\n            **{\n",
    ),
)

TRITON_EDITS = (
    (
        b"            V = tl.where(sw_mask_v, V, 0.0)\n"
        b"        if USE_PER_TOKEN_HEAD_SCALES:\n",
        b"""            V = tl.where(sw_mask_v, V, 0.0)
        if USE_R_SWA:
            # Masked softmax weights do not protect the dot from 0 * NaN.
            visible_v = tl.sum(
                (seq_mask & query_mask_0[:, None] & query_mask_1[:, None]).to(tl.int32),
                axis=0,
            ) > 0
            V = tl.where(visible_v[:, None], V, 0.0)
        if USE_PER_TOKEN_HEAD_SCALES:
""",
    ),
)

EDITS = {MODEL_PATH: MODEL_EDITS, TRITON_PATH: TRITON_EDITS}


def digest(source):
    return hashlib.sha256(source).hexdigest()


def transform(path, source):
    for old, new in EDITS[path]:
        if source.count(old) != 1:
            raise RuntimeError(f"Expected exactly one RSWA patch anchor in {path}")
        source = source.replace(old, new, 1)
    compile(source, path, "exec")
    return source


def verify_or_apply(*, apply=False):
    """Refuse unknown source files, even if they have the expected wheel version."""
    installed = distribution("vllm")
    if installed.version != VLLM_VERSION:
        raise RuntimeError(f"RSWA requires vLLM {VLLM_VERSION}; found {installed.version}")

    changes = []
    for path, original_hash in ORIGINAL_HASHES.items():
        target = Path(installed.locate_file(path))
        source = target.read_bytes()
        source_hash = digest(source)
        if source_hash == PATCHED_HASHES[path]:
            continue
        if source_hash != original_hash:
            raise RuntimeError(f"Unexpected source SHA256 for {target}: {source_hash}")
        if not apply:
            raise RuntimeError(f"RSWA patch missing from {target}; run with --apply")
        patched = transform(path, source)
        if digest(patched) != PATCHED_HASHES[path]:
            raise RuntimeError(f"Unexpected patched SHA256 for {target}")
        changes.append((target, patched))

    # Preflight every file before replacing any of them.
    for target, patched in changes:
        with NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            temporary.write_bytes(patched)
            temporary.chmod(target.stat().st_mode & 0o777)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Patch the installed wheel")
    args = parser.parse_args()
    try:
        verify_or_apply(apply=args.apply)
    except (OSError, RuntimeError, PackageNotFoundError) as exc:
        parser.exit(1, f"RSWA: {exc}\n")
    print(f"RSWA patch verified for vLLM {VLLM_VERSION}")


if __name__ == "__main__":
    main()
