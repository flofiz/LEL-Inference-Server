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
PRUNED_TRITON_HASH = "66471c41551c4b577014017b4d566237f254bbb1784f18094fc5454e93186c7c"

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

PRUNE_HELPER = b"""@triton.jit
def _rswa_pruned_loop(loop_lo, loop_hi, prefix_len, q_lo, q_end, window,
                      TILE_SIZE: tl.constexpr):
    # Keep boundary tiles: the gap contains keys in [prefix_len, q_lo-window+1).
    # Keys >= q_end are in the causal future for the entire query tile.
    loop_hi = tl.minimum(loop_hi, cdiv_fn(q_end, TILE_SIZE))
    gap_first = tl.maximum(loop_lo, cdiv_fn(prefix_len, TILE_SIZE))
    gap_end = tl.minimum(loop_hi, tl.maximum(0, q_lo - window + 1) // TILE_SIZE)
    skipped = tl.maximum(0, gap_end - gap_first)
    return loop_hi - skipped, gap_first, skipped


"""
KERNEL_ANCHOR = b"@triton.jit\ndef kernel_unified_attention(\n"
PRUNED_EDITS = (
    (KERNEL_ANCHOR, PRUNE_HELPER + KERNEL_ANCHOR),
    (
        b"    # iterate through tiles (now limited to the sliding window range)\n"
        b"    for j in range(loop_lo, loop_hi):\n",
        b"""    PRUNE_RSWA: tl.constexpr = (
        USE_R_SWA and USE_CAUSAL and not USE_PER_SEQ_CAUSAL and not USE_MM_PREFIX
        and not USE_QQ_BIAS and SLIDING_WINDOW == 0 and CHUNK_LOOKBACK < 0
    )
    if PRUNE_RSWA:
        prefix_len = tl.load(rswa_prefix_lens_ptr + seq_idx)
        q_lo = context_len + q_block_local_idx * BLOCK_Q
        loop_hi, gap_first, skipped = _rswa_pruned_loop(
            loop_lo, loop_hi, prefix_len, q_lo, q_lo + q_block_local_len,
            R_SWA_WINDOW, TILE_SIZE,
        )

    # Iterate in logical KV order; remap loop indices over the invisible gap.
    # Empty split-K segments still write their neutral softmax state below.
    for loop_j in range(loop_lo, loop_hi):
        j = loop_j
        if PRUNE_RSWA:
            j += tl.where(loop_j >= gap_first, skipped, 0)
""",
    ),
)

SPEC_PATH = "vllm/v1/worker/gpu/spec_decode/speculator.py"
AR_SPEC_PATH = "vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py"
MTP_ORIGINAL_HASHES = {
    SPEC_PATH: "3087a6b61c8ac534e06175903d7c4d94a61d52e90691894d7d89c996f8fbc8c7",
    AR_SPEC_PATH: "fb722001fd6f70c0202d4eacc6660a4fb01b265a45a286e4ed1c07c6dd623fa3",
}
MTP_PATCHED_HASHES = {
    SPEC_PATH: "f0c14a3a5df0772a70e7684f6ad7bda608f732612e4268262f221e625469bf50",
    AR_SPEC_PATH: "2c49b9712ef5a8c86f85f6edc5d60b2a0ba1bc1a8c76565b009e68e98ac737b1",
}
MTP_EDITS = {
    SPEC_PATH: (
        (
            b"        slot_mappings: torch.Tensor | None = None,\n    ) -> dict[str, Any] | None:\n",
            b"        slot_mappings: torch.Tensor | None = None,\n"
            b"        rswa_prefix_lens: torch.Tensor | None = None,\n    ) -> dict[str, Any] | None:\n",
        ),
        (
            b"            is_prefilling=self.draft_is_prefilling[:num_reqs_padded],\n",
            b"            is_prefilling=self.draft_is_prefilling[:num_reqs_padded],\n"
            b"            rswa_prefix_lens=rswa_prefix_lens,\n",
        ),
        (
            b"    ) -> dict[str, Any] | None:\n        query_start_loc_np = self.arange_np",
            b"        rswa_prefix_lens: torch.Tensor | None = None,\n"
            b"    ) -> dict[str, Any] | None:\n        query_start_loc_np = self.arange_np",
        ),
        (
            b"        )\n\n    def _update_draft_decode_metadata(\n",
            b"            rswa_prefix_lens=rswa_prefix_lens,\n"
            b"        )\n\n    def _update_draft_decode_metadata(\n",
        ),
    ),
    AR_SPEC_PATH: (
        (
            b"            num_speculative_steps,\n        )\n        self.on_multi_step_decode_end(num_reqs)\n",
            b"            num_speculative_steps,\n"
            b"            rswa_prefix_lens=(input_batch.prompt_lens[:num_reqs]\n"
            b"                              if input_batch.prompt_lens is not None else None),\n"
            b"        )\n        self.on_multi_step_decode_end(num_reqs)\n",
        ),
        (
            b"    ) -> None:\n        positions = self.input_buffers.positions[:num_reqs]\n"
            b"        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]\n"
            b"        idx_mapping = self.idx_mapping[:num_reqs]\n",
            b"        rswa_prefix_lens: torch.Tensor | None = None,\n"
            b"    ) -> None:\n        positions = self.input_buffers.positions[:num_reqs]\n"
            b"        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]\n"
            b"        idx_mapping = self.idx_mapping[:num_reqs]\n",
        ),
        (
            b"                    step=step,\n                )\n",
            b"                    step=step,\n                    rswa_prefix_lens=rswa_prefix_lens,\n                )\n",
        ),
        (
            b"    ) -> None:\n        if batch_desc.cg_mode == CUDAGraphMode.FULL:\n"
            b"            self._map_kernel_block_tables()\n",
            b"        rswa_prefix_lens: torch.Tensor | None = None,\n"
            b"    ) -> None:\n        if batch_desc.cg_mode == CUDAGraphMode.FULL:\n"
            b"            self._map_kernel_block_tables()\n",
        ),
        (
            b"                step=1,\n            )\n\n        self._fused_multi_step_decode(",
            b"                step=1,\n                rswa_prefix_lens=rswa_prefix_lens,\n"
            b"            )\n\n        self._fused_multi_step_decode(",
        ),
    ),
}


def digest(source):
    return hashlib.sha256(source).hexdigest()


def transform(path, source, edits=None):
    for old, new in (EDITS | MTP_EDITS)[path] if edits is None else edits:
        if source.count(old) != 1:
            raise RuntimeError(f"Expected exactly one RSWA patch anchor in {path}")
        source = source.replace(old, new, 1)
    compile(source, path, "exec")
    return source


def verify_or_apply(*, apply=False, variant=None, mtp3=False):
    """Refuse unknown source files, even if they have the expected wheel version."""
    if variant is not None and (not apply or variant not in ("baseline", "pruned")):
        raise ValueError("A patch variant requires apply=True and must be baseline or pruned")
    installed = distribution("vllm")
    if installed.version != VLLM_VERSION:
        raise RuntimeError(f"RSWA requires vLLM {VLLM_VERSION}; found {installed.version}")

    changes = []
    current_variant = "baseline"
    originals = ORIGINAL_HASHES | (MTP_ORIGINAL_HASHES if mtp3 else {})
    expected_baseline = PATCHED_HASHES | (MTP_PATCHED_HASHES if mtp3 else {})
    for path, original_hash in originals.items():
        target = Path(installed.locate_file(path))
        source = target.read_bytes()
        source_hash = digest(source)
        is_original = source_hash == original_hash
        is_pruned = path == TRITON_PATH and source_hash == PRUNED_TRITON_HASH
        if not is_original and not is_pruned and source_hash != expected_baseline[path]:
            raise RuntimeError(f"Unexpected source SHA256 for {target}: {source_hash}")
        if is_original and not apply:
            option = "--apply --mtp3" if path in MTP_EDITS else "--apply"
            raise RuntimeError(f"RSWA patch missing from {target}; run with {option}")

        patched = transform(path, source) if is_original else source
        if path == TRITON_PATH:
            current_variant = "pruned" if is_pruned else "baseline"
            if variant is not None and variant != current_variant:
                edits = PRUNED_EDITS if variant == "pruned" else tuple(
                    (new, old) for old, new in reversed(PRUNED_EDITS)
                )
                patched = transform(path, patched, edits)
                current_variant = variant
        expected = (
            PRUNED_TRITON_HASH if path == TRITON_PATH and current_variant == "pruned"
            else expected_baseline[path]
        )
        if digest(patched) != expected:
            raise RuntimeError(f"Unexpected patched SHA256 for {target}")
        if patched != source:
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
    return current_variant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Patch the installed wheel")
    parser.add_argument("--variant", choices=("baseline", "pruned"), help="Select a patch variant with --apply")
    parser.add_argument("--mtp3", action="store_true", help="Include the sequential MTP3 metadata patch")
    args = parser.parse_args()
    if args.variant is not None and not args.apply:
        parser.error("--variant requires --apply")
    try:
        variant = verify_or_apply(apply=args.apply, variant=args.variant, mtp3=args.mtp3)
    except (OSError, RuntimeError, PackageNotFoundError) as exc:
        parser.exit(1, f"RSWA: {exc}\n")
    print(f"RSWA {variant} patch verified for vLLM {VLLM_VERSION}" + (" (with MTP3)" if args.mtp3 else ""))


if __name__ == "__main__":
    main()
