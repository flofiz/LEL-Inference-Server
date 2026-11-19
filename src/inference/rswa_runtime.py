"""Attest actual vLLM allocations and forbid unmasked decoding after startup."""

from functools import wraps

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.rswa_attention import RSWAAttention
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import RSWAManager
from vllm.v1.kv_cache_interface import MambaSpec, RSWASpec

from inference.rswa_policy import DRAFT, TARGET, assert_config

# EngineCore and GPU workers configure the vllm logger, not the root logger.
logger = init_logger("vllm.rswa")


def _observe_first_eviction(manager):
    remove = manager._remove_blocks_in_range

    def observe(request_id, first_block, last_block):
        blocks = manager.req_to_blocks.get(request_id, ())
        last = min(last_block, len(blocks)) - 1
        was_live = last >= first_block and blocks[last] != manager._null_block
        remove(request_id, first_block, last_block)
        if was_live and blocks[last] == manager._null_block:
            logger.info("RSWA first KV eviction: logical block %d, block size %d",
                        last, manager.block_size)
            del manager._remove_blocks_in_range

    manager._remove_blocks_in_range = observe


class RSWAScheduler(Scheduler):
    def __init__(self, vllm_config, *args, **kwargs):
        window = vllm_config.model_config.rswa_window
        assert_config(vllm_config, window)
        super().__init__(vllm_config, *args, **kwargs)
        managers = self.kv_cache_manager.coordinator.single_type_managers
        rswa = [m for m in managers if isinstance(m, RSWAManager)]
        gdn = [m for m in managers if isinstance(m.kv_cache_spec, MambaSpec)]
        if (not rswa or not gdn or len(rswa) + len(gdn) != len(managers)
                or any(m.rswa_window != window for m in rswa)):
            raise RuntimeError("RSWA scheduler did not allocate the expected RSWA/GDN groups")
        _observe_first_eviction(rswa[0])
        logger.info("RSWA scheduler ready: window=%d attention_groups=%d gdn_groups=%d",
                    window, len(rswa), len(gdn))


def _guard_forward(forward, window, role=None):
    @wraps(forward)
    def guarded(layer, query, key, value, kv_cache, attn_metadata, *args, **kwargs):
        nonlocal role
        if (attn_metadata is None or attn_metadata.rswa_window != window
                or attn_metadata.rswa_prefix_lens is None
                or attn_metadata.rswa_prefix_lens.shape != attn_metadata.seq_lens.shape
                or attn_metadata.causal is not True
                or attn_metadata.mm_prefix_range_tensor is not None):
            raise RuntimeError("Missing or incompatible RSWA metadata during Triton forward")
        output = forward(layer, query, key, value, kv_cache, attn_metadata, *args, **kwargs)
        if role is not None:
            logger.info("RSWA %s ran with causal full-prompt metadata", role)
            role = None
        return output

    return guarded


class RSWAWorkerExtension:
    def rswa_inspect(self, window):
        config = self.vllm_config
        assert_config(config, window)
        target, draft = self.get_model(), self.get_draft_model()
        if (type(target).__name__, type(draft).__name__) != (TARGET, DRAFT):
            raise RuntimeError("RSWA worker loaded an unexpected target or draft model")

        checked = {}
        for role, model in (("target", target.language_model), ("draft", draft)):
            layers = [m for m in model.modules() if isinstance(m, Attention)]
            if not layers or (role == "draft" and len(layers) != 1):
                raise RuntimeError(f"Unexpected RSWA attention layer count in {role}")
            for layer in layers:
                spec = layer.get_kv_cache_spec(config)
                if (not isinstance(layer, RSWAAttention)
                        or not isinstance(spec, RSWASpec) or spec.rswa_window != window
                        or not isinstance(layer.impl, TritonAttentionImpl)
                        or layer.kv_cache.numel() == 0 or layer.kv_cache.dtype != torch.bfloat16):
                    raise RuntimeError(f"RSWA {role} attention is not backed by a BF16 Triton RSWA cache")
            checked[role] = layers

        gdn = [m for m in target.language_model.modules()
               if isinstance(m, QwenGatedDeltaNetAttention)]
        mtp_tokens = config.speculative_config.num_speculative_tokens
        if mtp_tokens == 3 and self.model_runner.speculator.supports_decode_metadata_update:
            raise RuntimeError("RSWA MTP3 requires fresh attention metadata per draft step")
        if not gdn or any(
            not isinstance(spec := layer.get_kv_cache_spec(config), MambaSpec)
            or spec.dtypes[-1] != torch.float32 or spec.num_speculative_blocks != mtp_tokens
            for layer in gdn
        ):
            raise RuntimeError(f"RSWA requires GDN FP32 recurrent state with native MTP K={mtp_tokens}")

        for role, layers in checked.items():
            for index, layer in enumerate(layers):
                layer.impl.forward = _guard_forward(
                    layer.impl.forward, window, role if index == 0 else None
                )
        return {"rank": self.rank, "window": window, "target": len(checked["target"]),
                "draft": len(checked["draft"]), "gdn": len(gdn), "runtime_enforced": True,
                "mtp_tokens": mtp_tokens,
                "block_size": max(layer.get_kv_cache_spec(config).block_size
                                  for layer in checked["target"])}
