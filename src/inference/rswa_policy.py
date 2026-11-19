"""Fail-closed contract for the BF16 Qwen3.5 RSWA export."""

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path

TARGET = "Qwen3_5ForConditionalGeneration"
DRAFT = "Qwen3_5MTP"
BACKEND = "TRITON_ATTN"
REQUIRED_METADATA = {
    "allow_rswa": True,
    "version": 1,
    "policy": "causal_prefix_response_window",
    "native_vllm_autoload": False,
}


@dataclass(frozen=True)
class Policy:
    window: int
    metadata: dict


def _json_object(path):
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _same_metadata(value, expected):
    # Unlike Python dict equality, this distinguishes bool, int and float.
    return json.dumps(value, sort_keys=True) == json.dumps(expected, sort_keys=True)


def discover(model):
    """Return None for ordinary checkpoints, never for an incomplete RSWA export."""
    directory = Path(model).resolve()
    config_path = directory / "config.json"
    manifest_path = directory / "rswa_training.json"
    has_manifest = manifest_path.exists() or manifest_path.is_symlink()
    if not config_path.is_file():
        if has_manifest:
            raise ValueError("RSWA checkpoint config.json missing")
        return None
    config = _json_object(config_path)
    text = config.get("text_config", {})
    if not isinstance(text, dict):
        raise ValueError("Checkpoint text_config must be an object")
    if "rswa_window" in config or "rswa_window" in text:
        raise ValueError("Checkpoint must not set an unnamespaced rswa_window")
    if not has_manifest and "rswa_training" not in config and "rswa_training" not in text:
        return None
    if not manifest_path.is_file() or "rswa_training" not in config or "rswa_training" not in text:
        raise ValueError("Incomplete RSWA export: require manifest and root/text metadata")

    metadata = _json_object(manifest_path)
    metadata.pop("serving_requirement", None)
    for key, expected in REQUIRED_METADATA.items():
        if type(metadata.get(key)) is not type(expected) or metadata[key] != expected:
            raise ValueError(f"Invalid RSWA manifest field: {key}")
    window = metadata.get("rswa_window")
    if type(window) is not int or window < 1:
        raise ValueError("RSWA window must be a positive integer")
    if metadata.get("backend_train") not in ("flex", "sdpa_dense"):
        raise ValueError("Unknown RSWA training backend")
    if metadata.keys() != REQUIRED_METADATA.keys() | {"rswa_window", "backend_train"}:
        raise ValueError("Unknown RSWA manifest fields")
    if any(not _same_metadata(c["rswa_training"], metadata) for c in (config, text)):
        raise ValueError("RSWA manifest differs from config root/text metadata")
    if config.get("model_type") != "qwen3_5" or config.get("architectures") != [TARGET]:
        raise ValueError("RSWA requires native dense multimodal Qwen3.5")
    if type(text.get("mtp_num_hidden_layers")) is not int or text["mtp_num_hidden_layers"] != 1:
        raise ValueError("RSWA requires exactly one native MTP layer")
    if any(c.get("quantization_config") is not None for c in (config, text)):
        raise ValueError("This RSWA integration supports BF16 weights only")
    return Policy(window, metadata)


def hf_override(policy, config):
    model_type = config.model_type
    # vLLM probes the callable on a dummy config before parsing the checkpoint.
    if model_type == "dummy_qwen3_5" and config.architectures == [""]:
        return config
    expected = {"qwen3_5": TARGET, "qwen3_5_mtp": DRAFT}.get(model_type)
    if expected is None or config.architectures != [expected]:
        raise ValueError(f"Unexpected RSWA model type/architecture: {model_type}/{config.architectures}")
    for cfg in (config, config.text_config):
        if not _same_metadata(getattr(cfg, "rswa_training", None), policy.metadata):
            raise ValueError("Loaded HF config differs from RSWA manifest")
        cfg.rswa_window = policy.window
    return config


def engine_options(policy, kwargs):
    options = {
        "attention_config": {"backend": BACKEND},
        "speculative_config": {
            "method": "mtp", "num_speculative_tokens": 1, "attention_backend": BACKEND,
        },
        "enable_prefix_caching": False,
        "async_scheduling": False,
        "enforce_eager": True,
        "mm_encoder_attn_backend": "TORCH_SDPA",
        "mm_processor_cache_gb": 0,
        "dtype": "bfloat16",
        "kv_cache_dtype": "auto",
        "mamba_ssm_cache_dtype": "float32",
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "distributed_executor_backend": "mp",
        "scheduler_cls": "inference.rswa_runtime.RSWAScheduler",
        "worker_extension_cls": "inference.rswa_runtime.RSWAWorkerExtension",
    }
    if "hf_overrides" in kwargs:
        raise ValueError("RSWA reserves hf_overrides")
    for key, value in options.items():
        if key in kwargs and kwargs[key] != value:
            raise ValueError(f"RSWA requires {key}={value!r}, got {kwargs[key]!r}")
    return {**kwargs, **options, "hf_overrides": partial(hf_override, policy)}


def assert_config(config, window):
    model = config.model_config
    spec = config.speculative_config
    if (type(window) is not int or window < 1
            or model.architectures != [TARGET] or model.rswa_window != window):
        raise RuntimeError("RSWA target architecture/window mismatch")
    if (spec is None or spec.method != "mtp" or spec.num_speculative_tokens != 1
            or spec.parallel_drafting or spec.use_multi_module_mtp()):
        raise RuntimeError("RSWA requires fixed native sequential MTP K=1")
    draft = spec.draft_model_config
    if draft.architectures != [DRAFT] or draft.rswa_window != window:
        raise RuntimeError("RSWA draft architecture/window mismatch")
    if model.hf_text_config.mtp_num_hidden_layers != 1:
        raise RuntimeError("RSWA requires exactly one native MTP layer")
    if any(getattr(b, "name", b) != BACKEND for b in (config.attention_config.backend, spec.attention_backend)):
        raise RuntimeError("RSWA requires Triton attention in both target and draft")
    if any(str(m.dtype) != "torch.bfloat16" or m.quantization is not None for m in (model, draft)):
        raise RuntimeError("RSWA supports only unquantized BF16 target and draft")
    p = config.parallel_config
    if (p.tensor_parallel_size, p.pipeline_parallel_size, p.data_parallel_size,
        p.prefill_context_parallel_size, p.decode_context_parallel_size) != (1, 1, 1, 1, 1):
        raise RuntimeError("RSWA only supports single-GPU inference")
    if p.distributed_executor_backend != "mp":
        raise RuntimeError("RSWA requires the local mp executor")
    if (not model.enforce_eager or config.scheduler_config.async_scheduling
            or config.cache_config.enable_prefix_caching
            or config.cache_config.cache_dtype not in ("auto", "bfloat16")
            or config.cache_config.mamba_ssm_cache_dtype != "float32"):
        raise RuntimeError("RSWA requires eager synchronous execution, BF16 KV, FP32 recurrent state and no prefix cache")
    mm = model.multimodal_config
    if (mm is None or getattr(mm.mm_encoder_attn_backend, "name", mm.mm_encoder_attn_backend) != "TORCH_SDPA"
            or mm.mm_processor_cache_gb != 0):
        raise RuntimeError("RSWA requires a TORCH_SDPA vision encoder and no multimodal processor cache")
