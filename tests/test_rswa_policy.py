"""CPU tests of the RSWA checkpoint and engine configuration contract."""

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def rswa_policy(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "src"))
    from inference import rswa_policy

    return rswa_policy


@pytest.fixture
def export(tmp_path):
    meta = {
        "allow_rswa": True,
        "rswa_window": 512,
        "policy": "causal_prefix_response_window",
        "version": 1,
        "backend_train": "flex",
        "native_vllm_autoload": False,
    }
    config = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "rswa_training": meta.copy(),
        "text_config": {"rswa_training": meta.copy(), "mtp_num_hidden_layers": 1},
    }
    manifest = {**meta, "serving_requirement": "RSWA cache manager + attention"}
    return tmp_path, config, manifest


def write_export(directory, config, manifest):
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (directory / "rswa_training.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_standard_checkpoint_does_not_enable_rswa(rswa_policy, tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"model_type": "qwen3_5"}', encoding="utf-8")
    assert rswa_policy.discover(tmp_path) is None
    path.write_text('{"text_config": {"rswa_window": 512}}', encoding="utf-8")
    with pytest.raises(ValueError, match="unnamespaced"):
        rswa_policy.discover(tmp_path)


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_mtp"])
def test_rswa_override_only_in_memory(rswa_policy, export, model_type):
    directory, config, manifest = export
    write_export(*export)
    path = directory / "config.json"
    original = path.read_bytes()
    policy = rswa_policy.discover(directory)
    options = rswa_policy.engine_options(policy, {"model": str(directory)})
    override = pickle.loads(pickle.dumps(options["hf_overrides"]))
    architecture = rswa_policy.TARGET if model_type == "qwen3_5" else rswa_policy.DRAFT
    hf = SimpleNamespace(
        **{**config, "model_type": model_type, "architectures": [architecture],
           "text_config": SimpleNamespace(**config["text_config"])},
    )
    dummy = SimpleNamespace(model_type="dummy_qwen3_5", architectures=[""])
    assert override(dummy) is dummy
    assert not hasattr(dummy, "rswa_window")
    assert override(hf) is hf
    assert hf.architectures == [architecture]
    assert hf.rswa_window == hf.text_config.rswa_window == 512
    assert options["speculative_config"]["num_speculative_tokens"] == 1
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "invalid",
    ["missing-manifest", "missing-text", "autoload", "drift", "bool-window", "quantized"],
)
def test_invalid_export_is_rejected(rswa_policy, export, invalid):
    directory, config, manifest = export
    if invalid == "missing-text":
        del config["text_config"]["rswa_training"]
    elif invalid == "autoload":
        manifest["native_vllm_autoload"] = True
    elif invalid == "drift":
        config["text_config"]["rswa_training"]["version"] = True
    elif invalid == "bool-window":
        manifest["rswa_window"] = True
    elif invalid == "quantized":
        config["quantization_config"] = {"quant_method": "compressed-tensors"}
    write_export(*export)
    if invalid == "missing-manifest":
        (directory / "rswa_training.json").unlink()
    with pytest.raises(ValueError, match="RSWA"):
        rswa_policy.discover(directory)


@pytest.mark.parametrize(
    ("key", "value"),
    [("enable_prefix_caching", True), ("tensor_parallel_size", 2),
     ("speculative_config", {"method": "mtp", "num_speculative_tokens": 2}),
     ("hf_overrides", {"rswa_window": 32})],
)
def test_rswa_disallows_overriding_reserved_options(rswa_policy, export, key, value):
    write_export(*export)
    policy = rswa_policy.discover(export[0])
    with pytest.raises(ValueError, match=key):
        rswa_policy.engine_options(policy, {key: value})


def test_rswa_mtp3_is_opt_in_and_sequential(rswa_policy, export):
    write_export(*export)
    policy = rswa_policy.discover(export[0])
    assert rswa_policy.engine_options(policy, {}, mtp_tokens=3)["speculative_config"] == {
        "method": "mtp", "num_speculative_tokens": 3, "attention_backend": "TRITON_ATTN",
    }
    with pytest.raises(ValueError, match="K=1 or K=3"):
        rswa_policy.engine_options(policy, {}, mtp_tokens=2)
