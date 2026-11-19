from pathlib import Path

import pytest


@pytest.fixture
def config(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "src"))
    from server_config import ROOT, load_engine_args

    return ROOT, load_engine_args


def test_server_toml_defaults_and_local_overrides(config, tmp_path):
    root, load = config
    (tmp_path / "server.toml").write_bytes((root / "server.toml").read_bytes())
    defaults = load(tmp_path, {})
    assert defaults["model"].endswith("/Qwen3.5-4B-c400-wsp-hybrid-60-50")
    assert defaults["mtp_tokens"] == 3 and defaults["require_rswa"] is True
    assert defaults["max_model_len"] == 24576

    (tmp_path / "server.local.toml").write_text('[engine]\nmodel = "/new/model"\nmtp_tokens = 1\n')
    local = load(tmp_path, {})
    assert local["model"] == "/new/model" and local["mtp_tokens"] == 1
    assert local["limit_mm_per_prompt"] == {"image": 1}
    env = load(tmp_path, {"HTR_MODEL_PATH": "/systemd/override", "HTR_MTP_TOKENS": "3"})
    assert env["model"] == "/systemd/override" and env["mtp_tokens"] == 3

    with pytest.raises(FileNotFoundError):
        load(tmp_path, {"HTR_CONFIG_PATH": str(tmp_path / "missing.toml")})
