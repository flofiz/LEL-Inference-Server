"""Read versioned engine defaults and optional host-specific overrides."""

import os
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def _read_engine(path):
    with path.open("rb") as file:
        engine = tomllib.load(file).get("engine")
    if not isinstance(engine, dict):
        raise ValueError(f"Missing [engine] table in {path}")
    return engine


def load_engine_args(root=ROOT, environ=None):
    env = os.environ if environ is None else environ
    args = _read_engine(root / "server.toml")
    local = env.get("HTR_CONFIG_PATH")
    override = Path(local).expanduser() if local else root / "server.local.toml"
    if local or override.exists():
        args.update(_read_engine(override))

    if "HTR_MODEL_PATH" in env:
        args["model"] = env["HTR_MODEL_PATH"]
    if "HTR_REQUIRE_RSWA" in env:
        args["require_rswa"] = env["HTR_REQUIRE_RSWA"] != "0"
    if "HTR_MTP_TOKENS" in env:
        args["mtp_tokens"] = int(env["HTR_MTP_TOKENS"])
    if "HTR_ENABLE_THINKING" in env:
        args["enable_thinking"] = env["HTR_ENABLE_THINKING"] == "1"

    if not isinstance(args.get("model"), str) or not args["model"].strip():
        raise ValueError("The [engine] model must be a checkpoint path or model name")
    for flag in ("require_rswa", "enable_thinking"):
        if type(args.get(flag)) is not bool:
            raise ValueError(f"[engine] {flag} must be a TOML boolean")
    if type(args.get("mtp_tokens")) is not int or args["mtp_tokens"] not in (1, 3):
        raise ValueError("[engine] mtp_tokens must be 1 or 3")
    args.setdefault("standard_mtp", "Qwen3.5" in args["model"])
    return args
