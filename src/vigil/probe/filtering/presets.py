"""Portable, versioned JSON presets for the object-pose filters."""

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from ...config import PROJECT_ROOT
from .pose import FILTER_METHODS, FilterTuning


PRESET_SCHEMA_VERSION = 1
MAX_PRESET_BYTES = 64 * 1024
DEFAULT_PRESET_PATH = PROJECT_ROOT / "configs" / "probe-filter.json"


def encode_filter_preset(method: str, tuning: FilterTuning) -> bytes:
    if method not in FILTER_METHODS:
        raise ValueError(f"unknown pose filter: {method}")
    document = {
        "schema_version": PRESET_SCHEMA_VERSION,
        "filter_method": method,
        "tuning": tuning.to_dict(),
    }
    return (json.dumps(document, indent=2, allow_nan=False) + "\n").encode("utf-8")


def decode_filter_preset(contents: bytes | str) -> tuple[str, FilterTuning]:
    """Validate the entire preset before any live settings are changed."""
    if not isinstance(contents, (bytes, str)):
        raise ValueError("preset must be UTF-8 JSON")
    if len(contents) > MAX_PRESET_BYTES:
        raise ValueError("preset is too large (maximum 64 KiB)")
    try:
        text = contents.decode("utf-8-sig") if isinstance(contents, bytes) else contents
        document = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("preset must be valid UTF-8 JSON") from error
    if not isinstance(document, dict) or set(document) != {
        "schema_version", "filter_method", "tuning"
    }:
        raise ValueError("preset must contain schema_version, filter_method and tuning")
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != PRESET_SCHEMA_VERSION
    ):
        raise ValueError("unsupported preset schema version")
    method = document["filter_method"]
    if not isinstance(method, str) or method not in FILTER_METHODS:
        raise ValueError("preset contains an unknown pose filter")
    return method, FilterTuning.from_dict(document["tuning"])


def load_filter_preset_file(
    path: Path = DEFAULT_PRESET_PATH,
) -> tuple[str, FilterTuning]:
    """Read and validate a preset file without changing a live estimator."""
    with path.open("rb") as preset_file:
        contents = preset_file.read(MAX_PRESET_BYTES + 1)
    return decode_filter_preset(contents)


def save_filter_preset_file(
    method: str,
    tuning: FilterTuning,
    path: Path = DEFAULT_PRESET_PATH,
) -> Path:
    """Atomically replace a preset file with validated current settings."""
    contents = encode_filter_preset(method, tuning)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_file.write(contents)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            temporary_path = Path(temporary_file.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return path
