"""Small config reader for migration-time key/value YAML files."""

from __future__ import annotations

from pathlib import Path


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def read_key_value_config(path: str | Path) -> dict[str, str]:
    """Read the flat `key: value` config files used by the migration CLIs."""
    return _read_key_value_config(Path(path), seen=set())


def _read_key_value_config(config_path: Path, *, seen: set[Path]) -> dict[str, str]:
    """Read a config and its optional single-parent ``include``."""
    config_path = config_path.resolve()
    if config_path in seen:
        chain = " -> ".join(str(item) for item in (*seen, config_path))
        raise ValueError(f"config include cycle: {chain}")
    seen = {*seen, config_path}
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")

    values: dict[str, str] = {}
    for lineno, raw_line in enumerate(config_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"{config_path}:{lineno}: expected 'key: value'")
        key, value = line.split(":", 1)
        values[key.strip()] = _strip_quotes(value.strip())
    include = values.pop("include", None)
    if not include:
        return values
    parent_path = Path(include)
    if not parent_path.is_absolute():
        parent_path = config_path.parent / parent_path
    parent = _read_key_value_config(parent_path, seen=seen)
    return {**parent, **values}


def prefixed_values(config: dict[str, str], prefix: str) -> dict[str, str]:
    """Return entries whose key starts with `prefix`, removing the prefix."""
    return {key[len(prefix) :]: value for key, value in config.items() if key.startswith(prefix)}
