"""YAML configuration with single-level inheritance (`base:`) and dotted command-line overrides."""

import copy
from pathlib import Path

import yaml


def _merge(base, override):
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _set_dotted(cfg, dotted, value):
    keys = dotted.split(".")
    node = cfg
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def load_config(path, overrides=()):
    """Load a config file. `overrides` are strings such as "train.max_epochs=2"."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = cfg.pop("base", None)
    if base:
        cfg = _merge(load_config(path.parent / base), cfg)
    for item in overrides:
        key, _, raw = item.partition("=")
        _set_dotted(cfg, key.strip(), yaml.safe_load(raw))
    return cfg
