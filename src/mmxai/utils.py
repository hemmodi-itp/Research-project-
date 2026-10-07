"""Small shared helpers: seeding, devices, modality subsets, JSON I/O."""

import json
import logging
import random
from itertools import combinations
from pathlib import Path

import numpy as np

from . import MODALITIES, MODALITY_LETTERS

# Every non-empty modality subset (the seven-arm ablation, Sec. 4.2).
ALL_SUBSETS = [s for r in (1, 2, 3) for s in combinations(MODALITIES, r)]
FULL = MODALITIES

# Baseline models of Table 4. The ablation adds the remaining subset (I+T).
BASELINES = {
    "A": ("tab",),
    "B": ("image",),
    "C": ("text",),
    "D": ("image", "tab"),
    "E": ("text", "tab"),
    "F": ("image", "text", "tab"),
}


def subset_key(subset):
    """("image", "tab") -> "I+S", always in canonical order."""
    return "+".join(MODALITY_LETTERS[m] for m in MODALITIES if m in subset)


def parse_subset_key(key):
    return tuple(m for m in MODALITIES if MODALITY_LETTERS[m] in key.split("+")) if key else ()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(name="auto"):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def get_logger(name="mmxai", log_file=None):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_file and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)
    return logger
