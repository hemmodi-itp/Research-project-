import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmxai.config import load_config  # noqa: E402
from mmxai.utils import ensure_dir, get_logger  # noqa: E402


def parse(description, extra=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config values, e.g. --set train.max_epochs=2 seeds=[0]")
    if extra:
        extra(parser)
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    out = ensure_dir(cfg["output_dir"])
    log = get_logger(log_file=out / "log.txt")
    return args, cfg, log
