"""Train all models (7 modality subsets + XGBoost) for every seed and store their predictions.

Finished runs are skipped, so the script can be re-run after an interruption.
"""

from _common import parse

from mmxai.experiments import run_all
from mmxai.utils import parse_subset_key


def extra(p):
    p.add_argument("--subsets", nargs="*", help="limit to subsets such as I+T+S T+S")
    p.add_argument("--seeds", nargs="*", type=int)


if __name__ == "__main__":
    args, cfg, log = parse(__doc__, extra)
    subsets = [parse_subset_key(s) for s in args.subsets] if args.subsets else None
    run_all(cfg, log, subsets=subsets, seeds=args.seeds)
