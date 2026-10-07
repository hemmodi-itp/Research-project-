"""Compute Tables 4-6, hypothesis tests, ablation, robustness, subgroup and calibration results and figures."""

from _common import parse

from mmxai.evaluation.report import Evaluator

if __name__ == "__main__":
    _, cfg, log = parse(__doc__)
    Evaluator(cfg, log).run()
