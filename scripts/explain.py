"""Explain sampled test predictions (Grad-CAM, IG, SHAP, modality Shapley) and run faithfulness checks."""

from _common import parse

from mmxai.explain.run import run_explanations

if __name__ == "__main__":
    _, cfg, log = parse(__doc__)
    run_explanations(cfg, log)
