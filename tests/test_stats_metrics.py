import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from mmxai.data.cohort import patient_split
from mmxai.evaluation.metrics import metrics_from_confusion, select_threshold
from mmxai.evaluation.shapley import shapley_values
from mmxai.evaluation.stats import PatientBootstrap, delong_auc_cov, delong_test, holm
from mmxai.explain.attributions import permutation_shap


def test_metrics_reproduce_paper_model_f():
    # Section 4.3: Model F has TP 120, FN 30, FP 187, TN 663 -> Table 4 row F.
    m = metrics_from_confusion({"tp": 120, "fn": 30, "fp": 187, "tn": 663})
    assert round(m["accuracy"], 2) == 0.78
    assert round(m["precision"], 2) == 0.39
    assert round(m["recall"], 2) == 0.80
    assert round(m["specificity"], 2) == 0.78
    assert round(m["f1"], 2) == 0.52 or round(m["f1"], 2) == 0.53


def test_delong_auc_matches_sklearn_and_variance_matches_bootstrap():
    rng = np.random.default_rng(0)
    y = (rng.random(600) < 0.2).astype(int)
    a = y + rng.normal(0, 0.6, 600)
    b = y + rng.normal(0, 2.0, 600)
    aucs, cov = delong_auc_cov(y, np.vstack([a, b]))
    assert np.allclose(aucs, [roc_auc_score(y, a), roc_auc_score(y, b)])
    boot = [roc_auc_score(y[i], a[i]) for i in (rng.integers(0, 600, 600) for _ in range(800))]
    assert np.sqrt(cov[0, 0]) == pytest.approx(np.std(boot), rel=0.2)
    assert delong_test(y, a, b)["p"] < 0.05


def test_holm():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_shapley_efficiency_and_symmetry():
    v = {frozenset(): 0.5, frozenset("a"): 0.7, frozenset("b"): 0.7, frozenset("c"): 0.6,
         frozenset("ab"): 0.8, frozenset("ac"): 0.75, frozenset("bc"): 0.75, frozenset("abc"): 0.85}
    phi = shapley_values("abc", v)
    assert sum(phi.values()) == pytest.approx(0.35)
    assert phi["a"] == pytest.approx(phi["b"])


def test_permutation_shap_exact_for_linear_model():
    w = np.array([1.0, -2.0, 0.5])
    f = lambda rows: rows @ w  # noqa: E731
    x = np.array([1.0, 1.0, 1.0])
    bg = np.zeros((5, 3))
    phi, base = permutation_shap(f, x, bg, n_permutations=4)
    assert np.allclose(phi, w)
    assert base == pytest.approx(0.0)


def test_patient_split_has_no_leakage_and_bootstrap_resamples_patients():
    df = pd.DataFrame({"subject_id": np.repeat(np.arange(300), 3), "label": np.tile([0, 0, 1], 300)})
    df.loc[df.subject_id % 4 == 0, "label"] = 0
    split = patient_split(df, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=1)
    assert df.assign(split=split).groupby("subject_id")["split"].nunique().max() == 1
    assert split.value_counts(normalize=True)["train"] == pytest.approx(0.7, abs=0.03)
    boot = PatientBootstrap(df["subject_id"], n_boot=5)
    for idx in boot.samples():
        counts = df.iloc[idx].groupby("subject_id").size()
        assert (counts % 3 == 0).all()  # every stay of a drawn patient comes along


def test_threshold_rules():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.6, 0.4, 0.7, 0.9])
    assert select_threshold(y, p, "sensitivity", 1.0) == pytest.approx(0.4)
    assert 0.4 <= select_threshold(y, p, "youden") <= 0.7
