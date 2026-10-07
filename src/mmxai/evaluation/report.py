"""Build every table and figure of Section 4 from stored predictions.

Outputs (in <output_dir>/results):
  table4_baselines.csv        Table 4 + XGBoost, with AUROC CIs and seed-to-seed SD
  hypothesis_tests.csv        one-sided DeLong F vs A/B/C, Holm-adjusted (H0/H1, RQ1)
  ablation.csv                Fig. 4: seven retrained subsets and the same subsets fed to the fused model
  table5_modality_contribution.csv   Shapley values and AUROC lost if removed (RQ2)
  robustness.csv              Fig. 6: fused model with a modality withheld vs retrained specialist (RQ4)
  confusion.csv               confusion matrices of F and B at their validation thresholds
  subgroups.csv               sensitivity / specificity by sex and age, with between-group differences
  calibration.csv             ECE before and after calibration
  uncertainty.csv             predictions flagged for clinician review
  error_review.csv            Model F errors, with an empty column for the reviewer's cause (Table 6)
  report.md                   all of the above as one readable document
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .. import MODALITIES, MODALITY_NAMES
from ..experiments import XGB_KEY, load_predictions, run_dir
from ..utils import ALL_SUBSETS, BASELINES, FULL, ensure_dir, load_json, subset_key
from .metrics import all_metrics, auroc, confusion, expected_calibration_error, metrics_from_confusion
from .shapley import shapley_values
from .stats import PatientBootstrap, delong_test, holm

ERROR_CATEGORIES = [
    "Subtle or low-quality imaging",
    "Probable label noise",
    "Clinical mimic (oedema, atelectasis, heart failure)",
    "Atypical or uninformative chief complaint",
    "Laboratory results not yet available",
    "Other",
]

ROBUSTNESS_SCENARIOS = [  # (scenario, withheld modality)
    ("A: all present", None),
    ("B: image missing", "image"),
    ("C: text missing", "text"),
    ("D: structured missing", "tab"),
]


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Evaluator:
    def __init__(self, cfg, log):
        self.cfg, self.log = cfg, log
        self.ecfg = cfg["eval"]
        self.seed = cfg["primary_seed"]
        self.out = ensure_dir(Path(cfg["output_dir"]) / "results")
        self.keys = [subset_key(s) for s in ALL_SUBSETS] + [XGB_KEY]
        self.preds = {s: {k: load_predictions(cfg, s, k) for k in self.keys} for s in cfg["seeds"]}
        self.meta = {s: {k: load_json(run_dir(cfg, s, k) / "meta.json") for k in self.keys
                         if (run_dir(cfg, s, k) / "meta.json").exists()} for s in cfg["seeds"]}
        base = self.preds[self.seed][subset_key(FULL)]
        if base is None:
            raise FileNotFoundError("Predictions of the full model for the primary seed are missing; run training first.")
        for key, df in self.preds[self.seed].items():
            if df is not None and not np.array_equal(df["stay_id"].to_numpy(), base["stay_id"].to_numpy()):
                raise ValueError(f"Test predictions of {key} are not aligned with the full model.")
        self.base = base
        self.y = base["label"].to_numpy()
        self.boot = PatientBootstrap(base["subject_id"].to_numpy(), self.ecfg["n_bootstrap"], seed=0)

    # ------------------------------------------------------------------ helpers
    def prob(self, key, seed=None):
        df = self.preds[self.seed if seed is None else seed][key]
        return None if df is None else df["prob"].to_numpy()

    def threshold(self, key, seed=None):
        return self.meta[self.seed if seed is None else seed][key]["threshold"]

    def auroc_ci(self, prob):
        y = self.y
        return self.boot.ci(lambda i: auroc(y[i], prob[i]), self.ecfg["alpha"])

    def seed_aurocs(self, key):
        vals = []
        for s in self.cfg["seeds"]:
            df = self.preds[s][key]
            if df is not None:
                vals.append(auroc(df["label"], df["prob"]))
        return np.array(vals)

    # ------------------------------------------------------------------ RQ1
    def baselines(self):
        rows = []
        for letter, subset in list(BASELINES.items()) + [("A-xgb", ("tab",))]:
            key = XGB_KEY if letter == "A-xgb" else subset_key(subset)
            p = self.prob(key)
            if p is None:
                continue
            m = all_metrics(self.y, p, self.threshold(key), self.ecfg["ece_bins"])
            lo, hi = self.auroc_ci(p)
            seeds = self.seed_aurocs(key)
            rows.append({"model": letter, "inputs": subset_key(subset) + (" (XGBoost)" if letter == "A-xgb" else ""),
                         **{k: m[k] for k in ("accuracy", "precision", "recall", "specificity", "f1", "auroc")},
                         "auroc_ci_low": lo, "auroc_ci_high": hi, "auprc": m["auprc"], "ece": m["ece"],
                         "auroc_seed_mean": seeds.mean(), "auroc_seed_sd": seeds.std(ddof=1) if len(seeds) > 1 else np.nan,
                         "n_seeds": len(seeds)})
        return pd.DataFrame(rows)

    def hypothesis_tests(self):
        full = self.prob(subset_key(FULL))
        rows = []
        for letter in ("A", "B", "C"):
            p = self.prob(subset_key(BASELINES[letter]))
            rows.append({"comparison": f"F vs {letter}", **delong_test(self.y, full, p, alternative="greater")})
        df = pd.DataFrame(rows)
        df["p_holm"] = holm(df["p"])
        df["reject_H0"] = df["p_holm"] < self.ecfg["alpha"]
        return df

    # ------------------------------------------------------------------ RQ2
    def _value_function(self, seed, masked):
        """v(S) = test AUROC with only subset S, v(empty) = 0.5 (chance)."""
        df_full = self.preds[seed][subset_key(FULL)]
        y = df_full["label"].to_numpy()
        v = {frozenset(): 0.5}
        for sub in ALL_SUBSETS:
            if masked:
                col = "prob" if sub == FULL else f"prob_only_{subset_key(sub)}"
                p = df_full[col].to_numpy()
            else:
                df = self.preds[seed][subset_key(sub)]
                if df is None:
                    return None
                p = df["prob"].to_numpy()
            v[frozenset(sub)] = auroc(y, p)
        return v

    def ablation(self):
        full_df = self.base
        rows = []
        for sub in ALL_SUBSETS:
            key = subset_key(sub)
            p = self.prob(key)
            p_masked = full_df["prob" if sub == FULL else f"prob_only_{key}"].to_numpy()
            lo, hi = self.auroc_ci(p) if p is not None else (np.nan, np.nan)
            mlo, mhi = self.auroc_ci(p_masked)
            seeds = self.seed_aurocs(key)
            rows.append({"subset": key, "n_modalities": len(sub),
                         "auroc_retrained": auroc(self.y, p) if p is not None else np.nan,
                         "ci_low": lo, "ci_high": hi,
                         "auroc_seed_sd": seeds.std(ddof=1) if len(seeds) > 1 else np.nan,
                         "auroc_fused_masked": auroc(self.y, p_masked), "masked_ci_low": mlo, "masked_ci_high": mhi})
        return pd.DataFrame(rows)

    def modality_contribution(self):
        rows = []
        for variant, masked in (("retrained", False), ("fused_masked", True)):
            per_seed = [v for s in self.cfg["seeds"] if (v := self._value_function(s, masked)) is not None]
            phis = [shapley_values(MODALITIES, v) for v in per_seed]
            # Bootstrap CI of phi on the primary seed (same resamples for every subset -> paired).
            v_primary = self._value_function(self.seed, masked)
            boot_phi = self._bootstrap_shapley(masked) if v_primary is not None else None
            for m in MODALITIES:
                vals = np.array([phi[m] for phi in phis])
                lost = np.array([v[frozenset(FULL)] - v[frozenset(x for x in FULL if x != m)] for v in per_seed])
                rows.append({"variant": variant, "modality": MODALITY_NAMES[m],
                             "shapley_auroc": vals.mean(), "shapley_seed_sd": vals.std(ddof=1) if len(vals) > 1 else np.nan,
                             "shapley_ci_low": boot_phi[m][0] if boot_phi else np.nan,
                             "shapley_ci_high": boot_phi[m][1] if boot_phi else np.nan,
                             "auroc_lost_if_removed": lost.mean(),
                             "full_gain_over_chance": np.mean([v[frozenset(FULL)] - 0.5 for v in per_seed])})
        return pd.DataFrame(rows)

    def _bootstrap_shapley(self, masked):
        full_df = self.base
        probs = {}
        for sub in ALL_SUBSETS:
            if masked:
                probs[sub] = full_df["prob" if sub == FULL else f"prob_only_{subset_key(sub)}"].to_numpy()
            else:
                probs[sub] = self.prob(subset_key(sub))
        draws = {m: [] for m in MODALITIES}
        for idx in self.boot.samples():
            y = self.y[idx]
            if len(np.unique(y)) < 2:
                continue
            v = {frozenset(): 0.5, **{frozenset(s): auroc(y, p[idx]) for s, p in probs.items()}}
            for m, val in shapley_values(MODALITIES, v).items():
                draws[m].append(val)
        a = self.ecfg["alpha"]
        return {m: (np.quantile(d, a / 2), np.quantile(d, 1 - a / 2)) for m, d in draws.items()}

    # ------------------------------------------------------------------ RQ4
    def robustness(self):
        full_key = subset_key(FULL)
        thr = self.threshold(full_key)
        rows = []
        for name, withheld in ROBUSTNESS_SCENARIOS:
            remaining = tuple(m for m in FULL if m != withheld)
            key = subset_key(remaining)
            fused = self.base["prob" if withheld is None else f"prob_only_{key}"].to_numpy()
            special = self.prob(key)
            c = metrics_from_confusion(confusion(self.y, fused, thr))
            row = {"scenario": name, "inputs": key, "auroc_fused": auroc(self.y, fused),
                   "fused_ci": self.auroc_ci(fused), "sensitivity_fused": c["recall"],
                   "specificity_fused": c["specificity"]}
            if withheld is not None and special is not None:
                y = self.y
                diff_ci = self.boot.ci(lambda i: auroc(y[i], fused[i]) - auroc(y[i], special[i]), self.ecfg["alpha"])
                test = delong_test(self.y, fused, special, alternative="two-sided")
                row.update({"auroc_specialist": auroc(self.y, special), "specialist_ci": self.auroc_ci(special),
                            "diff_fused_minus_specialist": test["diff"], "diff_ci": diff_ci, "delong_p": test["p"]})
            rows.append(row)
        df = pd.DataFrame(rows)
        for col in ("fused_ci", "specialist_ci", "diff_ci"):
            if col in df:
                df[[f"{col}_low", f"{col}_high"]] = pd.DataFrame(
                    [c if isinstance(c, tuple) else (np.nan, np.nan) for c in df.pop(col)], index=df.index)
        return df

    # ------------------------------------------------------------------ error analysis
    def confusion_table(self):
        rows = []
        for letter in ("F", "B"):
            key = subset_key(BASELINES[letter])
            if self.prob(key) is not None:
                rows.append({"model": letter, **confusion(self.y, self.prob(key), self.threshold(key))})
        return pd.DataFrame(rows)

    def uncertainty(self):
        key = subset_key(FULL)
        df = self.base.copy()
        thr = self.threshold(key)
        df["near_threshold"] = (df["prob"] - thr).abs() < self.ecfg["uncertainty_margin"]
        votes = []
        for m in MODALITIES:
            vote = (df[f"aux_prob_{m}"] >= 0.5).astype(float)
            votes.append(vote.where(df[f"present_{m}"].astype(bool)))
        votes = pd.concat(votes, axis=1)
        df["heads_disagree"] = votes.nunique(axis=1) > 1
        df["flag_review"] = df["near_threshold"] | df["heads_disagree"]
        df["pred"] = (df["prob"] >= thr).astype(int)
        df["error"] = df["pred"] != df["label"]
        return df

    def error_review(self, unc):
        cohort = pd.read_csv(self.cfg["data"]["cohort_csv"], usecols=lambda c: c in (
            "stay_id", "chief_complaint", "view_position", "image_path"))
        err = unc[unc["error"]].merge(cohort, on="stay_id", how="left")
        err["error_type"] = np.where(err["label"] == 1, "FN", "FP")
        err["reviewer_category"] = ""
        err["reviewer_notes"] = ""
        cols = ["stay_id", "subject_id", "error_type", "prob", "near_threshold", "heads_disagree"] + \
               [f"aux_prob_{m}" for m in MODALITIES] + \
               [c for c in ("view_position", "chief_complaint", "image_path") if c in err] + \
               ["reviewer_category", "reviewer_notes"]
        return err[cols].sort_values(["error_type", "prob"])

    # ------------------------------------------------------------------ fairness
    def subgroups(self, letter):
        key = subset_key(BASELINES[letter])
        p = self.prob(key)
        if p is None:
            return pd.DataFrame()
        pred = (p >= self.threshold(key)).astype(int)
        y = self.y
        bins = self.ecfg["age_bins"]
        labels = [f"{lo}-{hi - 1}" if hi < 150 else f"{lo}+" for lo, hi in zip(bins[:-1], bins[1:])]
        groups = {
            "sex": self.base["sex"].astype(str).to_numpy(),
            "age": pd.cut(self.base["age"], bins, right=False, labels=labels).astype(str).to_numpy(),
        }

        def sens(i, g, attr):
            sel = i[(groups[attr][i] == g) & (y[i] == 1)]
            return pred[sel].mean() if len(sel) else np.nan

        def spec(i, g, attr):
            sel = i[(groups[attr][i] == g) & (y[i] == 0)]
            return 1 - pred[sel].mean() if len(sel) else np.nan

        a = self.ecfg["alpha"]
        everyone = np.arange(len(y))
        rows = []
        for attr, values in groups.items():
            levels = [g for g in pd.unique(values) if g not in ("nan", "None")]
            for g in sorted(levels):
                rows.append({"model": letter, "attribute": attr, "group": g, "n": int((values == g).sum()),
                             "positives": int(((values == g) & (y == 1)).sum()),
                             "sensitivity": sens(everyone, g, attr),
                             "sensitivity_ci": self.boot.ci(lambda i: sens(i, g, attr), a),
                             "specificity": spec(everyone, g, attr),
                             "specificity_ci": self.boot.ci(lambda i: spec(i, g, attr), a)})
            for g1, g2 in combinations(sorted(levels), 2):
                row = {"model": letter, "attribute": attr, "group": f"{g1} minus {g2}", "fairness_concern": False}
                for metric, fn in (("sensitivity", sens), ("specificity", spec)):
                    ci = self.boot.ci(lambda i: fn(i, g1, attr) - fn(i, g2, attr), a)
                    row[metric] = fn(everyone, g1, attr) - fn(everyone, g2, attr)
                    row[f"{metric}_ci"] = ci
                    row["fairness_concern"] |= bool(ci[0] > 0 or ci[1] < 0)  # interval excludes zero
                rows.append(row)
        return pd.DataFrame(rows)

    def calibration(self):
        rows = []
        for sub in ALL_SUBSETS:
            df = self.preds[self.seed][subset_key(sub)]
            if df is None:
                continue
            rows.append({"subset": subset_key(sub),
                         "ece_uncalibrated": expected_calibration_error(df["label"], _sigmoid(df["raw_log_odds"]),
                                                                        self.ecfg["ece_bins"]),
                         "ece_calibrated": expected_calibration_error(df["label"], df["prob"], self.ecfg["ece_bins"]),
                         "temperature": self.meta[self.seed][subset_key(sub)]["temperature"],
                         "logit_bias": self.meta[self.seed][subset_key(sub)]["logit_bias"]})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ run everything
    def run(self):
        from .figures import plot_ablation, plot_baselines, plot_robustness

        log = self.log
        t = {}
        log.info("Table 4: baselines");               t["table4_baselines"] = self.baselines()
        log.info("DeLong tests");                     t["hypothesis_tests"] = self.hypothesis_tests()
        log.info("Ablation");                         t["ablation"] = self.ablation()
        log.info("Table 5: modality Shapley values"); t["table5_modality_contribution"] = self.modality_contribution()
        log.info("Missing-modality robustness");      t["robustness"] = self.robustness()
        t["confusion"] = self.confusion_table()
        log.info("Subgroups");                        t["subgroups"] = pd.concat([self.subgroups("F"), self.subgroups("B")])
        t["calibration"] = self.calibration()
        unc = self.uncertainty()
        t["uncertainty"] = pd.DataFrame([{
            "flagged": int(unc["flag_review"].sum()), "flagged_share": unc["flag_review"].mean(),
            "error_rate_flagged": unc.loc[unc["flag_review"], "error"].mean(),
            "error_rate_unflagged": unc.loc[~unc["flag_review"], "error"].mean(),
            "near_threshold": int(unc["near_threshold"].sum()), "heads_disagree": int(unc["heads_disagree"].sum())}])
        t["error_review"] = self.error_review(unc)

        for name, df in t.items():
            df.to_csv(self.out / f"{name}.csv", index=False)
        plot_baselines(t["table4_baselines"], self.out / "fig3_baselines.png")
        plot_ablation(t["ablation"], self.out / "fig4_ablation.png")
        plot_robustness(t["robustness"], self.out / "fig6_robustness.png")
        write_markdown(t, self.cfg, self.out / "report.md")
        log.info(f"results written to {self.out}")
        return t


def _fmt(v):
    if isinstance(v, tuple):
        return "(" + "–".join(_fmt(x) for x in v) + ")"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else ""
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return ""
        return f"{v:.2e}" if 0 < abs(v) < 1e-3 else f"{v:.3f}"
    return str(v)


def markdown_table(df):
    df = df.copy()
    for c in df.columns:  # counts become floats when a column has gaps; show them as integers again
        v = df[c]
        if v.dtype.kind == "f" and v.notna().any() and (v.dropna() % 1 == 0).all() and v.max() > 1:
            df[c] = v.astype("Int64").astype(str).replace("<NA>", "")
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_markdown(t, cfg, path):
    sections = [
        f"# Results: {cfg['name']}",
        f"Primary seed {cfg['primary_seed']}; seeds {cfg['seeds']}; {cfg['eval']['n_bootstrap']} patient-level "
        f"bootstrap resamples; label column `{cfg['data']['label_col']}`.",
        "## Table 4. Baseline comparison (test set)", markdown_table(t["table4_baselines"]),
        "## Hypothesis tests (one-sided DeLong, Holm-adjusted)", markdown_table(t["hypothesis_tests"]),
        "## Ablation (Fig. 4)", markdown_table(t["ablation"]),
        "## Table 5. Modality contribution", markdown_table(t["table5_modality_contribution"]),
        "## Missing-modality robustness (Fig. 6)", markdown_table(t["robustness"]),
        "## Confusion matrices", markdown_table(t["confusion"]),
        "## Subgroup sensitivity and specificity", markdown_table(t["subgroups"]),
        "## Calibration", markdown_table(t["calibration"]),
        "## Flagged for clinician review", markdown_table(t["uncertainty"]),
        "## Error review",
        f"{len(t['error_review'])} errors of Model F are listed in `error_review.csv`. Assign each a "
        "`reviewer_category` from: " + "; ".join(ERROR_CATEGORIES) + ".",
        "![Fig. 3](fig3_baselines.png)\n\n![Fig. 4](fig4_ablation.png)\n\n![Fig. 6](fig6_robustness.png)",
    ]
    Path(path).write_text("\n\n".join(sections) + "\n", encoding="utf-8")
