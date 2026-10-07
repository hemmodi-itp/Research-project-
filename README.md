# mmxai: Explainable multimodal pneumonia diagnosis

Reference implementation of *"An Explainable Multimodal AI Framework for Medical Diagnosis Using Clinical Text,
Medical Imaging, and Patient Data"* (Modi & Mehta). It fuses the first frontal chest radiograph of an ED stay, the
triage chief complaint and structured patient data into one calibrated pneumonia probability. Every prediction comes
with image, text, feature and modality-level explanations.

> Decision-support research code. It is not a medical device and must not be used for clinical decisions.

## Where each part of the paper lives

| Paper | Code |
|---|---|
| §3.1 cohort, leakage control, labels | [sql/cohort.sql](sql/cohort.sql), [data/cohort.py](src/mmxai/data/cohort.py) |
| §3.1 preprocessing, patient-level 70/15/15 split | [data/preprocess.py](src/mmxai/data/preprocess.py), [data/dataset.py](src/mmxai/data/dataset.py) |
| §3.2 encoders (ResNet-50, Bio_ClinicalBERT, feature tokeniser) | [models/encoders.py](src/mmxai/models/encoders.py) |
| §3.2 transformer fusion, `[MISSING]` tokens, auxiliary heads; Eqs. 1-5 | [models/fusion.py](src/mmxai/models/fusion.py) |
| XGBoost structured baseline | [models/xgb_baseline.py](src/mmxai/models/xgb_baseline.py) |
| Eqs. 6-7, modality dropout, AdamW + cosine, early stopping (Table 3) | [training/trainer.py](src/mmxai/training/trainer.py) |
| Temperature scaling (Eq. 5) | [training/calibration.py](src/mmxai/training/calibration.py) |
| Eqs. 8-12, AUROC, AUPRC, ECE, validation threshold | [evaluation/metrics.py](src/mmxai/evaluation/metrics.py) |
| DeLong, Holm, patient-level bootstrap | [evaluation/stats.py](src/mmxai/evaluation/stats.py) |
| Modality Shapley values (Table 5) | [evaluation/shapley.py](src/mmxai/evaluation/shapley.py) |
| Table 4, ablation (Fig. 4), Table 5, robustness (Fig. 6), subgroups, error analysis | [evaluation/report.py](src/mmxai/evaluation/report.py), [evaluation/figures.py](src/mmxai/evaluation/figures.py) |
| §4.3 Grad-CAM, Integrated Gradients, SHAP, local modality Shapley (Fig. 5) | [explain/attributions.py](src/mmxai/explain/attributions.py) |
| Faithfulness: deletion curves, weight randomisation | [explain/faithfulness.py](src/mmxai/explain/faithfulness.py) |
| Six baselines + seven-arm ablation × five seeds | [experiments.py](src/mmxai/experiments.py) |

## Setup

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; use .venv/bin/activate elsewhere
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # or /whl/cpu
pip install -r requirements.txt
pytest                                              # 14 unit tests, ~30 s on CPU
```

## Running on MIMIC

You need PhysioNet credentialed access to MIMIC-IV (hosp), MIMIC-IV-ED and MIMIC-CXR-JPG.

```bash
# 1. Build the cohort (DuckDB reads the .csv.gz files directly; labevents takes a while)
python scripts/build_cohort.py --hosp /data/mimiciv/3.1/hosp --ed /data/mimic-iv-ed/2.2/ed \
    --cxr /data/mimic-cxr-jpg/2.1.0 --out data/cohort.csv

# 2. Set data.image_root in configs/default.yaml to the MIMIC-CXR-JPG root, then train everything:
#    7 modality subsets + XGBoost, for 5 seeds. Finished runs are skipped on restart.
python scripts/train.py --config configs/default.yaml

# 3. Tables, hypothesis tests and figures
python scripts/evaluate.py --config configs/default.yaml

# 4. Explanations and faithfulness checks for a sample of test cases
python scripts/explain.py --config configs/default.yaml
```

Any config value can be overridden from the command line, for example
`--set train.max_epochs=10 seeds=[0]`. The sensitivity analysis with report-derived labels is the same pipeline
with a different label and output folder:
`--set data.label_col=label_report output_dir=outputs/report_labels`.

Compute: 35 neural runs (7 subsets × 5 seeds) fine-tune ResNet-50 at 384 × 384 and Bio_ClinicalBERT. Plan on one
GPU with at least 16 GB. Use `--subsets I+T+S --seeds 0` to try the full model alone first.

## Trying it without MIMIC

```bash
python scripts/make_synthetic.py --out data/synthetic
python scripts/train.py    --config configs/smoke.yaml
python scripts/evaluate.py --config configs/smoke.yaml
python scripts/explain.py  --config configs/smoke.yaml
```

The smoke config uses toy images, a small ResNet-18 trained from scratch and a word-level text encoder, so it runs
on a laptop CPU without downloads. Its numbers only show that the pipeline works. They say nothing about pneumonia.

## Outputs

`outputs/<name>/`
- `splits.csv`: patient → split. It is reused on later runs, so every model and seed sees the same partition.
- `preprocessor.json`: training-split statistics (winsor limits, medians, means, SDs, category lists).
- `runs/seed<k>/<subset>/`: `model.pt`, `meta.json` (temperature, threshold, training history),
  `predictions_val.csv` and `predictions_test.csv`. The full model's test file also has `prob_only_<subset>` columns:
  the same model with the other modalities withheld.
- `results/`: `table4_baselines.csv`, `hypothesis_tests.csv`, `ablation.csv`, `table5_modality_contribution.csv`,
  `robustness.csv`, `confusion.csv`, `subgroups.csv`, `calibration.csv`, `uncertainty.csv`, `error_review.csv`,
  the figures `fig3_baselines.png`, `fig4_ablation.png` and `fig6_robustness.png`, and `report.md`, which collects
  them all.
- `explanations/`: one PNG and JSON per case, `cases.csv`, `plausibility_ratings.csv` (a blank sheet for the
  clinician rating), `faithfulness.csv`, `sanity_randomisation.csv` and `deletion_curves.png`.

`error_review.csv` and `plausibility_ratings.csv` are where the human steps of §4.3 happen. Reviewers fill in the
error cause from Table 6 and the plausibility score. Nothing is assigned automatically.

## Decisions the paper leaves open

These are implemented as described below. Each one can be changed in the config or in one obvious place in the code.

1. **36 region tokens.** ResNet-50 at 384 px gives a 12 × 12 map. It is average-pooled to 6 × 6 to match Table 3
   (`model.image_grid`). Grad-CAM uses the unpooled 12 × 12 map.
2. **Structured tokens.** Each numeric feature's token is `x·W + flag·U + b`, so value and missingness share one
   token. Categorical and comorbidity flags get embedding tokens, with code 0 for missing or unseen.
3. **Learning rates.** 1e-5 goes to the *pretrained* encoders (ResNet, BERT). Parts trained from scratch, including
   the feature tokeniser, use 1e-4. The tokeniser cannot learn usefully at 1e-5.
4. **Modality dropout.** p = 0.3 per available modality. If a sample would lose all of them, one random modality is
   put back.
5. **Auxiliary heads** sit on each modality's projected tokens before fusion (masked mean pool). The loss counts
   them only for samples where that modality is present (Eq. 7).
6. **Threshold.** Youden's J on the validation set by default. `threshold_rule: sensitivity` with
   `target_sensitivity` picks a sensitivity-first operating point instead.
7. **Calibration caveat.** Class-weighted loss (Eq. 6) moves the predicted base rate toward 50%. Temperature scaling
   (the paper's method) cannot correct a shift, so post-calibration ECE can stay high. `train.calibration: platt`
   adds a bias term. `calibration.csv` reports ECE before and after, so you can choose. T is clipped to
   [0.05, 20], with a logged warning. Otherwise an offset model is "calibrated" toward a flat 0.5, which
   erases its ranking. The smoke run hit this once.
8. **Shapley values.** Table 5 is computed two ways. *retrained* uses the seven separately trained models.
   *fused_masked* uses one full model with modalities swapped for `[MISSING]` (the §3.3 definition). Both use
   v(∅) = 0.5. For a single case the reference is the mean validation log-odds, because the model never sees an
   all-missing input.
9. **SHAP** uses an interventional permutation estimator implemented in-house, the same estimator as
   `shap.PermutationExplainer`. That avoids the `shap`/`numba` dependency. A feature's value and its missingness flag
   are treated as one feature.
10. **Labs.** The latest value per test whose result was *available* (`storetime`) by t0, within the ED stay. Check
    the item ids in `cohort.sql` against `d_labitems` for your MIMIC-IV release.
11. **Hypothesis tests and CIs** use the primary seed (`primary_seed`). The other seeds give the seed-to-seed SD
    reported next to them.
