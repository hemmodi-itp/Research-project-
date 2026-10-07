"""Train every model of the study and store its validation/test predictions.

Per seed: the seven modality subsets (baselines A-F of Table 4 + the I+T ablation arm) and the XGBoost structured
baseline. The full model (I+T+S) is also run with every modality subset withheld at test time, which feeds the
masked Shapley values (Sec. 3.3) and the missing-modality analysis (Fig. 6). All later tables are computed from
these prediction files, so evaluation never needs a GPU.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import MODALITIES
from .data.cohort import get_or_make_split, load_cohort
from .data.dataset import MultimodalDataset, build_tokenizer
from .data.preprocess import TabularPreprocessor, normalize_text
from .evaluation.metrics import select_threshold
from .models.fusion import MultimodalFusionModel
from .models.xgb_baseline import train_xgboost, xgb_matrix
from .training.calibration import T_MAX, T_MIN, fit_calibration
from .training.trainer import predict, train_model
from .utils import ALL_SUBSETS, FULL, ensure_dir, get_device, load_json, save_json, set_seed, subset_key

XGB_KEY = "S-xgb"


def prepare(cfg, log):
    """Load the cohort, fix the patient split, fit preprocessing on the training split only."""
    out_dir = ensure_dir(cfg["output_dir"])
    df = load_cohort(cfg)
    df["split"] = get_or_make_split(df, cfg, out_dir)
    frames = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    log.info("cohort: " + ", ".join(f"{s} {len(f)} stays / {f['subject_id'].nunique()} patients / "
                                    f"{f['label'].mean():.1%} positive" for s, f in frames.items()))

    dcfg = cfg["data"]
    pre = TabularPreprocessor(dcfg["numeric_features"], dcfg["categorical_features"],
                              dcfg.get("valid_ranges"), dcfg["winsor_quantiles"]).fit(frames["train"])
    save_json(pre.state_dict(), out_dir / "preprocessor.json")
    tab = {s: pre.transform(f) for s, f in frames.items()}

    tokenizer = build_tokenizer(cfg["model"], [normalize_text(t) for t in frames["train"]["chief_complaint"]])
    if hasattr(tokenizer, "vocab"):
        save_json(tokenizer.vocab, out_dir / "vocab.json")

    datasets = {
        s: MultimodalDataset(frames[s], tab[s], tokenizer, dcfg["image_root"], dcfg["image_size"],
                             dcfg["max_text_tokens"], train=(s == "train"))
        for s in frames
    }
    return {"frames": frames, "pre": pre, "tab": tab, "tokenizer": tokenizer, "datasets": datasets,
            "out_dir": out_dir}


def loader(ctx, split, cfg, shuffle=False, seed=0):
    tcfg = cfg["train"]
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ctx["datasets"][split], batch_size=tcfg["batch_size"], shuffle=shuffle,
                      num_workers=tcfg["num_workers"], pin_memory=torch.cuda.is_available(), generator=gen,
                      persistent_workers=tcfg["num_workers"] > 0)


def build_model(cfg, subset, ctx):
    pre = ctx["pre"]
    return MultimodalFusionModel(cfg["model"], subset, n_numeric=len(pre.numeric),
                                 cardinalities=pre.cardinalities, vocab_size=ctx["tokenizer"].vocab_size,
                                 max_tokens=cfg["data"]["max_text_tokens"])


def run_dir(cfg, seed, key):
    return Path(cfg["output_dir"]) / "runs" / f"seed{seed}" / key


def _prediction_frame(frame, preds, model):
    idx = preds["index"]
    out = frame.iloc[idx][["stay_id", "subject_id", "label", "sex", "age"]].reset_index(drop=True)
    out["prob"] = preds["prob"]
    out["raw_log_odds"] = preds["logits"][:, 1] - preds["logits"][:, 0]
    for m in model.modalities:
        out[f"aux_prob_{m}"] = preds["aux_prob"][m]
        out[f"present_{m}"] = preds["present"][m]
    return out


def run_neural(cfg, ctx, subset, seed, log):
    key = subset_key(subset)
    rdir = ensure_dir(run_dir(cfg, seed, key))
    if (rdir / "meta.json").exists():
        log.info(f"[seed {seed}] {key}: already trained, skipping")
        return
    log.info(f"[seed {seed}] training {key}")
    set_seed(seed)
    device = get_device(cfg["train"]["device"])
    model = build_model(cfg, subset, ctx)
    fit = train_model(model, loader(ctx, "train", cfg, shuffle=True, seed=seed), loader(ctx, "val", cfg),
                      ctx["frames"]["train"]["label"].to_numpy(), cfg, device, log)

    # Calibrate and choose the threshold on validation data only.
    val_loader, test_loader = loader(ctx, "val", cfg), loader(ctx, "test", cfg)
    val = predict(model, val_loader, device)
    y_val = ctx["frames"]["val"]["label"].to_numpy()[val["index"]]
    T, b = fit_calibration(val["logits"], y_val, cfg["train"]["calibration"])
    if T in (T_MIN, T_MAX):
        log.warning(f"[seed {seed}] {key}: temperature hit its bound ({T}); the validation logits are offset, not just "
                    "over-confident. Consider train.calibration=platt.")
    model.temperature.fill_(T)
    model.logit_bias.fill_(b)
    val = predict(model, val_loader, device)
    threshold = select_threshold(y_val, val["prob"], cfg["train"]["threshold_rule"], cfg["train"]["target_sensitivity"])

    _prediction_frame(ctx["frames"]["val"], val, model).to_csv(rdir / "predictions_val.csv", index=False)
    test_df = _prediction_frame(ctx["frames"]["test"], predict(model, test_loader, device), model)
    if tuple(subset) == FULL:
        for sub in ALL_SUBSETS:
            if sub != FULL:
                masked = predict(model, test_loader, device, force_absent=[m for m in MODALITIES if m not in sub])
                test_df[f"prob_only_{subset_key(sub)}"] = masked["prob"]
    test_df.to_csv(rdir / "predictions_test.csv", index=False)

    torch.save({"state_dict": model.state_dict(), "subset": list(subset)}, rdir / "model.pt")
    save_json({"subset": list(subset), "seed": seed, "temperature": T, "logit_bias": b, "threshold": threshold,
               **fit}, rdir / "meta.json")
    log.info(f"[seed {seed}] {key}: val AUROC {fit['best_val_auroc']:.4f}, T={T:.3f}, threshold={threshold:.3f}")


def run_xgboost(cfg, ctx, seed, log):
    rdir = ensure_dir(run_dir(cfg, seed, XGB_KEY))
    if (rdir / "meta.json").exists():
        return
    frames, pre = ctx["frames"], ctx["pre"]
    model = train_xgboost(pre, frames["train"], frames["val"], cfg, seed)
    out = {}
    for split in ("val", "test"):
        f = frames[split][["stay_id", "subject_id", "label", "sex", "age"]].copy()
        f["prob"] = model.predict_proba(xgb_matrix(pre, frames[split]))[:, 1]
        out[split] = f
        f.to_csv(rdir / f"predictions_{split}.csv", index=False)
    threshold = select_threshold(out["val"]["label"], out["val"]["prob"], cfg["train"]["threshold_rule"],
                                 cfg["train"]["target_sensitivity"])
    model.save_model(rdir / "model.json")
    save_json({"seed": seed, "threshold": threshold, "best_iteration": int(model.best_iteration)}, rdir / "meta.json")
    log.info(f"[seed {seed}] {XGB_KEY}: best iteration {model.best_iteration}")


def run_all(cfg, log, subsets=None, seeds=None):
    ctx = prepare(cfg, log)
    for seed in seeds or cfg["seeds"]:
        for subset in subsets or ALL_SUBSETS:
            run_neural(cfg, ctx, subset, seed, log)
        run_xgboost(cfg, ctx, seed, log)
    return ctx


def load_trained(cfg, ctx, subset, seed, device=None):
    """Rebuild a trained model with its calibration, plus its stored metadata."""
    rdir = run_dir(cfg, seed, subset_key(subset))
    meta = load_json(rdir / "meta.json")
    model = build_model(cfg, subset, ctx)
    state = torch.load(rdir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state["state_dict"])
    model.to(device or get_device(cfg["train"]["device"])).eval()
    return model, meta


def load_predictions(cfg, seed, key, split="test"):
    path = run_dir(cfg, seed, key) / f"predictions_{split}.csv"
    return pd.read_csv(path) if path.exists() else None
