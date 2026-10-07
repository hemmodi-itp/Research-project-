"""Explain a sample of test predictions of the full model and check the explanations' faithfulness.

Outputs (in <output_dir>/explanations):
  case_<stay_id>.png / .json   image with Grad-CAM, top tokens, top structured features, modality contributions
  cases.csv                    one row per explained case
  plausibility_ratings.csv     template for the clinician plausibility rating (1-5 per explanation type)
  faithfulness.csv             deletion-curve areas: explanation order vs random order, per case
  sanity_randomisation.csv     rank correlation after re-initialising the top of each encoder
  deletion_curves.png          mean deletion curves
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .. import MODALITIES, MODALITY_NAMES
from ..data.dataset import IMAGENET_MEAN, IMAGENET_STD, stack_items, to_device
from ..experiments import load_predictions, load_trained, prepare
from ..evaluation.figures import INK, INK_2, MUTED, SERIES, SURFACE, GRID
from ..utils import FULL, ensure_dir, get_device, save_json, subset_key
from . import faithfulness as fa
from .attributions import (TabularValueFunction, grad_cam, integrated_gradients_text, local_modality_shapley,
                           permutation_shap)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def select_cases(pred, threshold, n, seed=0):
    """Spread the sample over TP, FN, FP and TN so reviewers also see explanations of errors."""
    rng = np.random.default_rng(seed)
    y, yhat = pred["label"].to_numpy(), (pred["prob"] >= threshold).to_numpy().astype(int)
    groups = {"TP": (y == 1) & (yhat == 1), "FN": (y == 1) & (yhat == 0),
              "FP": (y == 0) & (yhat == 1), "TN": (y == 0) & (yhat == 0)}
    per = max(1, n // 4)
    chosen = []
    for name, mask in groups.items():
        idx = np.where(mask)[0]
        for i in rng.choice(idx, size=min(per, len(idx)), replace=False):
            chosen.append((int(i), name))
    return chosen


def _background_rows(ctx, n, seed=0):
    """Random training encounters as structured rows (numeric z with NaN for missing, then category codes)."""
    tab = ctx["tab"]["train"]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(tab["num"]), size=min(n, len(tab["num"])), replace=False)
    num = np.where(tab["flag"][idx] > 0, np.nan, tab["num"][idx])
    return np.hstack([num, tab["cat"][idx].astype(float)])


def _reference_row(ctx):
    """Deletion reference: training median for numeric (z of median, present), modal category otherwise."""
    pre, tab = ctx["pre"], ctx["tab"]["train"]
    modes = [np.bincount(tab["cat"][:, j][tab["cat"][:, j] > 0]).argmax() if (tab["cat"][:, j] > 0).any() else 0
             for j in range(tab["cat"].shape[1])]
    return np.concatenate([pre.impute_z, np.array(modes, dtype=float)])


def _plot_case(path, image, cam, title, tokens, tok_scores, feat_names, phi, modality_phi):
    fig = plt.figure(figsize=(11, 3.8), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.3, 1.0], wspace=0.45)

    ax = fig.add_subplot(gs[0])
    ax.imshow(image, cmap="gray")
    overlay = np.zeros((*cam.shape, 4))
    overlay[..., :3] = matplotlib.colors.to_rgb(SERIES[1])
    overlay[..., 3] = 0.6 * cam
    ax.imshow(overlay)
    ax.set_title("Grad-CAM", color=INK, fontsize=9, loc="left")
    ax.axis("off")

    ax = fig.add_subplot(gs[1])
    order = np.argsort(-np.abs(phi))[:8][::-1]
    ax.barh(np.arange(len(order)), phi[order], color=[SERIES[0] if v > 0 else MUTED for v in phi[order]], height=0.6)
    ax.set_yticks(np.arange(len(order)), [feat_names[i] for i in order], fontsize=8, color=INK_2)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_title("Structured features (SHAP, log-odds)", color=INK, fontsize=9, loc="left")
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)

    ax = fig.add_subplot(gs[2])
    ax.axis("off")
    lines = ["Modality contribution (log-odds)"]
    lines += [f"  {MODALITY_NAMES[m]:<16} {v:+.2f}" for m, v in modality_phi.items()]
    lines += ["", "Chief-complaint tokens (IG)"]
    if len(tokens):
        top = np.argsort(-tok_scores)[:6]
        lines += [f"  {tokens[i]:<16} {tok_scores[i]:+.3f}" for i in top]
    else:
        lines += ["  (no chief complaint)"]
    ax.text(0, 1, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=8, color=INK_2)

    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def run_explanations(cfg, log):
    xcfg = cfg["explain"]
    seed = cfg["primary_seed"]
    out = ensure_dir(Path(cfg["output_dir"]) / "explanations")
    ctx = prepare(cfg, log)
    device = get_device(cfg["train"]["device"])
    model, meta = load_trained(cfg, ctx, FULL, seed, device)
    threshold = meta["threshold"]
    pred = load_predictions(cfg, seed, subset_key(FULL), "test")
    base_value = float(_logit(load_predictions(cfg, seed, subset_key(FULL), "val")["prob"].to_numpy()).mean())

    pre, tokenizer, ds = ctx["pre"], ctx["tokenizer"], ctx["datasets"]["test"]
    feat_names = pre.feature_names
    background = _background_rows(ctx, xcfg["shap_background"])
    reference = _reference_row(ctx)
    rand_models = {part: fa.randomized_copy(model, part) for part in MODALITIES}
    rng = np.random.default_rng(0)
    steps = xcfg["deletion_steps"]

    cases, faith, sanity, curves = [], [], [], {k: [] for k in
                                                 ("image", "image_random", "text", "text_random", "tab", "tab_random")}
    for i, outcome in select_cases(pred, threshold, xcfg["n_cases"]):
        batch = to_device(stack_items([ds[i]]), device)
        stay_id = int(pred["stay_id"].iloc[i])
        with torch.no_grad():
            prob = float(model.predict_proba(model(batch)["logits"])[0])

        has_image = bool(batch["present_image"][0])
        cam = grad_cam(model, batch) if has_image else np.zeros(batch["image"].shape[-2:])
        tokens, tok_scores, positions = integrated_gradients_text(model, batch, tokenizer, xcfg["ig_steps"]) \
            if bool(batch["present_text"][0]) else ([], np.array([]), np.array([], dtype=int))
        f = TabularValueFunction(model, batch, pre)
        x_row = f.encounter_row()
        phi, shap_base = permutation_shap(f, x_row, background, xcfg["shap_permutations"])
        modality_phi, log_odds = local_modality_shapley(model, batch, base_value)

        # Faithfulness: explanation-ordered deletion vs random-ordered deletion.
        c_tab = fa.deletion_curve_tab(f, x_row, reference, phi, steps)
        c_tab_r = fa.deletion_curve_tab(f, x_row, reference, phi, steps, rng=rng)
        row = {"stay_id": stay_id, "tab_deletion": fa.area(c_tab), "tab_deletion_random": fa.area(c_tab_r)}
        curves["tab"].append(c_tab - c_tab[0]); curves["tab_random"].append(c_tab_r - c_tab_r[0])
        if has_image:
            c_img = fa.deletion_curve_image(model, batch, cam, steps)
            c_img_r = fa.deletion_curve_image(model, batch, cam, steps, order=rng.permutation(cam.size))
            row.update({"image_deletion": fa.area(c_img), "image_deletion_random": fa.area(c_img_r)})
            curves["image"].append(c_img - c_img[0]); curves["image_random"].append(c_img_r - c_img_r[0])
        if len(tokens) > 1:
            c_txt = fa.deletion_curve_text(model, batch, tokenizer, positions, tok_scores, steps)
            c_txt_r = fa.deletion_curve_text(model, batch, tokenizer, positions, tok_scores, steps, rng=rng)
            row.update({"text_deletion": fa.area(c_txt), "text_deletion_random": fa.area(c_txt_r)})
            curves["text"].append(c_txt - c_txt[0]); curves["text_random"].append(c_txt_r - c_txt_r[0])
        faith.append(row)

        # Sanity check: explanations should change when the model's learned weights are destroyed.
        s_row = {"stay_id": stay_id}
        if has_image:
            s_row["image_gradcam_rho"] = fa.rank_similarity(cam, grad_cam(rand_models["image"], batch))
        if len(tokens) > 1:
            s_row["text_ig_rho"] = fa.rank_similarity(
                tok_scores, integrated_gradients_text(rand_models["text"], batch, tokenizer, xcfg["ig_steps"])[1])
        f_rand = TabularValueFunction(rand_models["tab"], batch, pre)
        s_row["tab_shap_rho"] = fa.rank_similarity(phi, permutation_shap(f_rand, x_row, background,
                                                                         xcfg["shap_permutations"])[0])
        sanity.append(s_row)

        image = batch["image"][0].cpu().numpy().transpose(1, 2, 0) * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
        top_feats = [feat_names[j] for j in np.argsort(-phi)[:3]]
        top_toks = [tokens[j] for j in np.argsort(-tok_scores)[:3]] if len(tokens) else []
        title = (f"stay {stay_id} | {outcome} | label {int(pred['label'].iloc[i])} | p = {prob:.2f} "
                 f"(threshold {threshold:.2f})")
        _plot_case(out / f"case_{stay_id}.png", image.clip(0, 1)[..., 0], cam, title, tokens, tok_scores,
                   feat_names, phi, modality_phi)
        record = {"stay_id": stay_id, "outcome": outcome, "label": int(pred["label"].iloc[i]), "prob": prob,
                  "log_odds": log_odds, "base_log_odds": base_value,
                  **{f"modality_phi_{m}": v for m, v in modality_phi.items()},
                  "top_modality": MODALITY_NAMES[max(modality_phi, key=lambda m: modality_phi[m])],
                  "top_features": ", ".join(top_feats), "top_tokens": ", ".join(top_toks)}
        save_json({**record, "tokens": list(tokens), "token_ig": tok_scores, "features": feat_names,
                   "shap": phi, "shap_base": shap_base}, out / f"case_{stay_id}.json")
        cases.append(record)
        log.info(f"explained stay {stay_id} ({outcome}): top modality {record['top_modality']}, "
                 f"features {record['top_features']}")

    cases = pd.DataFrame(cases)
    cases.to_csv(out / "cases.csv", index=False)
    rating = cases[["stay_id", "outcome", "prob", "top_modality", "top_features", "top_tokens"]].copy()
    for col in ("gradcam_plausibility_1to5", "text_plausibility_1to5", "features_plausibility_1to5", "rater", "notes"):
        rating[col] = ""
    rating.to_csv(out / "plausibility_ratings.csv", index=False)

    faith = pd.DataFrame(faith)
    faith.to_csv(out / "faithfulness.csv", index=False)
    pd.DataFrame(sanity).to_csv(out / "sanity_randomisation.csv", index=False)
    _plot_deletion(curves, out / "deletion_curves.png")
    summary = {c: float(faith[c].mean()) for c in faith.columns if c != "stay_id"}
    summary.update({f"mean_{c}": float(pd.DataFrame(sanity)[c].mean()) for c in pd.DataFrame(sanity).columns
                    if c != "stay_id"})
    save_json(summary, out / "faithfulness_summary.json")
    log.info(f"explanations written to {out}")
    return summary


def _plot_deletion(curves, path):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), facecolor=SURFACE, sharey=False)
    for ax, (name, label) in zip(axes, (("image", "Image (Grad-CAM)"), ("text", "Text (IG)"),
                                        ("tab", "Structured (SHAP)"))):
        ax.set_facecolor(SURFACE)
        for key, color, lab in ((name, SERIES[0], "Explanation order"), (f"{name}_random", SERIES[1], "Random order")):
            if curves[key]:
                c = np.mean(curves[key], axis=0)
                ax.plot(np.linspace(0, 100, len(c)), c, color=color, linewidth=2, label=lab)
        ax.set_title(label, color=INK, fontsize=9, loc="left")
        ax.set_xlabel("% of evidence removed", color=INK_2, fontsize=8)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        ax.grid(color=GRID, linewidth=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Change in log-odds", color=INK_2, fontsize=8)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
