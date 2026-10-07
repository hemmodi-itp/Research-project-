"""Faithfulness checks for the explanations (Sec. 4.3): deletion curves and model-randomisation sanity checks [13].

Deletion: remove the most-attributed evidence first and track the pneumonia log-odds. A faithful explanation makes
the score fall faster than deleting in random order (lower area under the deletion curve).
Randomisation: re-initialise the top of an encoder and recompute the explanation. Explanations that barely change
(high rank correlation) are not reflecting what the model learned.
"""

import copy

import numpy as np
import torch
from scipy.stats import spearmanr


@torch.no_grad()
def deletion_curve_image(model, batch, cam, steps=10, order=None):
    """Replace the top-k fraction of pixels (by `cam`, or by `order`) with the image mean, k = 0..1."""
    img = batch["image"][0]
    flat_rank = np.argsort(-cam.ravel()) if order is None else order
    n = len(flat_rank)
    variants = []
    for k in np.linspace(0, 1, steps + 1):
        x = img.clone().view(img.shape[0], -1)
        idx = torch.as_tensor(flat_rank[: int(round(k * n))], device=img.device)
        x[:, idx] = img.view(img.shape[0], -1).mean(1, keepdim=True)
        variants.append(x.view_as(img))
    images = torch.stack(variants)
    present = {m: batch[f"present_{m}"].bool() for m in model.modalities}
    tokens = {m: model.encode_present(m, batch, present[m]) for m in model.modalities if m != "image"}
    n_var = len(images)
    tokens = {m: (t.expand(n_var, *t.shape[1:]), p.expand(n_var, *p.shape[1:])) for m, (t, p) in tokens.items()}
    tokens["image"] = model.encode("image", {"image": images})
    out = model.fuse(tokens, {m: v.expand(n_var) for m, v in present.items()})
    return model.log_odds(out["logits"]).float().cpu().numpy()


@torch.no_grad()
def deletion_curve_text(model, batch, tokenizer, positions, scores, steps=10, rng=None):
    """Replace the most-attributed tokens with [PAD] (attention kept, as in the IG baseline)."""
    order = positions[np.argsort(-scores)] if rng is None else rng.permutation(positions)
    ids = batch["input_ids"][0]
    variants = []
    for k in np.linspace(0, 1, steps + 1):
        v = ids.clone()
        v[torch.as_tensor(order[: int(round(k * len(order)))], dtype=torch.long, device=ids.device)] = tokenizer.pad_token_id
        variants.append(v)
    n_var = len(variants)
    present = {m: batch[f"present_{m}"].bool() for m in model.modalities}
    tokens = {m: model.encode_present(m, batch, present[m]) for m in model.modalities if m != "text"}
    tokens = {m: (t.expand(n_var, *t.shape[1:]), p.expand(n_var, *p.shape[1:])) for m, (t, p) in tokens.items()}
    tokens["text"] = model.encode("text", {"input_ids": torch.stack(variants),
                                           "attention_mask": batch["attention_mask"].expand(n_var, -1)})
    out = model.fuse(tokens, {m: v.expand(n_var) for m, v in present.items()})
    return model.log_odds(out["logits"]).float().cpu().numpy()


def deletion_curve_tab(f, x, reference, phi, steps=10, rng=None):
    """Switch structured features to a reference row (training medians / modes), most positive SHAP first."""
    order = np.argsort(-phi) if rng is None else rng.permutation(len(phi))
    rows = []
    for k in np.linspace(0, 1, steps + 1):
        r = x.copy()
        sel = order[: int(round(k * len(order)))]
        r[sel] = reference[sel]
        rows.append(r)
    return f(np.array(rows))


def area(curve):
    """Mean drop of the score below its starting value; larger = evidence removed mattered more."""
    curve = np.asarray(curve)
    return float(np.mean(curve[0] - curve))


def randomized_copy(model, part):
    """Copy of the model with the top of one encoder re-initialised (cascading randomisation, top layer)."""
    clone = copy.deepcopy(model)
    if part == "image":
        target = clone.encoders["image"].backbone[-1]
    elif part == "text":
        enc = clone.encoders["text"]
        target = enc.model.encoder.layer[-1] if hasattr(enc, "model") else enc.encoder.layers[-1]
    else:
        target = clone.encoders["tab"]
    torch.manual_seed(1234)
    for module in target.modules():
        if hasattr(module, "reset_parameters") and module is not target:
            module.reset_parameters()
    if part == "tab":
        bound = 1 / np.sqrt(target.out_dim)
        for p in (target.num_weight, target.flag_weight, target.num_bias, target.cat_bias):
            torch.nn.init.uniform_(p, -bound, bound)
    return clone.eval()


def rank_similarity(a, b):
    """Spearman correlation of absolute attributions."""
    a, b = np.abs(np.ravel(a)), np.abs(np.ravel(b))
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)
