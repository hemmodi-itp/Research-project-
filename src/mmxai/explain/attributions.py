"""Per-prediction explanations at three levels (Sec. 4.3, Fig. 5).

Every attribution targets the fused model's calibrated log-odds of pneumonia for one encounter:
  - image : Grad-CAM on the last ResNet stage (layer4) [9]
  - text  : Integrated Gradients over chief-complaint token embeddings [20]
  - tab   : SHAP values (interventional, permutation estimator [10]) for each structured feature
  - local modality Shapley values over the 8 subsets, using the learned [MISSING] token for absent modalities
All functions take a single-encounter batch (batch size 1) already on the model's device.
"""

from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F

from .. import MODALITIES
from ..evaluation.shapley import shapley_values


def _present(model, batch, override=None):
    present = {m: batch[f"present_{m}"].bool() for m in model.modalities}
    for m, v in (override or {}).items():
        present[m] = torch.full_like(present[m], v)
    return present


# ----------------------------------------------------------------------------- image
def grad_cam(model, batch):
    """Grad-CAM heat map (H, W) in [0, 1] for the positive-class log-odds."""
    layer = model.encoders["image"].backbone[-1]
    store = {}

    def hook(_, __, output):
        store["act"] = output
        output.register_hook(lambda g: store.__setitem__("grad", g))

    handle = layer.register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            out = model(batch)
            model.log_odds(out["logits"]).sum().backward()
    finally:
        handle.remove()
    act, grad = store["act"].detach(), store["grad"].detach()
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=batch["image"].shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    return (cam / cam.max().clamp(min=1e-8)).cpu().numpy()


# ----------------------------------------------------------------------------- text
def integrated_gradients_text(model, batch, tokenizer, steps=50):
    """IG per token, baseline = [PAD] embeddings at non-special positions.

    Returns (tokens, scores, positions) for the non-special, non-padding tokens.
    """
    enc = model.encoders["text"]
    ids = batch["input_ids"]
    mask = batch["attention_mask"]
    embed = enc.get_input_embeddings()
    with torch.no_grad():
        x = embed(ids)
        special = torch.tensor([[int(t) in tokenizer.special_ids for t in ids[0]]], device=ids.device)
        baseline_ids = torch.where(special, ids, torch.full_like(ids, tokenizer.pad_token_id))
        x0 = embed(baseline_ids)
        # The other modalities do not change along the path, so encode them once.
        present = _present(model, batch)
        cached = {m: model.encode_present(m, batch, present[m]) for m in model.modalities if m != "text"}

    alphas = torch.linspace(1 / steps, 1, steps, device=x.device).view(-1, 1, 1)  # right Riemann sum
    path = (x0 + alphas * (x - x0)).detach().requires_grad_(True)
    tokens = {m: (t.expand(steps, *t.shape[1:]), p.expand(steps, *p.shape[1:])) for m, (t, p) in cached.items()}
    sub = {"text_embeds": path, "attention_mask": mask.expand(steps, -1)}
    with torch.enable_grad():
        tokens["text"] = model.encode("text", sub)
        pres = {m: v.expand(steps) for m, v in present.items()}
        out = model.fuse(tokens, pres)
        grads = torch.autograd.grad(model.log_odds(out["logits"]).sum(), path)[0]
    scores = ((x - x0)[0] * grads.mean(0)).sum(-1)
    keep = mask[0].bool() & ~special[0]
    toks = tokenizer.convert_ids_to_tokens(ids[0][keep].tolist())
    return toks, scores[keep].detach().cpu().numpy(), keep.nonzero(as_tuple=True)[0].cpu().numpy()


# ----------------------------------------------------------------------------- structured
class TabularValueFunction:
    """f(rows) -> calibrated log-odds for one encounter, with its structured features replaced by `rows`.

    A row is a vector over features: numeric z-values (NaN = missing) followed by categorical codes. Image and
    text are encoded once and reused, so thousands of evaluations stay cheap.
    """

    def __init__(self, model, batch, pre):
        self.model, self.batch = model, batch
        self.n_num = len(pre.numeric)
        self.impute = torch.tensor(pre.impute_z, device=batch["tab_num"].device)
        self.present = _present(model, batch)
        with torch.no_grad():
            self.cached = {m: model.encode_present(m, batch, self.present[m]) for m in model.modalities if m != "tab"}

    def encounter_row(self):
        num = self.batch["tab_num"][0].clone()
        num[self.batch["tab_flag"][0].bool()] = float("nan")
        return torch.cat([num, self.batch["tab_cat"][0].float()]).cpu().numpy()

    @torch.no_grad()
    def __call__(self, rows, chunk=512):
        out = []
        for start in range(0, len(rows), chunk):
            r = torch.as_tensor(rows[start:start + chunk], dtype=torch.float32, device=self.impute.device)
            n = len(r)
            num = r[:, : self.n_num]
            flag = torch.isnan(num)
            tab = {"tab_num": torch.where(flag, self.impute.expand_as(num), num), "tab_flag": flag.float(),
                   "tab_cat": r[:, self.n_num:].round().long()}
            tokens = {m: (t.expand(n, *t.shape[1:]), p.expand(n, *p.shape[1:])) for m, (t, p) in self.cached.items()}
            tokens["tab"] = self.model.encode("tab", tab)
            present = {m: v.expand(n) for m, v in self.present.items()}
            out.append(self.model.log_odds(self.model.fuse(tokens, present)["logits"]).float().cpu())
        return torch.cat(out).numpy()


def permutation_shap(f, x, background, n_permutations=8, seed=0):
    """Interventional SHAP values by antithetic permutation sampling (the estimator of shap's PermutationExplainer).

    For each permutation and background row, features are switched from the background value to the encounter's
    value one at a time; each feature's marginal effect is averaged. Values sum (in expectation, and exactly per
    permutation) to f(x) - mean f(background).
    """
    rng = np.random.default_rng(seed)
    n_feat = len(x)
    perms = []
    for _ in range(max(1, n_permutations // 2)):
        p = rng.permutation(n_feat)
        perms += [p, p[::-1]]
    rows = []
    for perm in perms:
        for b in background:
            cur = b.copy()
            rows.append(cur.copy())
            for j in perm:
                cur[j] = x[j]
                rows.append(cur.copy())
    values = f(np.array(rows)).reshape(len(perms), len(background), n_feat + 1)
    phi = np.zeros(n_feat)
    for k, perm in enumerate(perms):
        deltas = np.diff(values[k], axis=1).mean(0)  # average over background rows
        phi[perm] += deltas
    phi /= len(perms)
    base = values[:, :, 0].mean()
    return phi, float(base)


# ----------------------------------------------------------------------------- modality level
@torch.no_grad()
def local_modality_shapley(model, batch, base_value):
    """Shapley value of each modality for one prediction, in calibrated log-odds.

    v(S) = log-odds with only S supplied (others -> [MISSING]); v(empty) = `base_value`, the mean validation
    log-odds (the same reference SHAP uses), because an all-missing input is never seen in training.
    """
    present = _present(model, batch)
    tokens = {m: model.encode_present(m, batch, present[m]) for m in model.modalities}
    available = [m for m in model.modalities if bool(present[m][0])]
    value = {frozenset(): base_value}
    for r in range(1, len(available) + 1):
        for S in combinations(available, r):
            pres = {m: present[m] & (m in S) for m in model.modalities}
            value[frozenset(S)] = float(model.log_odds(model.fuse(tokens, pres)["logits"])[0])
    phi = shapley_values(available, value)
    return {m: phi.get(m, 0.0) for m in model.modalities}, value[frozenset(available)]
