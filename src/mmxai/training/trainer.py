"""Training loop (Sec. 3.4): weighted cross-entropy + auxiliary heads, modality dropout, AdamW with cosine decay,
early stopping on validation AUROC."""

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from ..data.dataset import to_device


def class_weights(labels, n_classes=2):
    """w_k = N / (K * N_k) (Eq. 6)."""
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    return torch.tensor(len(labels) / (n_classes * np.maximum(counts, 1)), dtype=torch.float32)


def modality_dropout(present, p, generator=None):
    """Drop each available modality with probability p, but never every modality of a sample.

    If every available modality of a sample was dropped, one of them (chosen at random) is put back.
    """
    if p <= 0 or len(present) < 2:
        return present
    names = list(present)
    orig = torch.stack([present[m] for m in names], 1)  # (B, M)
    kept = orig & (torch.rand(orig.shape, generator=generator, device=orig.device) >= p)
    none_left = ~kept.any(1)
    scores = torch.rand(orig.shape, generator=generator, device=orig.device).masked_fill(~orig, -1.0)
    rescue = torch.zeros_like(orig).scatter_(1, scores.argmax(1, keepdim=True), True)
    kept = torch.where(none_left[:, None], rescue & orig, kept)
    return {m: kept[:, j] for j, m in enumerate(names)}


def multimodal_loss(out, y, weight, aux_weight):
    """Eq. 7: L = CE(y, y_hat) + lambda * sum_m CE(y, y_hat^(m)) over modalities available for each sample."""
    loss = F.cross_entropy(out["logits"].float(), y, weight=weight)
    aux_terms = []
    for m, logits in out["aux_logits"].items():
        mask = out["present"][m]
        if mask.any():
            aux_terms.append(F.cross_entropy(logits[mask].float(), y[mask], weight=weight))
    if aux_terms:
        loss = loss + aux_weight * torch.stack(aux_terms).sum()
    return loss


@torch.no_grad()
def predict(model, loader, device, force_absent=()):
    """Run the model over a loader. `force_absent` withholds modalities at test time (robustness, Shapley)."""
    model.eval()
    logits, aux, present_all, index = [], {m: [] for m in model.modalities}, {m: [] for m in model.modalities}, []
    for batch in loader:
        batch = to_device(batch, device)
        present = {m: batch[f"present_{m}"].bool() & (m not in force_absent) for m in model.modalities}
        out = model(batch, present=present)
        logits.append(out["logits"].float().cpu())
        for m in model.modalities:
            aux[m].append(torch.softmax(out["aux_logits"][m].float(), -1)[:, 1].cpu())
            present_all[m].append(present[m].cpu())
        index.append(batch["index"].cpu())
    logits = torch.cat(logits)
    return {
        "logits": logits.numpy(),
        "prob": model.predict_proba(logits.double().to(model.temperature.device)).cpu().numpy(),
        "aux_prob": {m: torch.cat(v).numpy() for m, v in aux.items()},
        "present": {m: torch.cat(v).numpy() for m, v in present_all.items()},
        "index": torch.cat(index).numpy(),
    }


def train_model(model, train_loader, val_loader, train_labels, cfg, device, log):
    tcfg, mcfg = cfg["train"], cfg["model"]
    model.to(device)
    weight = class_weights(train_labels).to(device)
    opt = torch.optim.AdamW(model.parameter_groups(tcfg["lr_encoders"], tcfg["lr_fusion"]),
                            weight_decay=tcfg["weight_decay"])
    total_steps = max(1, tcfg["max_epochs"] * len(train_loader))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    use_amp = bool(tcfg["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_auc, best_state, best_epoch, bad_epochs, history = -1.0, None, -1, 0, []
    for epoch in range(tcfg["max_epochs"]):
        model.train()
        running, n = 0.0, 0
        for batch in train_loader:
            batch = to_device(batch, device)
            present = {m: batch[f"present_{m}"].bool() for m in model.modalities}
            present = modality_dropout(present, mcfg["modality_dropout"])
            with torch.autocast(device.type, enabled=use_amp):
                out = model(batch, present=present)
                loss = multimodal_loss(out, batch["label"], weight, mcfg["aux_weight"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item() * len(batch["label"])
            n += len(batch["label"])

        val = predict(model, val_loader, device)
        y_val = val_loader.dataset.df["label"].to_numpy()[val["index"]]
        val_auc = roc_auc_score(y_val, val["prob"]) if len(np.unique(y_val)) > 1 else float("nan")
        history.append({"epoch": epoch + 1, "train_loss": running / max(n, 1), "val_auroc": val_auc})
        log.info(f"  epoch {epoch + 1:2d}  loss {running / max(n, 1):.4f}  val AUROC {val_auc:.4f}")
        if val_auc > best_auc + 1e-4:
            best_auc, best_epoch, bad_epochs = val_auc, epoch + 1, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad_epochs += 1
            if bad_epochs >= tcfg["patience"]:
                log.info(f"  early stop at epoch {epoch + 1}; best epoch {best_epoch}")
                break
    model.load_state_dict(best_state)
    return {"best_val_auroc": best_auc, "best_epoch": best_epoch, "history": history}
