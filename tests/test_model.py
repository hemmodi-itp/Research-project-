import numpy as np
import pandas as pd
import torch

from mmxai.data.preprocess import TabularPreprocessor, normalize_text
from mmxai.models.fusion import MultimodalFusionModel
from mmxai.training.calibration import fit_calibration
from mmxai.training.trainer import modality_dropout, multimodal_loss

CFG = {"image_backbone": "resnet18", "image_pretrained": False, "image_grid": 2, "text_encoder": "simple",
       "d_model": 32, "fusion_layers": 2, "fusion_heads": 4, "dropout": 0.0}


def _batch(B=4, seq=8):
    g = torch.Generator().manual_seed(0)
    return {
        "image": torch.randn(B, 3, 64, 64, generator=g),
        "input_ids": torch.randint(4, 20, (B, seq), generator=g),
        "attention_mask": torch.ones(B, seq, dtype=torch.long),
        "tab_num": torch.randn(B, 3, generator=g),
        "tab_flag": torch.zeros(B, 3),
        "tab_cat": torch.ones(B, 2, dtype=torch.long),
        "present_image": torch.tensor([True, False, True, True]),
        "present_text": torch.tensor([True, True, False, True]),
        "present_tab": torch.ones(B, dtype=torch.bool),
        "label": torch.tensor([0, 1, 0, 1]),
    }


def _model():
    torch.manual_seed(0)
    return MultimodalFusionModel(CFG, ("image", "text", "tab"), n_numeric=3, cardinalities=[2, 3],
                                 vocab_size=20, max_tokens=8).eval()


def test_forward_shapes_and_missing_inputs_are_ignored():
    model, batch = _model(), _batch()
    out = model(batch)
    assert out["logits"].shape == (4, 2)
    assert set(out["aux_logits"]) == {"image", "text", "tab"}
    # Changing the pixels of a sample whose image is absent must not change its prediction.
    batch2 = {k: v.clone() for k, v in batch.items()}
    batch2["image"][1] = torch.randn(3, 64, 64)
    batch2["image"][0] = torch.randn(3, 64, 64)
    with torch.no_grad():
        a, b = model(batch)["logits"], model(batch2)["logits"]
    assert torch.allclose(a[1], b[1], atol=1e-6)      # image absent -> unaffected
    assert not torch.allclose(a[0], b[0], atol=1e-6)  # image present -> affected


def test_partial_encoding_matches_full_encoding():
    model, batch = _model(), _batch()
    with torch.no_grad():
        full = model(batch, present={m: torch.ones(4, dtype=torch.bool) for m in model.modalities})["logits"]
        tokens = {m: model.encode(m, batch) for m in model.modalities}
        cached = model.fuse(tokens, {m: torch.ones(4, dtype=torch.bool) for m in model.modalities})["logits"]
    assert torch.allclose(full, cached, atol=1e-5)


def test_modality_dropout_never_drops_everything():
    torch.manual_seed(0)
    present = {m: torch.ones(5000, dtype=torch.bool) for m in ("image", "text", "tab")}
    kept = modality_dropout(present, 0.9)
    assert torch.stack(list(kept.values())).any(0).all()
    # P(kept) = 0.1 + P(all dropped) / 3 = 0.1 + 0.729 / 3
    assert abs(kept["image"].float().mean() - (0.1 + 0.729 / 3)) < 0.03
    # A modality that was never present is never "rescued".
    present["text"][:] = False
    assert not modality_dropout(present, 0.9)["text"].any()


def test_loss_and_backward():
    model, batch = _model().train(), _batch()
    out = model(batch)
    loss = multimodal_loss(out, batch["label"], torch.tensor([0.6, 2.5]), 0.1)
    loss.backward()
    assert model.missing_token.grad is not None and torch.isfinite(loss)


def test_temperature_scaling_recovers_overconfidence():
    rng = np.random.default_rng(0)
    true = rng.normal(0, 1.5, 4000)
    y = (rng.random(4000) < 1 / (1 + np.exp(-true))).astype(float)
    logits = np.stack([np.zeros_like(true), 3 * true], 1)  # three times too confident
    T, b = fit_calibration(logits, y, "temperature")
    assert abs(T - 3) < 0.4 and b == 0


def test_preprocessor_is_fitted_on_train_only():
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, np.nan, 1000.0], "c": ["a", "b", "a", None, "b"]})
    pre = TabularPreprocessor(["x"], ["c"], valid_ranges={"x": [0, 500]}).fit(train)
    t = pre.transform(pd.DataFrame({"x": [np.nan, 2.0, 9999.0], "c": ["b", "zzz", None]}))
    assert t["flag"][:, 0].tolist() == [1, 0, 1]          # 9999 fails the range check -> missing
    assert t["num"][0, 0] == t["num"][2, 0] == pre.impute_z[0]
    assert t["cat"][:, 0].tolist() == [2, 0, 0]            # unseen and missing -> 0
    assert normalize_text("SOB, Fever") == "shortness of breath fever"


def test_temperature_is_bounded_when_logits_are_offset():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.15).astype(float)
    diff = 50 + 5 * y + rng.normal(0, 1, 500)  # ranks well, but every case is called positive
    T, _ = fit_calibration(np.stack([np.zeros(500), diff], 1), y, "temperature")
    assert T == 20.0
    T, b = fit_calibration(np.stack([np.zeros(500), diff], 1), y, "platt")
    assert b < -1  # Platt's bias term can undo the offset
