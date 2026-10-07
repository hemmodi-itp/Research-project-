"""Transformer fusion with learned [MISSING] tokens and auxiliary unimodal heads (Sec. 3.2-3.3, Fig. 1).

forward() = encode each available modality -> fuse(). The two halves are public so the explainers can encode
once and re-fuse many times with modalities swapped for [MISSING] (local Shapley, SHAP, Integrated Gradients).
"""

import torch
import torch.nn as nn

from .. import MODALITIES
from .encoders import ImageEncoder, TabularEncoder, build_text_encoder

MODALITY_KEYS = {
    "image": ("image",),
    "text": ("input_ids", "attention_mask", "text_embeds"),
    "tab": ("tab_num", "tab_flag", "tab_cat"),
}


class MultimodalFusionModel(nn.Module):
    def __init__(self, model_cfg, modalities, n_numeric, cardinalities, vocab_size, max_tokens):
        super().__init__()
        self.modalities = tuple(m for m in MODALITIES if m in modalities)
        d = model_cfg["d_model"]
        self.d = d

        self.encoders = nn.ModuleDict()
        if "image" in self.modalities:
            self.encoders["image"] = ImageEncoder(model_cfg["image_backbone"], model_cfg["image_pretrained"],
                                                  model_cfg["image_grid"])
        if "text" in self.modalities:
            self.encoders["text"] = build_text_encoder(model_cfg, vocab_size, max_tokens)
        if "tab" in self.modalities:
            self.encoders["tab"] = TabularEncoder(n_numeric, cardinalities, d)

        # Project every token to the shared width d.
        self.proj = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(enc.out_dim, d), nn.LayerNorm(d)) for m, enc in self.encoders.items()
        })
        self.aux_heads = nn.ModuleDict({m: nn.Linear(d, 2) for m in self.modalities})

        self.modality_embed = nn.Parameter(torch.randn(len(MODALITIES), d) * 0.02)
        self.missing_token = nn.Parameter(torch.randn(len(MODALITIES), d) * 0.02)
        self.fusion_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, model_cfg["fusion_heads"], 4 * d, model_cfg["dropout"],
                                           activation="gelu", batch_first=True, norm_first=True)
        self.fusion = nn.TransformerEncoder(layer, model_cfg["fusion_layers"], enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2)

        # Post-hoc calibration (Eq. 5): calibrated log-odds = (l1 - l0) / T + bias. bias stays 0 for
        # temperature scaling and is only fitted by the optional Platt variant.
        self.register_buffer("temperature", torch.ones(()))
        self.register_buffer("logit_bias", torch.zeros(()))

    # ------------------------------------------------------------------ encoding
    def encode(self, m, batch):
        """Encode every row of `batch` for modality m -> (tokens (B, n, d), padding mask (B, n))."""
        enc = self.encoders[m]
        if m == "image":
            tokens = enc(batch["image"])
            pad = torch.zeros(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        elif m == "text":
            embeds = batch.get("text_embeds")
            tokens = enc(input_ids=None if embeds is not None else batch["input_ids"],
                         attention_mask=batch["attention_mask"], inputs_embeds=embeds)
            pad = batch["attention_mask"] == 0
        else:
            tokens = enc(batch["tab_num"], batch["tab_flag"], batch["tab_cat"])
            pad = torch.zeros(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        return self.proj[m](tokens), pad

    def encode_present(self, m, batch, present):
        """Run the encoder only on rows where the modality is present; other rows are filled by fuse()."""
        B = present.shape[0]
        idx = present.nonzero(as_tuple=True)[0]
        device = present.device
        if idx.numel() == B:
            return self.encode(m, batch)
        if idx.numel() == 0:
            return torch.zeros(B, 1, self.d, device=device), torch.zeros(B, 1, dtype=torch.bool, device=device)
        sub = {k: batch[k].index_select(0, idx) for k in MODALITY_KEYS[m] if k in batch}
        tokens, pad = self.encode(m, sub)
        full = tokens.new_zeros(B, *tokens.shape[1:]).index_copy(0, idx, tokens)
        full_pad = torch.zeros(B, pad.shape[1], dtype=torch.bool, device=device)
        full_pad[idx] = pad
        return full, full_pad

    # ------------------------------------------------------------------ fusion
    def fuse(self, tokens, present):
        """Eq. 4: joint self-attention over [FUSION] + all modality tokens; returns logits and aux logits."""
        B = tokens[self.modalities[0]][0].shape[0]
        device = self.fusion_token.device
        seq = [self.fusion_token.expand(B, -1, -1)]
        masks = [torch.zeros(B, 1, dtype=torch.bool, device=device)]
        aux = {}
        for m in self.modalities:
            i = MODALITIES.index(m)
            tok, pad = tokens[m]
            missing = ~present[m]
            if missing.any():
                # An absent modality contributes a single learned [MISSING] token.
                tok = torch.where(missing[:, None, None], self.missing_token[i].to(tok.dtype).expand_as(tok), tok)
                only_first = torch.ones_like(pad)
                only_first[:, 0] = False
                pad = torch.where(missing[:, None], only_first, pad)
            keep = (~pad).unsqueeze(-1).to(tok.dtype)
            aux[m] = self.aux_heads[m]((tok * keep).sum(1) / keep.sum(1).clamp(min=1))
            seq.append(tok + self.modality_embed[i])
            masks.append(pad)
        x = self.fusion(torch.cat(seq, dim=1), src_key_padding_mask=torch.cat(masks, dim=1))
        z = self.norm(x[:, 0])
        return {"logits": self.head(z), "aux_logits": aux, "z_fused": z}

    def forward(self, batch, present=None, tokens=None):
        """`present` maps modality -> bool (B,) tensor; defaults to the batch's own presence flags.
        `tokens` may hold pre-computed encodings for some modalities (used by the explainers)."""
        if present is None:
            present = {m: batch[f"present_{m}"].bool() for m in self.modalities}
        tokens = dict(tokens or {})
        for m in self.modalities:
            if m not in tokens:
                tokens[m] = self.encode_present(m, batch, present[m])
        out = self.fuse(tokens, present)
        out["present"] = present
        return out

    # ------------------------------------------------------------------ outputs
    def log_odds(self, logits):
        return (logits[:, 1] - logits[:, 0]) / self.temperature + self.logit_bias

    def predict_proba(self, logits):
        return torch.sigmoid(self.log_odds(logits))

    def parameter_groups(self, lr_encoders, lr_fusion):
        """Pretrained encoders get the small learning rate; everything initialised from scratch the larger one."""
        pretrained, scratch = [], []
        for m, enc in self.encoders.items():
            (pretrained if enc.pretrained else scratch).extend(enc.parameters())
        enc_ids = {id(p) for enc in self.encoders.values() for p in enc.parameters()}
        scratch.extend(p for p in self.parameters() if id(p) not in enc_ids)
        groups = [{"params": scratch, "lr": lr_fusion}]
        if pretrained:
            groups.append({"params": pretrained, "lr": lr_encoders})
        return groups
