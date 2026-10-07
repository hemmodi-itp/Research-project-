"""Modality encoders f_img, f_txt, f_tab (Eqs. 1-3). Each returns a sequence of tokens."""

import math

import torch
import torch.nn as nn
import torchvision


class ImageEncoder(nn.Module):
    """ResNet trunk whose final feature map is pooled to a grid x grid set of region tokens.

    For ResNet-50 at 384 x 384 the last map is 12 x 12; pooling to 6 x 6 gives the 36 tokens of Table 3.
    `self.backbone[-1]` (layer4) is the Grad-CAM target layer.
    """

    pretrained = True

    def __init__(self, name="resnet50", pretrained=True, grid=6):
        super().__init__()
        net = getattr(torchvision.models, name)(weights="DEFAULT" if pretrained else None)
        self.backbone = nn.Sequential(*list(net.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(grid)
        self.out_dim = net.fc.in_features
        self.pretrained = pretrained

    def forward(self, x):
        fmap = self.backbone(x)
        return self.pool(fmap).flatten(2).transpose(1, 2)  # (B, grid*grid, C)


class HFTextEncoder(nn.Module):
    """Bio_ClinicalBERT (or any Hugging Face encoder). All token states are passed on, not just [CLS]."""

    pretrained = True

    def __init__(self, name):
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(name)
        self.out_dim = self.model.config.hidden_size

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds)
        return out.last_hidden_state


class SimpleTextEncoder(nn.Module):
    """Small randomly initialised transformer; used only for offline smoke tests."""

    pretrained = False

    def __init__(self, vocab_size, dim=128, max_len=128, layers=1, heads=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, 2 * dim, 0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out_dim = dim

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        x = inputs_embeds if inputs_embeds is not None else self.embed(input_ids)
        x = x + self.pos[:, : x.size(1)]
        return self.encoder(x, src_key_padding_mask=attention_mask == 0)


class TabularEncoder(nn.Module):
    """FT-Transformer feature tokeniser: one token per structured feature.

    Numeric feature j with value x_j and missingness flag m_j -> x_j * W_j + m_j * U_j + b_j.
    Categorical feature j with code c -> E_j[c] + b_j (code 0 = missing/unseen).
    """

    pretrained = False

    def __init__(self, n_numeric, cardinalities, d):
        super().__init__()
        bound = 1 / math.sqrt(d)
        self.num_weight = nn.Parameter(torch.empty(n_numeric, d).uniform_(-bound, bound))
        self.flag_weight = nn.Parameter(torch.empty(n_numeric, d).uniform_(-bound, bound))
        self.num_bias = nn.Parameter(torch.empty(n_numeric, d).uniform_(-bound, bound))
        self.cat_embed = nn.ModuleList(nn.Embedding(c + 1, d) for c in cardinalities)
        self.cat_bias = nn.Parameter(torch.empty(len(cardinalities), d).uniform_(-bound, bound))
        self.n_numeric = n_numeric
        self.out_dim = d

    def forward(self, x_num, x_flag, x_cat):
        tokens = []
        if self.n_numeric:
            tokens.append(x_num.unsqueeze(-1) * self.num_weight + x_flag.unsqueeze(-1) * self.flag_weight
                          + self.num_bias)
        if len(self.cat_embed):
            cat = torch.stack([emb(x_cat[:, j]) for j, emb in enumerate(self.cat_embed)], dim=1)
            tokens.append(cat + self.cat_bias)
        return torch.cat(tokens, dim=1)


def build_text_encoder(model_cfg, vocab_size, max_tokens):
    if model_cfg["text_encoder"] == "simple":
        return SimpleTextEncoder(vocab_size, max_len=max_tokens)
    return HFTextEncoder(model_cfg["text_encoder"])
