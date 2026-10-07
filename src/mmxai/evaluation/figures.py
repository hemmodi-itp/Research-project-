"""Static figures for the paper (Figs. 3, 4 and 6): AUROC with 95% bootstrap CIs."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Reference palette (light mode): categorical slots in fixed order, recessive chrome, text in ink tokens.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def _style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=26)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(ax.get_xlim()[1], 0.5, " chance", va="center", ha="left", color=MUTED, fontsize=8)


def _ylo(lows):
    """Bars start at the axis floor, which sits below every CI and below chance."""
    finite = [v for v in lows if np.isfinite(v)]
    return min(0.45, min(finite) - 0.03) if finite else 0.45


def _legend(ax, handles, labels):
    """Legend on its own row between the title and the plot, so it never covers a bar or a label."""
    ax.legend(handles, labels, frameon=False, fontsize=8, labelcolor=INK_2, ncols=len(labels),
              loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.2, handlelength=1.2)


def _errorbars(ax, x, mid, lo, hi):
    yerr = np.vstack([np.asarray(mid) - np.asarray(lo), np.asarray(hi) - np.asarray(mid)])
    ax.errorbar(x, mid, yerr=yerr, fmt="none", ecolor=INK_2, elinewidth=1.2, capsize=3)


def _value_labels(ax, x, mid, hi):
    for xi, m, h in zip(x, mid, hi):
        ax.text(xi, (h if np.isfinite(h) else m) + 0.012, f"{m:.2f}", ha="center", va="bottom", color=INK_2, fontsize=8)


def plot_baselines(table, path):
    """Fig. 3: single-modality versus multimodal models (bars coloured by number of modalities)."""
    t = table.reset_index(drop=True)
    n_mod = t["inputs"].str.split(" ").str[0].str.count(r"\+") + 1
    colors = [SERIES[0] if k == 1 else SERIES[1] for k in n_mod]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(t))
    ylo = _ylo(t["auroc_ci_low"])
    ax.bar(x, t["auroc"] - ylo, bottom=ylo, width=0.6, color=colors, edgecolor=SURFACE, linewidth=2)
    _errorbars(ax, x, t["auroc"], t["auroc_ci_low"], t["auroc_ci_high"])
    _value_labels(ax, x, t["auroc"], t["auroc_ci_high"])
    ax.set_xticks(x, [f"{m}\n{i}" for m, i in zip(t["model"], t["inputs"].str.replace(" (XGBoost)", "\nXGBoost"))])
    ax.set_ylim(ylo, 1.04)  # headroom for value labels
    _style(ax, "Test AUROC by model (95% patient-level bootstrap CI)", "AUROC")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SERIES[:2]]
    _legend(ax, handles, ["Single modality", "Multimodal"])
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_ablation(table, path):
    """Fig. 4: every modality subset, retrained, sorted by AUROC."""
    t = table.sort_values("auroc_retrained").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(t))
    ylo = _ylo(t["ci_low"])
    ax.bar(x, t["auroc_retrained"] - ylo, bottom=ylo, width=0.6, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    _errorbars(ax, x, t["auroc_retrained"], t["ci_low"], t["ci_high"])
    _value_labels(ax, x, t["auroc_retrained"], t["ci_high"])
    ax.set_xticks(x, t["subset"])
    ax.set_ylim(ylo, 1.04)  # headroom for value labels
    _style(ax, "Ablation: test AUROC for each modality subset (I image, T text, S structured)", "AUROC")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_robustness(table, path):
    """Fig. 6: fused model with a modality withheld versus a specialist retrained without it."""
    t = table.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(t))
    w = 0.36
    lows = list(t["fused_ci_low"]) + (list(t["specialist_ci_low"]) if "specialist_ci_low" in t else [])
    ylo = _ylo(lows)
    ax.bar(x - w / 2, t["auroc_fused"] - ylo, bottom=ylo, width=w, color=SERIES[0], edgecolor=SURFACE, linewidth=2,
           label="Fused model, modality withheld")
    _errorbars(ax, x - w / 2, t["auroc_fused"], t["fused_ci_low"], t["fused_ci_high"])
    _value_labels(ax, x - w / 2, t["auroc_fused"], t["fused_ci_high"])
    if "auroc_specialist" in t:
        has = t["auroc_specialist"].notna().to_numpy()
        ax.bar(x[has] + w / 2, t.loc[has, "auroc_specialist"] - ylo, bottom=ylo, width=w, color=SERIES[1],
               edgecolor=SURFACE, linewidth=2, label="Specialist retrained without it")
        _errorbars(ax, x[has] + w / 2, t.loc[has, "auroc_specialist"], t.loc[has, "specialist_ci_low"],
                   t.loc[has, "specialist_ci_high"])
        _value_labels(ax, x[has] + w / 2, t.loc[has, "auroc_specialist"], t.loc[has, "specialist_ci_high"])
    ax.set_xticks(x, [s.replace(": ", "\n") for s in t["scenario"]])
    ax.set_ylim(ylo, 1.04)  # headroom for value labels
    _style(ax, "Missing-modality robustness: test AUROC (95% CI)", "AUROC")
    _legend(ax, *ax.get_legend_handles_labels())
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
