"""Post-hoc calibration on the validation set (Eq. 5, Guo et al. [19])."""

import math

import torch
import torch.nn.functional as F

# A temperature outside this range means the logits are offset rather than over- or under-confident; scaling them
# towards 0.5 would erase the ranking, so T is clipped and the caller warns (use "platt" in that case).
T_MIN, T_MAX = 0.05, 20.0


def fit_calibration(logits, labels, method="temperature"):
    """Fit T (and, for Platt, a bias b) so that sigmoid((l1 - l0) / T + b) minimises validation NLL.

    Temperature scaling only rescales confidence. Class-weighted training also shifts the base rate, which a
    temperature cannot undo; "platt" adds the bias term for that case. Returns (T, b).
    """
    logits = torch.as_tensor(logits, dtype=torch.float64)
    y = torch.as_tensor(labels, dtype=torch.float64)
    diff = logits[:, 1] - logits[:, 0]
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=method == "platt")
    params = [log_t] + ([bias] if method == "platt" else [])
    opt = torch.optim.LBFGS(params, lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(diff / log_t.exp() + bias, y)
        loss.backward()
        return loss

    opt.step(closure)
    T = float(log_t.detach().exp())
    if not math.isfinite(T):
        T = T_MAX
    return min(max(T, T_MIN), T_MAX), float(bias.detach())
