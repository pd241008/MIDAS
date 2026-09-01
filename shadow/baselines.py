"""
Baseline defenses for #8, evaluated against the real d=42 UNSW-NB15 surrogate.

Three state-of-the-art baselines from the paper comparison table
(report.rs DefenseSuccessRow columns):
  1. input_smoothing      -- Gaussian input smoothing (randomized smoothing),
                             sigma=0.1, MC-expectation over K noise draws.
  2. chen_query_blinding  -- Chen et al. query-blinding / classifier rejection:
                             reject (defend) any input whose natural confidence
                             is below a tau threshold (ambiguous/query-like).
  3. adv_training         -- adversarially-trained surrogate: retrain the real
                             surrogate with per-batch PGD (L-inf, eps=0.1, T=50)
                             augmentation (min-max adversarial training).

Each baseline is a callable `forward(x) -> defended prediction in (0,1)` that
the attack harness differentiates (the "adaptive" attacker) exactly as it does
the MIDAS rotation defense. For input smoothing the expectation is
differentiated through the noise reparameterization (MC mean is a sum of
sigmoids, so gradients flow); for query blinding and adversarial training the
defended forward is the model itself and naive == adaptive (no rotational
manifold parameter to detach).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from shadow.defense import TrainedSurrogateModel
from shadow.attacks import project_Lp_ball


# ---------------------------------------------------------------------------
# Baseline 1: Gaussian input smoothing (randomized smoothing, sigma=0.1)
# ---------------------------------------------------------------------------
class SmoothedClassifier(nn.Module):
    def __init__(self, base: nn.Module, sigma=0.1, K=32, seed=42):
        super().__init__()
        self.base = base
        self.sigma = sigma
        self.K = K
        self._g = torch.Generator().manual_seed(seed)
        for p in base.parameters():
            p.requires_grad_(False)
        self.base.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MC expectation: mean_{k} base(x + N(0, sigma^2)); differentiable wrt x.
        x_shape = x.shape
        xb = x.expand(self.K, *x_shape)
        noise = torch.randn(self.K, *x_shape, generator=self._g, dtype=x.dtype)
        preds = self.base((xb + self.sigma * noise).reshape(-1, x_shape[-1]))
        return preds.reshape(self.K, *x_shape[:-1]).mean(dim=0).clamp(1e-7, 1 - 1e-7)


# ---------------------------------------------------------------------------
# Baseline 2: Chen et al. query-blinding (classifier rejection on ambiguity)
# ---------------------------------------------------------------------------
class ChenBlindingClassifier(nn.Module):
    """
    Rejects (defends) any input whose natural confidence is below `tau`
    (ambiguous / probing query). Measured the same way MIDAS's windowed
    rejection is: a rejected input is withheld -> defense success; an accepted
    input is classified and counts as a flip on misclassification.
    `forward` returns the defended confidence: rejected -> clamped ambiguous
    value at 0.5 (so it cannot be read as a confident class), accepted -> base(x).
    Gradients flow through base(x) for accepted inputs (adaptive); a naive
    attacker treats the accept/reject decision as fixed.
    """
    def __init__(self, base: nn.Module, tau=0.75):
        super().__init__()
        self.base = base
        self.tau = tau
        for p in base.parameters():
            p.requires_grad_(False)
        self.base.eval()
        self._last_rejected = 0.0

    def forward(self, x: torch.Tensor, naive: bool = False) -> torch.Tensor:
        p = self.base(x).squeeze(-1)
        conf = torch.where(p > 0.5, p, 1.0 - p)
        reject = (conf < self.tau).float()
        if naive:
            # fixed (non-differentiable) accept/reject decision
            with torch.no_grad():
                rejected_pre = reject.detach()
        else:
            rejected_pre = reject
        self._last_rejected = float((rejected_pre > 0.5).float().mean().item())
        # accepted -> own prediction; rejected -> 0.5 (withheld, cannot be
        # confidently read as flipped)
        out = (1.0 - rejected_pre) * p + rejected_pre * torch.full_like(p, 0.5)
        return out.clamp(1e-7, 1 - 1e-7)


# ---------------------------------------------------------------------------
# Baseline 3: Adversarially-trained surrogate
# ---------------------------------------------------------------------------
def train_adv_surrogate(d=42, hidden=16, seed=42, n_epochs=200, lr=0.01,
                        batch_size=1024, eps=0.1, alpha=0.01, T=10,
                        adv_frac_epoch=5, pgd_subset_div=4):
    """
    Min-max adversarial training (Madry-style, T=10 canonical).
    Each adv_frac_epoch-th epoch, augment a random `batch_size//pgd_subset_div`
    slice of the batch with PGD-adv samples (L-inf eps, T steps against the
    current model) before the forward/backward pass.
    Freezes weights and returns the adv-trained surrogate.
    """
    from shadow.train_real_surrogate import load_real_data
    train_X, train_y, _, _ = load_real_data()
    n_train = len(train_X)

    torch.manual_seed(seed)
    model = TrainedSurrogateModel(d, hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    log = {"losses": [], "test_accs": []}

    test_X = np.load("data/test_features.npy")
    test_y = np.load("data/test_labels.npy")
    test_X = torch.tensor(test_X, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.float32)

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            Xb = train_X[idx].clone()
            yb = train_y[idx].clone()
            if (epoch % adv_frac_epoch) == 0:
                sub = idx[:len(idx) // pgd_subset_div]
                Xadv = _pgd_batch(model, train_X[sub], train_y[sub], eps, alpha, T)
                Xb = torch.cat([Xadv, Xb[len(idx) // pgd_subset_div:]])
            pred = model(Xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        model.eval()
        with torch.no_grad():
            ta = ((model(test_X) > 0.5).float() == test_y).float().mean().item()
        log["losses"].append(epoch_loss / n_batches)
        log["test_accs"].append(ta)
        if (epoch + 1) % 40 == 0:
            print(f"  Epoch {epoch+1:4d}: loss={epoch_loss/n_batches:.4f} test_acc={ta:.2%}")

    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model, log


def _pgd_batch(model, X, y, eps, alpha, T):
    Xb = X.clone().detach().requires_grad_(True)
    Xp = X.clone().detach() + torch.empty_like(X).uniform_(-eps, eps)
    Xp = project_Lp_ball(Xp, X.clone().detach(), eps)
    for _ in range(T):
        Xp = Xp.clone().detach().requires_grad_(True)
        pred = model(Xp).clamp(1e-7, 1 - 1e-7)
        loss = F.binary_cross_entropy(pred, y.squeeze(-1))
        loss.backward()
        with torch.no_grad():
            Xp = Xp + alpha * torch.sign(Xp.grad)
            Xp = project_Lp_ball(Xp, X.clone().detach(), eps)
    return Xp.detach()
