import torch
import torch.nn as nn
import os
import json

def momentum(v_t: torch.Tensor, v_prev: torch.Tensor) -> torch.Tensor:
    """
    Computes cosine similarity between v_t and v_prev.
    Matches Rust edge-core/src/trajectory.rs `compute_momentum`.
    """
    if v_t.shape != v_prev.shape:
        return torch.tensor(0.0, device=v_t.device)

    norm_t = torch.norm(v_t, p=2)
    norm_tm1 = torch.norm(v_prev, p=2)

    if norm_t < 1e-7 or norm_tm1 < 1e-7:
        return torch.tensor(0.0, device=v_t.device)

    dot = torch.sum(v_t * v_prev)
    m = dot / (norm_t * norm_tm1)
    return torch.clamp(m, min=-1.0, max=1.0)


def penetration_epsilon_windowed(window: list[torch.Tensor]) -> torch.Tensor:
    """
    epsilon_p = (1 / |W|) * sum_{i in W} ||v_i||_2 * max(0, M_i)
    Matches Rust edge-core/src/trajectory.rs `penetration_epsilon_windowed`.
    """
    if len(window) < 2:
        return torch.tensor(0.0, device=window[0].device if window else torch.device('cpu'))

    sum_val = torch.tensor(0.0, device=window[0].device)
    pairs = 0

    for i in range(len(window) - 1):
        v_prev = window[i]
        v_cur = window[i + 1]
        m = momentum(v_cur, v_prev)
        norm_cur = torch.norm(v_cur, p=2)
        # max(0, m)
        m_clamped = torch.clamp(m, min=0.0)
        sum_val = sum_val + norm_cur * m_clamped
        pairs += 1

    return sum_val / pairs


def rotation_angle(epsilon_p: torch.Tensor, gamma: float, lambda_: float, k: float, delta_theta_max: float) -> torch.Tensor:
    """
    Matches Rust edge-core/src/rotation.rs `rotation_angle`.
    Archimedean: lambda * epsilon_p
    Logarithmic: min(lambda * exp(k * (epsilon_p - gamma)), delta_theta_max)
    """
    if epsilon_p <= gamma:
        return lambda_ * epsilon_p
    else:
        val = lambda_ * torch.exp(k * (epsilon_p - gamma))
        return torch.clamp(val, max=delta_theta_max)


def rotate_manifold_fixed_basis(x: torch.Tensor, basis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Matches Rust edge-core/src/rotation.rs `rotate_manifold_givens_fixed_basis`.
    basis is a [2, D] tensor containing the two orthonormal PCA vectors.
    """
    if x.shape[0] < 2 or basis.shape[0] < 2:
        return x.clone()

    c = torch.cos(theta)
    s = torch.sin(theta)

    b0 = basis[0]
    b1 = basis[1]

    coord0 = torch.sum(x * b0)
    coord1 = torch.sum(x * b1)

    rot0 = c * coord0 - s * coord1
    rot1 = s * coord0 + c * coord1

    rotated = x.clone()
    rotated = rotated + (rot0 - coord0) * b0 + (rot1 - coord1) * b1
    return rotated


def project_to_manifold(x_adv: torch.Tensor, c_base: torch.Tensor) -> torch.Tensor:
    """
    Matches Rust edge-core/src/rotation.rs `project_to_manifold`.
    """
    if x_adv.shape != c_base.shape:
        return x_adv.clone()

    diff = torch.norm(x_adv - c_base, p=2)
    if diff <= 1e-7:
        return x_adv.clone()

    return x_adv - 0.5 * (x_adv - c_base)


def midas_defense_forward(x_t: torch.Tensor, trajectory_window: list[torch.Tensor], c_base: torch.Tensor, basis: torch.Tensor, config: dict, naive: bool = False):
    """
    The end-to-end differentiable forward pass of the MIDAS-Edge defense.
    `trajectory_window` includes previous x vectors up to x_{t-1}.
    Returns (x_defended, theta) so we can inspect gradients on theta.
    """
    gamma = config.get("gamma", 0.292)
    lambda_ = config.get("lambda", 1.0)
    k = config.get("k", 2.0)
    delta_theta_max_deg = config.get("delta_theta_max_deg", 45.0)
    delta_theta_max = delta_theta_max_deg * (3.141592653589793 / 180.0)

    full_window = trajectory_window + [x_t]

    epsilon_p = penetration_epsilon_windowed(full_window)
    theta = rotation_angle(epsilon_p, gamma, lambda_, k, delta_theta_max)
    
    if naive:
        theta = theta.detach()
        
    x_defended = rotate_manifold_fixed_basis(x_t, basis, theta)

    return x_defended, theta


class MockClassifierLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Matches Rust `MockModel`: label = (sum(x).abs() as u32) % 10. 
        The Rust naive finite-difference attacker uses the label directly as a continuous scalar for gradients.
        To mirror this exactly, we return fmod(abs(sum(x)), 10.0) as the continuous loss target.
        """
        s = torch.abs(torch.sum(x))
        return torch.fmod(s, 10.0)

class SmoothMockModel(torch.nn.Module):
    """
    Fixed-weight linear classifier used ONLY as a differentiable attack surrogate.
    Not a replacement for Rust's MockModel — that stays as-is for pipeline tests.

    bias b is calibrated against a representative batch so that ~half the points
    sit near the decision boundary, rather than drawn randomly (which pushes
    nearly every point to near-certain confidence at any dimensionality).
    """
    def __init__(self, d, calibration_batch, seed=1337):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w = torch.randn(d, generator=g)
        with torch.no_grad():
            scores = calibration_batch @ self.w
            self.b = scores.median().item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x @ self.w - self.b)


class SmoothMLPModel(torch.nn.Module):
    """
    Small, FIXED-WEIGHT (never trained) 2-layer MLP used only as an attack surrogate.
    Introduces the local curvature a linear model structurally lacks, without
    reopening the reproducibility concerns of a trainable model.

    bias on the final layer is calibrated the same way as SmoothMockModel:
    median of pre-activation scores over the calibration batch, so ~half the
    pool sits near the decision boundary.
    """
    def __init__(self, d, hidden=16, calibration_batch=None, seed=1337):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.fc1 = torch.nn.Linear(d, hidden)
        self.fc2 = torch.nn.Linear(hidden, 1)
        with torch.no_grad():
            torch.nn.init.normal_(self.fc1.weight, generator=g)
            torch.nn.init.normal_(self.fc1.bias, generator=g)
            torch.nn.init.normal_(self.fc2.weight, generator=g)
            if calibration_batch is not None:
                pre_scores = torch.tanh(self.fc1(calibration_batch)) @ self.fc2.weight.T
                self.fc2.bias.data.fill_(-pre_scores.median())
            else:
                self.fc2.bias.data.fill_(0.0)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.fc1(x))
        return torch.sigmoid(self.fc2(h)).squeeze(-1)


class TrainedSurrogateModel(nn.Module):
    """
    Small feedforward classifier, briefly trained on synthetic separable data
    to obtain real decision-boundary curvature. Used ONLY as an attack surrogate
    in the adaptive-attacker harness.

    All weights are frozen after training (requires_grad_(False)).
    """
    def __init__(self, d, hidden=16):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc2(torch.tanh(self.fc1(x)))).squeeze(-1)


def generate_synthetic_data(n_samples=1000, d=10, seed=42):
    """
    Generate two-class synthetic data with real feature correlations.

    Class 0: samples from N(mu0, Sigma) where Sigma has off-diagonal correlations.
    Class 1: samples from N(mu1, Sigma) with a rotated mean.
    The separating boundary is non-axis-aligned due to both the correlation
    structure and the mean rotation.
    """
    rng = torch.Generator().manual_seed(seed)

    # Shared covariance with off-diagonal correlations
    # Start with identity, add structured correlations
    A = torch.randn(d, d, generator=rng) * 0.3
    Sigma = torch.eye(d) + A @ A.T
    # Scale so diagonal is 1
    D_diag = torch.sqrt(torch.diag(Sigma))
    Sigma = Sigma / (D_diag.unsqueeze(0) * D_diag.unsqueeze(1))

    # Class means: separated along a random direction, not axis-aligned
    separation_dir = torch.randn(d, generator=rng)
    separation_dir = separation_dir / torch.norm(separation_dir)
    mu0 = -0.5 * separation_dir
    mu1 = 0.5 * separation_dir

    # Add per-class feature correlations via a random linear transform
    W_transform = torch.randn(d, d, generator=rng) * 0.2 + torch.eye(d)

    n_per_class = n_samples // 2
    L = torch.linalg.cholesky(Sigma)

    x0 = torch.randn(n_per_class, d, generator=rng) @ L.T + mu0
    x1 = torch.randn(n_per_class, d, generator=rng) @ L.T + mu1

    # Apply nonlinear feature correlation
    x0 = x0 @ W_transform.T
    x1 = x1 @ W_transform.T

    X = torch.cat([x0, x1], dim=0)
    y = torch.cat([torch.zeros(n_per_class), torch.ones(n_per_class)])

    # Shuffle
    perm = torch.randperm(n_samples, generator=rng)
    X = X[perm]
    y = y[perm]

    return X, y


def train_surrogate(d=10, hidden=16, seed=42, n_epochs=500, lr=0.01,
                    n_train=1000, save_dir=None):
    """
    Train a TrainedSurrogateModel on synthetic data, freeze weights, optionally save.
    Returns (model, training_log).
    """
    torch.manual_seed(seed)
    X, y = generate_synthetic_data(n_samples=n_train, d=d, seed=seed)

    model = TrainedSurrogateModel(d, hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    log = {"losses": [], "accuracies": []}
    for epoch in range(n_epochs):
        model.train()
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            acc = ((pred > 0.5).float() == y).float().mean().item()
        log["losses"].append(loss.item())
        log["accuracies"].append(acc)

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1:4d}: loss={loss.item():.4f}  acc={acc:.2%}")

    # Freeze all weights
    for p in model.parameters():
        p.requires_grad_(False)

    final_acc = log["accuracies"][-1]
    print(f"  Training complete: final accuracy = {final_acc:.2%}")

    # Save to disk
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        weights_path = os.path.join(save_dir, "trained_surrogate.pt")
        torch.save(model.state_dict(), weights_path)
        meta = {
            "d": d, "hidden": hidden, "seed": seed,
            "n_epochs": n_epochs, "lr": lr, "n_train": n_train,
            "final_accuracy": final_acc,
            "arch": "TrainedSurrogateModel",
        }
        meta_path = os.path.join(save_dir, "trained_surrogate_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  Saved to {weights_path}")
        print(f"  Meta saved to {meta_path}")

    return model, log


def load_surrogate(d=10, hidden=16, weights_path=None):
    """Load a pre-trained TrainedSurrogateModel from disk."""
    model = TrainedSurrogateModel(d, hidden=hidden)
    if weights_path is None:
        weights_path = os.path.join(os.path.dirname(__file__), "trained_surrogate.pt")
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    for p in model.parameters():
        p.requires_grad_(False)
    return model
