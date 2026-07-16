import torch
import pytest
from shadow.defense import midas_defense_forward, MockClassifierLoss, SmoothMLPModel


def finite_diff_check(classifier, x_t, trajectory_window, c_base, basis, config, eps=1e-4):
    """
    Central finite-difference cross-check on the end-to-end defense pipeline
    through an arbitrary classifier (SmoothMockModel, SmoothMLPModel, etc.).
    """
    # 1. Analytic gradient via autograd
    x_t_ad = x_t.clone().detach().requires_grad_(True)

    window_ad = [w.clone().detach() for w in trajectory_window]

    x_defended_ad, _ = midas_defense_forward(x_t_ad, window_ad, c_base, basis, config)
    loss_ad = classifier(x_defended_ad)

    grad_ad = torch.autograd.grad(loss_ad, x_t_ad, retain_graph=False)[0]

    # 2. Central difference gradient
    grad_fd = torch.zeros_like(x_t)

    for i in range(x_t.shape[0]):
        x_plus = x_t.clone()
        x_plus[i] += eps
        x_defended_plus, _ = midas_defense_forward(x_plus, trajectory_window, c_base, basis, config)
        l_plus = classifier(x_defended_plus)

        x_minus = x_t.clone()
        x_minus[i] -= eps
        x_defended_minus, _ = midas_defense_forward(x_minus, trajectory_window, c_base, basis, config)
        l_minus = classifier(x_defended_minus)

        grad_fd[i] = (l_plus - l_minus) / (2.0 * eps)

    # 3. Compare
    diff = torch.abs(grad_ad - grad_fd)
    max_abs_err = torch.max(diff)

    denom = torch.max(torch.abs(grad_ad), torch.abs(grad_fd))
    mask = denom > 1e-6
    if torch.any(mask):
        max_rel_err = torch.max(diff[mask] / denom[mask])
    else:
        max_rel_err = torch.tensor(0.0)

    if max_rel_err > 1e-2 and max_abs_err > 1e-4:
        raise AssertionError(
            f"Gradient mismatch! Max Rel Err: {max_rel_err.item():.4f}, "
            f"Max Abs Err: {max_abs_err.item():.4e}\n"
            f"Autograd: {grad_ad}\nFiniteDiff: {grad_fd}"
        )

    return True


def _make_config():
    return {
        "W": 4, "D": 10, "gamma": 0.5, "lambda": 1.0, "k": 2.0,
        "delta_theta_max_deg": 45.0, "tau": 0.3,
        "sla_budget_ms": 10, "channel_capacity": 4,
    }


def _make_basis(D):
    b0 = torch.zeros(D); b0[0] = 1.0
    b1 = torch.zeros(D); b1[1] = 1.0
    return torch.stack([b0, b1])


def test_finite_diff_linear():
    torch.manual_seed(42)
    D = 10
    config = _make_config()
    trajectory_window = [torch.randn(D) for _ in range(config["W"] - 1)]
    x_t = torch.randn(D)
    c_base = torch.zeros(D)
    basis = _make_basis(D)

    classifier = MockClassifierLoss()
    assert finite_diff_check(classifier, x_t, trajectory_window, c_base, basis, config)


def test_finite_diff_mlp():
    torch.manual_seed(42)
    D = 10
    config = _make_config()
    trajectory_window = [torch.randn(D) for _ in range(config["W"] - 1)]
    x_t = torch.randn(D)
    c_base = torch.zeros(D)
    basis = _make_basis(D)

    cal_batch = torch.randn(100, D)
    classifier = SmoothMLPModel(d=D, hidden=16, calibration_batch=cal_batch, seed=1337)
    assert finite_diff_check(classifier, x_t, trajectory_window, c_base, basis, config)
