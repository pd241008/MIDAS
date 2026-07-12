import torch
import pytest
from midas_shadow.defense import midas_defense_forward, MockClassifierLoss

def finite_diff_check(x_t: torch.Tensor, trajectory_window: list[torch.Tensor], c_base: torch.Tensor, basis: torch.Tensor, config: dict, eps=1e-4):
    """
    Central finite-difference cross-check on the end-to-end defense pipeline.
    """
    classifier = MockClassifierLoss()

    # 1. Analytic gradient via autograd
    x_t_ad = x_t.clone().detach().requires_grad_(True)
    
    # We must clone the window and requires_grad=False because we only want grad w.r.t x_t
    window_ad = [w.clone().detach() for w in trajectory_window]
    
    x_defended_ad, _ = midas_defense_forward(x_t_ad, window_ad, c_base, basis, config)
    loss_ad = classifier(x_defended_ad)
    
    grad_ad = torch.autograd.grad(loss_ad, x_t_ad, retain_graph=False)[0]

    # 2. Central difference gradient
    grad_fd = torch.zeros_like(x_t)
    
    for i in range(x_t.shape[0]):
        # +eps
        x_plus = x_t.clone()
        x_plus[i] += eps
        x_defended_plus, _ = midas_defense_forward(x_plus, trajectory_window, c_base, basis, config)
        l_plus = classifier(x_defended_plus)
        
        # -eps
        x_minus = x_t.clone()
        x_minus[i] -= eps
        x_defended_minus, _ = midas_defense_forward(x_minus, trajectory_window, c_base, basis, config)
        l_minus = classifier(x_defended_minus)
        
        grad_fd[i] = (l_plus - l_minus) / (2.0 * eps)
        
    # 3. Compare
    diff = torch.abs(grad_ad - grad_fd)
    
    # Compute relative error, avoid division by zero
    # We'll use max relative error over dimensions where the gradient is not zero
    # or just max absolute error if gradients are tiny
    max_abs_err = torch.max(diff)
    
    # relative error: |ad - fd| / max(|ad|, |fd|)
    denom = torch.max(torch.abs(grad_ad), torch.abs(grad_fd))
    # only compute rel error where denominator is larger than 1e-6
    mask = denom > 1e-6
    if torch.any(mask):
        max_rel_err = torch.max(diff[mask] / denom[mask])
    else:
        max_rel_err = torch.tensor(0.0)

    # We assert that the gradients match.
    # The prompt allows < 1e-2 for relative error. 
    # For tiny gradients, we also check absolute error.
    if max_rel_err > 1e-2 and max_abs_err > 1e-4:
        raise AssertionError(f"Gradient mismatch! Max Rel Err: {max_rel_err.item():.4f}, Max Abs Err: {max_abs_err.item():.4e}\nAutograd: {grad_ad}\nFiniteDiff: {grad_fd}")

    return True

def test_finite_diff():
    torch.manual_seed(42)
    D = 10
    config = {
        "W": 4,
        "D": D,
        "gamma": 0.5,
        "lambda": 1.0,
        "k": 2.0,
        "delta_theta_max_deg": 45.0,
        "tau": 0.3,
        "sla_budget_ms": 10,
        "channel_capacity": 4
    }

    # Synthetic trajectory
    trajectory_window = [torch.randn(D) for _ in range(config["W"] - 1)]
    x_t = torch.randn(D)
    
    # c_base
    c_base = torch.zeros(D)
    
    # basis
    b0 = torch.zeros(D)
    b0[0] = 1.0
    b1 = torch.zeros(D)
    b1[1] = 1.0
    basis = torch.stack([b0, b1])

    assert finite_diff_check(x_t, trajectory_window, c_base, basis, config)
