import torch
from shadow.defense import *

def vulnerable_basis(x_t: torch.Tensor, c_base: torch.Tensor) -> torch.Tensor:
    """
    Computes an attacker-coupled basis dynamically using x_t.
    b0 is aligned with x_t (or x_t - c_base).
    """
    dim = x_t.shape[0]
    b0 = x_t - c_base
    n0 = torch.norm(b0, p=2)
    if n0 > 1e-7:
        b0 = b0 / n0
    else:
        b0 = torch.zeros_like(x_t)
        b0[0] = 1.0
        
    # Gram-schmidt a standard vector for b1
    b1 = torch.zeros_like(x_t)
    if dim > 1:
        b1[1] = 1.0
    else:
        b1[0] = 1.0
        
    dot = torch.sum(b1 * b0)
    b1 = b1 - dot * b0
    n1 = torch.norm(b1, p=2)
    if n1 > 1e-7:
        b1 = b1 / n1
        
    return torch.stack([b0, b1])

def midas_defense_forward_vulnerable(x_t: torch.Tensor, trajectory_window: list[torch.Tensor], c_base: torch.Tensor, config: dict, naive: bool = False) -> torch.Tensor:
    gamma = config.get("gamma", 1.05)
    lambda_ = config.get("lambda", 1.0)
    k = config.get("k", 2.0)
    delta_theta_max = config.get("delta_theta_max_deg", 45.0) * (3.141592653589793 / 180.0)

    full_window = trajectory_window + [x_t]
    epsilon_p = penetration_epsilon_windowed(full_window)
    theta = rotation_angle(epsilon_p, gamma, lambda_, k, delta_theta_max)
    
    # Enable gradient tracking for theta to inspect it
    # We will compute gradients w.r.t theta using a hook or by returning it
    
    if naive:
        theta = theta.detach()
        basis = vulnerable_basis(x_t.detach(), c_base) # Naive attacker also treats the basis as fixed
    else:
        basis = vulnerable_basis(x_t, c_base)
        
    x_defended = rotate_manifold_fixed_basis(x_t, basis, theta)
    return x_defended, theta

def debug():
    torch.manual_seed(42)
    D = 10
    config = {"W": 4, "D": D, "gamma": 1.05, "lambda": 1.0, "k": 2.0, "delta_theta_max_deg": 45.0}
    c_base = torch.zeros(D)
    window = [torch.randn(D) for _ in range(config["W"] - 1)]
    x_t = torch.randn(D, requires_grad=True)
    
    classifier = MockClassifierLoss()
    
    print("--- Fixed Basis ---")
    basis = vulnerable_basis(torch.ones(D), c_base) # just some fixed basis
    x_def, _ = midas_defense_forward_vulnerable(x_t, window, c_base, config, naive=True) # use naive=True to simulate fixed basis for now
    loss = classifier(x_def)
    grad_x = torch.autograd.grad(loss, x_t, retain_graph=True)[0]
    print(f"Loss: {loss.item()}")
    print(f"Grad norm (fixed basis): {torch.norm(grad_x).item()}")
    
    print("\n--- Vulnerable Basis ---")
    x_t_vuln = x_t.clone().detach().requires_grad_(True)
    x_def_v, theta = midas_defense_forward_vulnerable(x_t_vuln, window, c_base, config, naive=False)
    
    loss_v = classifier(x_def_v)
    # We can inspect the grad of theta and x_t
    loss_v.backward()
    
    print(f"Loss: {loss_v.item()}")
    print(f"Grad norm (vulnerable basis x_t): {torch.norm(x_t_vuln.grad).item()}")
    
if __name__ == "__main__":
    debug()
