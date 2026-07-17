"""
Diagnostic: What magnitude does delta_theta actually reach during attacks?

Logs per-step delta_theta (degrees) and epsilon_p for both SmoothMockModel
and SmoothMLPModel on the vulnerable basis, eps=0.1, T=100.
"""
import json
import math
import os
import torch
import torch.nn.functional as F

from shadow.defense import SmoothMockModel, SmoothMLPModel, penetration_epsilon_windowed, rotation_angle
from shadow.scratch.debug_gradients import vulnerable_basis


def project_Lp_ball(x, x0, epsilon):
    diff = x - x0
    diff = torch.clamp(diff, min=-epsilon, max=epsilon)
    return torch.clamp(x0 + diff, min=0.0, max=1.0)


def midas_defense_forward_vulnerable_with_eps_p(x_t, trajectory_window, c_base, config, naive=False):
    """Same as vulnerable forward pass but also returns epsilon_p."""
    gamma = config.get("gamma", 0.292)
    lambda_ = config.get("lambda", 1.0)
    k = config.get("k", 2.0)
    delta_theta_max = config.get("delta_theta_max_deg", 45.0) * (3.141592653589793 / 180.0)

    full_window = trajectory_window + [x_t]
    epsilon_p = penetration_epsilon_windowed(full_window)
    theta = rotation_angle(epsilon_p, gamma, lambda_, k, delta_theta_max)

    if naive:
        theta = theta.detach()
        basis = vulnerable_basis(x_t.detach(), c_base)
    else:
        basis = vulnerable_basis(x_t, c_base)

    from shadow.defense import rotate_manifold_fixed_basis
    x_defended = rotate_manifold_fixed_basis(x_t, basis, theta)
    return x_defended, theta, epsilon_p


def pgd_with_delta_theta_logging(x0_init, y_target, classifier, traj, c_base, config,
                                  alpha, epsilon, steps, naive, label=""):
    """PGD with per-step delta_theta and epsilon_p logging."""
    x_t = x0_init.clone().detach()
    x_t = x_t + torch.empty_like(x_t).uniform_(-epsilon, epsilon)
    x_t = project_Lp_ball(x_t, x0_init, epsilon)

    print(f"\n  {'='*60}")
    print(f"  {label}")
    print(f"  {'='*60}")
    print(f"  {'step':>4s}  {'theta_deg':>10s}  {'theta_rad':>10s}  {'eps_p':>10s}  {'loss':>10s}  {'||dL/dx||':>10s}  {'||dL/dθ||':>10s}")

    step_data = []

    for step in range(steps):
        x_t.requires_grad_(True)
        window_cloned = [w.clone().detach() for w in traj]
        x_def, theta, eps_p = midas_defense_forward_vulnerable_with_eps_p(
            x_t, window_cloned, c_base, config, naive=naive
        )
        if not naive:
            theta.retain_grad()

        pred = classifier(x_def).clamp(1e-7, 1.0 - 1e-7)
        loss = F.binary_cross_entropy(pred, y_target.expand_as(pred))
        loss.backward()
        grad = x_t.grad

        grad_theta_norm = 0.0
        if not naive and theta.grad is not None:
            grad_theta_norm = torch.norm(theta.grad).item()

        theta_val = theta.item()
        theta_deg = math.degrees(theta_val)
        eps_p_val = eps_p.item()

        step_data.append({
            "step": step,
            "theta_deg": theta_deg,
            "theta_rad": theta_val,
            "eps_p": eps_p_val,
            "loss": loss.item(),
            "grad_x_norm": torch.norm(grad).item(),
            "grad_theta_norm": grad_theta_norm,
        })

        if step < 30 or step % 10 == 0 or step == steps - 1:
            print(f"  {step:4d}  {theta_deg:10.4f}  {theta_val:10.6f}  "
                  f"{eps_p_val:10.6f}  {loss.item():10.6f}  "
                  f"{torch.norm(grad).item():10.6f}  {grad_theta_norm:10.8f}")

        with torch.no_grad():
            x_t = x_t + alpha * torch.sign(grad)
            x_t = project_Lp_ball(x_t, x0_init, epsilon)

    theta_vals = [d["theta_deg"] for d in step_data]
    eps_p_vals = [d["eps_p"] for d in step_data]
    print(f"\n  Summary: theta_deg min={min(theta_vals):.4f}  max={max(theta_vals):.4f}  "
          f"mean={sum(theta_vals)/len(theta_vals):.4f}")
    print(f"           eps_p   min={min(eps_p_vals):.6f}  max={max(eps_p_vals):.6f}  "
          f"mean={sum(eps_p_vals)/len(eps_p_vals):.6f}")

    return step_data


def make_dataset(D, seed=42):
    CALIBRATION_POOL = 200
    N_SAMPLES = 50
    torch.manual_seed(seed)
    pool = [torch.rand(D) for _ in range(CALIBRATION_POOL)]
    calibration_batch = torch.stack(pool[:100])
    return pool, calibration_batch


def main():
    config = {
        "W": 4, "D": 10, "gamma": 0.292, "lambda": 1.0, "k": 2.0,
        "delta_theta_max_deg": 45.0, "tau": 0.3,
    }
    D = config["D"]

    if not os.path.exists("results/basis.json"):
        raise RuntimeError("results/basis.json not found!")
    with open("results/basis.json") as f:
        basis_data = json.load(f)
    c_base = torch.tensor(basis_data["c_base"], dtype=torch.float32)

    gamma = config["gamma"]
    lambda_ = config["lambda"]
    delta_theta_max_deg = config["delta_theta_max_deg"]
    delta_theta_max_rad = math.radians(delta_theta_max_deg)

    print("=" * 70)
    print("DELTA_THETA DIAGNOSTIC")
    print("=" * 70)
    print(f"\n  Config values:")
    print(f"    gamma             = {gamma}")
    print(f"    lambda            = {lambda_}")
    print(f"    k                 = {config['k']}")
    print(f"    delta_theta_max   = {delta_theta_max_deg} deg = {delta_theta_max_rad:.4f} rad")
    print(f"\n  Rotation angle function:")
    print(f"    Archimedean (eps_p <= {gamma}): theta = {lambda_} * eps_p")
    print(f"    Logarithmic (eps_p > {gamma}):  theta = min({lambda_} * exp({config['k']} * (eps_p - {gamma})), {delta_theta_max_deg} deg)")

    # Archimedean phase: theta at gamma boundary
    theta_at_gamma = lambda_ * gamma
    print(f"\n  theta at gamma boundary (eps_p={gamma}): {theta_at_gamma:.4f} rad = {math.degrees(theta_at_gamma):.4f} deg")
    # What eps_p is needed for theta = 1 deg?
    eps_p_for_1deg = 1.0 / lambda_  # Archimedean: theta = lambda * eps_p => eps_p = theta / lambda
    print(f"  eps_p needed for 1 deg in Archimedean phase: {eps_p_for_1deg:.4f}")

    # ================================================================
    # SMOOTH MOCK MODEL (linear)
    # ================================================================
    print("\n" + "=" * 70)
    print("SMOOTH MOCK MODEL (linear)")
    print("=" * 70)

    pool, calibration_batch = make_dataset(D)
    classifier = SmoothMockModel(d=D, calibration_batch=calibration_batch)

    with torch.no_grad():
        all_probs = torch.stack([classifier(x) for x in pool])
        nat_conf = torch.where(all_probs > 0.5, all_probs, 1.0 - all_probs)
        keep_mask = nat_conf.squeeze() <= 0.95
        filtered = [pool[i] for i in range(len(pool)) if keep_mask[i]]

    dataset = filtered[:50]
    trajectories = [[torch.rand(D) - 0.5 for _ in range(config["W"] - 1)] for _ in range(50)]

    # Sample 0, eps=0.1, T=100
    x0 = dataset[0]
    traj = trajectories[0]
    with torch.no_grad():
        y_nat = (classifier(x0) > 0.5).float()

    mock_naive = pgd_with_delta_theta_logging(
        x0, y_nat, classifier, traj, c_base, config,
        alpha=0.01, epsilon=0.1, steps=100, naive=True,
        label="SmoothMockModel — NAIVE — eps=0.1, T=100, alpha=0.01"
    )

    mock_adapt = pgd_with_delta_theta_logging(
        x0, y_nat, classifier, traj, c_base, config,
        alpha=0.01, epsilon=0.1, steps=100, naive=False,
        label="SmoothMockModel — ADAPTIVE — eps=0.1, T=100, alpha=0.01"
    )

    # ================================================================
    # SMOOTH MLP MODEL (16-unit tanh)
    # ================================================================
    print("\n" + "=" * 70)
    print("SMOOTH MLP MODEL (16-unit tanh)")
    print("=" * 70)

    mlp_classifier = SmoothMLPModel(d=D, hidden=16, calibration_batch=calibration_batch, seed=1337)

    mlp_naive = pgd_with_delta_theta_logging(
        x0, y_nat, mlp_classifier, traj, c_base, config,
        alpha=0.01, epsilon=0.1, steps=100, naive=True,
        label="SmoothMLPModel — NAIVE — eps=0.1, T=100, alpha=0.01"
    )

    mlp_adapt = pgd_with_delta_theta_logging(
        x0, y_nat, mlp_classifier, traj, c_base, config,
        alpha=0.01, epsilon=0.1, steps=100, naive=False,
        label="SmoothMLPModel — ADAPTIVE — eps=0.1, T=100, alpha=0.01"
    )

    # ================================================================
    # COMPARISON SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n  {'Surrogate':<25s} {'Attacker':<12s} {'theta_min':>10s} {'theta_max':>10s} {'theta_mean':>10s} {'eps_p_mean':>10s}")
    print(f"  {'-'*25} {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for name, data in [("SmoothMockModel", mock_naive), ("SmoothMockModel", mock_adapt),
                       ("SmoothMLPModel", mlp_naive), ("SmoothMLPModel", mlp_adapt)]:
        tag = "naive" if data is mock_naive or data is mlp_naive else "adaptive"
        if data is mock_adapt:
            tag = "adaptive"
        if data is mlp_naive:
            tag = "naive"
        if data is mlp_adapt:
            tag = "adaptive"

        theta_degs = [d["theta_deg"] for d in data]
        eps_ps = [d["eps_p"] for d in data]
        print(f"  {name:<25s} {tag:<12s} {min(theta_degs):10.4f} {max(theta_degs):10.4f} "
              f"{sum(theta_degs)/len(theta_degs):10.4f} {sum(eps_ps)/len(eps_ps):10.6f}")

    print(f"\n  Configured ceiling: delta_theta_max = {delta_theta_max_deg} deg")
    print(f"  gamma = {gamma}, lambda = {lambda_}")
    print(f"  Archimedean phase ceiling (at eps_p=gamma): theta = {math.degrees(lambda_ * gamma):.4f} deg")


if __name__ == "__main__":
    main()
