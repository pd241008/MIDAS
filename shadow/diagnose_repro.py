"""
Minimal diagnostic: reproduce EXACTLY the check4_corrected.py setup
and check epsilon_p for sample 0 vs what the trace showed.
"""
import json, math, os, torch
from shadow.defense import SmoothMockModel, SmoothMLPModel, penetration_epsilon_windowed, momentum, rotation_angle
from shadow.scratch.debug_gradients import vulnerable_basis

def main():
    D = 10
    W = 4

    config = {
        "W": W, "D": D, "gamma": 0.292, "lambda": 1.0, "k": 2.0,
        "delta_theta_max_deg": 45.0, "tau": 0.3,
    }

    if not os.path.exists("results/basis.json"):
        raise RuntimeError("results/basis.json not found!")
    with open("results/basis.json") as f:
        basis_data = json.load(f)
    c_base = torch.tensor(basis_data["c_base"], dtype=torch.float32)

    # EXACT reproduction of check4_corrected.py main()
    torch.manual_seed(42)
    CALIBRATION_POOL = 200
    N_SAMPLES = 50
    pool = [torch.rand(D) for _ in range(CALIBRATION_POOL)]
    calibration_batch = torch.stack(pool[:100])

    classifier_mock = SmoothMockModel(d=D, calibration_batch=calibration_batch)

    # EXACT reproduction of check4_corrected.py run_check4_for_classifier
    with torch.no_grad():
        all_probs = torch.stack([classifier_mock(x) for x in pool])
        nat_conf = torch.where(all_probs > 0.5, all_probs, 1.0 - all_probs)
        keep_mask = nat_conf.squeeze() <= 0.95
        filtered = [pool[i] for i in range(len(pool)) if keep_mask[i]]

    dataset = filtered[:N_SAMPLES]
    trajectories = [[torch.rand(D) - 0.5 for _ in range(config["W"] - 1)] for _ in range(N_SAMPLES)]

    print("=" * 70)
    print("REPRODUCTION: epsilon_p per sample (EXACT check4_corrected.py setup)")
    print("=" * 70)

    for i in range(min(10, N_SAMPLES)):
        x0 = dataset[i]
        traj = trajectories[i]
        full_window = traj + [x0]

        eps_p = penetration_epsilon_windowed(full_window)
        pair_m = []
        for j in range(len(full_window) - 1):
            m = momentum(full_window[j + 1], full_window[j])
            pair_m.append(m.item())

        print(f"  Sample {i:2d}: eps_p={eps_p.item():.6f}  "
              f"pair0={pair_m[0]:.4f}  pair1={pair_m[1]:.4f}  pair2={pair_m[2]:.4f}")

    # Now simulate one attack step for sample 0 to see what happens
    print("\n" + "=" * 70)
    print("SIMULATING ATTACK STEP 0 for sample 0 (eps=0.1)")
    print("=" * 70)

    import torch.nn.functional as F

    x0 = dataset[0]
    traj = trajectories[0]
    with torch.no_grad():
        y_nat = (classifier_mock(x0) > 0.5).float()

    epsilon = 0.1
    x_t = x0.clone().detach()
    x_t = x_t + torch.empty_like(x_t).uniform_(-epsilon, epsilon)

    # BEFORE project_Lp_ball
    full_before = traj + [x_t]
    eps_p_before = penetration_epsilon_windowed(full_before)
    m_pair2_before = momentum(x_t, traj[2])
    print(f"  Before project: eps_p={eps_p_before.item():.6f}  pair2_m={m_pair2_before.item():.4f}")

    # AFTER project_Lp_ball
    def project_Lp_ball(x, x0, epsilon):
        diff = x - x0
        diff = torch.clamp(diff, min=-epsilon, max=epsilon)
        return torch.clamp(x0 + diff, min=0.0, max=1.0)

    x_t = project_Lp_ball(x_t, x0, epsilon)

    full_after = traj + [x_t]
    eps_p_after = penetration_epsilon_windowed(full_after)
    m_pair2_after = momentum(x_t, traj[2])
    print(f"  After project:  eps_p={eps_p_after.item():.6f}  pair2_m={m_pair2_after.item():.4f}")
    print(f"  x0 range: [{x0.min():.4f}, {x0.max():.4f}]")
    print(f"  x_t range: [{x_t.min():.4f}, {x_t.max():.4f}]")
    print(f"  ||x_t - x0||_inf = {torch.norm(x_t - x0, p=float('inf')).item():.6f}")

    # Now do one PGD step
    x_t.requires_grad_(True)
    window_cloned = [w.clone().detach() for w in traj]
    full_window = window_cloned + [x_t]
    eps_p_step = penetration_epsilon_windowed(full_window)
    theta = rotation_angle(eps_p_step, config["gamma"], config["lambda"], config["k"],
                           config["delta_theta_max_deg"] * math.pi / 180.0)
    basis = vulnerable_basis(x_t, c_base)
    from shadow.defense import rotate_manifold_fixed_basis
    x_def = rotate_manifold_fixed_basis(x_t, basis, theta)
    pred = classifier_mock(x_def).clamp(1e-7, 1.0 - 1e-7)
    loss = F.binary_cross_entropy(pred, y_nat.expand_as(pred))
    loss.backward()
    grad = x_t.grad

    print(f"\n  After PGD step 0:")
    print(f"    eps_p = {eps_p_step.item():.6f}")
    print(f"    theta = {math.degrees(theta.item()):.4f} deg")
    print(f"    loss = {loss.item():.6f}")
    print(f"    ||grad|| = {torch.norm(grad).item():.6f}")

    # Check: what if we use the EXACT same setup as check4_corrected.py pgd_attack?
    print("\n" + "=" * 70)
    print("EXACT REPRODUCTION of check4_corrected.py pgd_attack step 0")
    print("=" * 70)

    x_t2 = x0.clone().detach()
    x_t2 = x_t2 + torch.empty_like(x_t2).uniform_(-epsilon, epsilon)
    x_t2 = project_Lp_ball(x_t2, x0, epsilon)

    x_t2.requires_grad_(True)
    window_cloned2 = [w.clone().detach() for w in traj]

    # This is EXACTLY what check4_corrected.py does:
    full_window2 = window_cloned2 + [x_t2]
    eps_p2 = penetration_epsilon_windowed(full_window2)
    theta2 = rotation_angle(eps_p2, config["gamma"], config["lambda"], config["k"],
                            config["delta_theta_max_deg"] * math.pi / 180.0)
    print(f"  eps_p = {eps_p2.item():.6f}")
    print(f"    theta = {math.degrees(theta2.item()):.4f} deg")

    basis2 = vulnerable_basis(x_t2, c_base)
    x_def2 = rotate_manifold_fixed_basis(x_t2, basis2, theta2)
    pred2 = classifier_mock(x_def2).clamp(1e-7, 1.0 - 1e-7)
    loss2 = F.binary_cross_entropy(pred2, y_nat.expand_as(pred2))
    loss2.backward()

    print(f"    loss = {loss2.item():.6f}")
    print(f"    ||grad|| = {torch.norm(x_t2.grad).item():.6f}")

    # Now check: what's the actual x_def?
    print(f"\n  x_t2 range: [{x_t2.min():.4f}, {x_t2.max():.4f}]")
    print(f"  x_def2 range: [{x_def2.min():.4f}, {x_def2.max():.4f}]")
    print(f"  pred2 = {pred2.item():.6f}")
    print(f"  y_nat = {y_nat.item():.6f}")

    # Check the actual output of midas_defense_forward_vulnerable
    from shadow.scratch.debug_gradients import midas_defense_forward_vulnerable
    x_def3, theta3 = midas_defense_forward_vulnerable(x_t2, window_cloned2, c_base, config, naive=False)
    print(f"\n  via midas_defense_forward_vulnerable:")
    print(f"    theta3 = {math.degrees(theta3.item()):.4f} deg")
    print(f"    eps_p = {penetration_epsilon_windowed(window_cloned2 + [x_t2]).item():.6f}")


if __name__ == "__main__":
    main()
