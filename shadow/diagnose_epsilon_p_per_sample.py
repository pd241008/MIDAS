"""
Diagnostic: Why is epsilon_p constant (or zero) across attack steps?

The corrected Check #4 shows epsilon_p = 0.000000 for every step of every
SmoothMockModel run, and epsilon_p = 0.167815 (constant) for every step of
every SmoothMLPModel run. This means the defense is either inactive or
applying a fixed rotation, regardless of what the attacker does.

Root cause hypothesis: the trajectory window is fixed (3 randn vectors),
and the cosine similarity between x_t and the last trajectory vector is
always negative (or always positive), making epsilon_p constant.

This script logs:
1. Per-sample epsilon_p BEFORE any attack (static window only)
2. Per-pair momentum values in the window
3. Per-sample epsilon_p at attack step 0, 10, 50, 99
4. Fraction of samples where epsilon_p changes during the attack
"""
import json
import math
import os
import torch

from shadow.defense import SmoothMockModel, SmoothMLPModel, penetration_epsilon_windowed, momentum
from shadow.scratch.debug_gradients import vulnerable_basis


def project_Lp_ball(x, x0, epsilon):
    diff = x - x0
    diff = torch.clamp(diff, min=-epsilon, max=epsilon)
    return torch.clamp(x0 + diff, min=0.0, max=1.0)


def main():
    torch.manual_seed(42)
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

    # ---- Same pool construction as check4_corrected.py ----
    CALIBRATION_POOL = 200
    N_SAMPLES = 50
    torch.manual_seed(42)
    pool = [torch.rand(D) for _ in range(CALIBRATION_POOL)]
    calibration_batch = torch.stack(pool[:100])

    classifier = SmoothMockModel(d=D, calibration_batch=calibration_batch)

    with torch.no_grad():
        all_probs = torch.stack([classifier(x) for x in pool])
        nat_conf = torch.where(all_probs > 0.5, all_probs, 1.0 - all_probs)
        keep_mask = nat_conf.squeeze() <= 0.95
        filtered = [pool[i] for i in range(len(pool)) if keep_mask[i]]

    dataset = filtered[:N_SAMPLES]
    trajectories = [[torch.rand(D) - 0.5 for _ in range(W - 1)] for _ in range(N_SAMPLES)]

    print("=" * 70)
    print("DIAGNOSTIC: epsilon_p behavior per sample")
    print("=" * 70)

    # ---- Part 1: Static epsilon_p (before attack) ----
    print("\n--- Part 1: Static epsilon_p (trajectory window + dataset point) ---")
    print(f"  {'sample':>6s}  {'eps_p':>8s}  {'pair0_m':>8s}  {'pair1_m':>8s}  {'pair2_m':>8s}  {'pair0_gt0':>9s}  {'pair1_gt0':>9s}  {'pair2_gt0':>9s}")

    static_eps_p_values = []
    static_momentum_values = {0: [], 1: [], 2: []}

    for i in range(N_SAMPLES):
        x0 = dataset[i]
        traj = trajectories[i]
        full_window = traj + [x0]  # window as it would be at attack start

        eps_p = penetration_epsilon_windowed(full_window)
        static_eps_p_values.append(eps_p.item())

        # Per-pair momentum
        pair_momenta = []
        for j in range(len(full_window) - 1):
            m = momentum(full_window[j + 1], full_window[j])
            pair_momenta.append(m.item())
            static_momentum_values[j].append(m.item())

        gt0 = [m > 0 for m in pair_momenta]
        print(f"  {i:6d}  {eps_p.item():8.6f}  {pair_momenta[0]:8.4f}  {pair_momenta[1]:8.4f}  {pair_momenta[2]:8.4f}  "
              f"{'Y' if gt0[0] else 'N':>9s}  {'Y' if gt0[1] else 'N':>9s}  {'Y' if gt0[2] else 'N':>9s}")

    static_eps_p = torch.tensor(static_eps_p_values)
    print(f"\n  Summary:")
    print(f"    Mean eps_p:   {static_eps_p.mean():.6f}")
    print(f"    Median eps_p: {static_eps_p.median():.6f}")
    print(f"    Frac = 0:     {(static_eps_p == 0).float().mean():.2%} ({(static_eps_p == 0).sum()}/{N_SAMPLES})")
    print(f"    Frac > 0:     {(static_eps_p > 0).float().mean():.2%}")
    print(f"    Frac > gamma: {(static_eps_p > config['gamma']).float().mean():.2%}")

    for j in range(3):
        vals = torch.tensor(static_momentum_values[j])
        print(f"    Pair {j} momentum: mean={vals.mean():.4f}  std={vals.std():.4f}  "
              f"frac>0={( vals > 0).float().mean():.2%}  frac<0={(vals < 0).float().mean():.2%}")

    # ---- Part 2: Dynamic epsilon_p during attack (sample 0, eps=0.1, T=100) ----
    print("\n--- Part 2: Dynamic epsilon_p during adaptive attack (sample 0, eps=0.1, T=100) ---")

    x0 = dataset[0]
    traj = trajectories[0]
    with torch.no_grad():
        y_nat = (classifier(x0) > 0.5).float()

    epsilon = 0.1
    alpha = 0.01
    steps = 100

    x_t = x0.clone().detach()
    x_t = x_t + torch.empty_like(x_t).uniform_(-epsilon, epsilon)
    x_t = project_Lp_ball(x_t, x0, epsilon)

    print(f"  {'step':>4s}  {'eps_p':>8s}  {'theta_deg':>10s}  {'pair2_m':>8s}  {'pair2_gt0':>9s}  {'||x_t-x0||':>12s}")

    sliding_window = [w.clone().detach() for w in traj]
    for step in range(steps):
        x_t.requires_grad_(True)
        window_slice = sliding_window[-(W - 1):]
        full_window = window_slice + [x_t]
        eps_p = penetration_epsilon_windowed(full_window)

        # Pair 2 momentum (traj[2] vs x_t)
        m_pair2 = momentum(x_t, traj[2])

        from shadow.defense import rotation_angle
        theta = rotation_angle(eps_p, config["gamma"], config["lambda"], config["k"],
                               config["delta_theta_max_deg"] * math.pi / 180.0)

        import torch.nn.functional as F
        # Use vulnerable_basis for gradient flow
        from shadow.scratch.debug_gradients import vulnerable_basis, midas_defense_forward_vulnerable
        x_def, theta_v = midas_defense_forward_vulnerable(x_t, window_cloned, c_base, config, naive=False)
        pred = classifier(x_def).clamp(1e-7, 1.0 - 1e-7)
        loss = F.binary_cross_entropy(pred, y_nat.expand_as(pred))
        loss.backward()
        grad = x_t.grad

        pert_norm = torch.norm(x_t - x0, p=float('inf')).item()

        if step < 10 or step % 20 == 0 or step == steps - 1:
            print(f"  {step:4d}  {eps_p.item():8.6f}  {math.degrees(theta.item()):10.4f}  "
                  f"{m_pair2.item():8.4f}  {'Y' if m_pair2.item() > 0 else 'N':>9s}  {pert_norm:12.6f}")

        with torch.no_grad():
            x_t = x_t + alpha * torch.sign(grad)
            x_t = project_Lp_ball(x_t, x0, epsilon)
            sliding_window.append(x_t.clone().detach())

    # ---- Part 3: Check if the issue is the [0,1] clamping ----
    print("\n--- Part 3: Effect of [0,1] clamping on cosine similarity ---")
    print("  Checking: how many pool components are negative? (clamped to 0)")

    neg_frac_per_sample = []
    for i in range(N_SAMPLES):
        x0 = dataset[i]
        n_neg = (x0 < 0).float().mean().item()
        neg_frac_per_sample.append(n_neg)

    neg_frac = torch.tensor(neg_frac_per_sample)
    print(f"  Mean fraction of negative components: {neg_frac.mean():.4f}")
    print(f"  Median: {neg_frac.median():.4f}")
    print(f"  Min: {neg_frac.min():.4f}  Max: {neg_frac.max():.4f}")

    # Check: after initial perturbation + [0,1] clamping, what does x_t look like?
    x0 = dataset[0]
    x_t_init = x0.clone().detach()
    x_t_init = x_t_init + torch.empty_like(x_t_init).uniform_(-0.1, 0.1)
    x_t_init = project_Lp_ball(x_t_init, x0, 0.1)

    traj = trajectories[0]
    m_before = momentum(x0, traj[2]).item()
    m_after = momentum(x_t_init, traj[2]).item()

    print(f"\n  Sample 0, traj[2] cosine similarity:")
    print(f"    Before perturbation: {m_before:.6f}")
    print(f"    After perturbation + clamping: {m_after:.6f}")
    print(f"    x0 norm: {torch.norm(x0).item():.6f}")
    print(f"    x_t norm: {torch.norm(x_t_init).item():.6f}")
    print(f"    x0 in [0,1]: {(x0 >= 0).all() and (x0 <= 1).all()}")
    print(f"    x_t in [0,1]: {(x_t_init >= 0).all() and (x_t_init <= 1).all()}")
    print(f"    x0 min: {x0.min():.4f}  max: {x0.max():.4f}")
    print(f"    x_t min: {x_t_init.min():.4f}  max: {x_t_init.max():.4f}")

    # Check ALL samples: what's the cosine similarity of each pool point with its traj[2]?
    print(f"\n--- Part 4: Per-sample cosine similarity between x0 and traj[2] ---")
    cos_sims = []
    for i in range(N_SAMPLES):
        m = momentum(dataset[i], trajectories[i][2])
        cos_sims.append(m.item())
    cos_sims = torch.tensor(cos_sims)
    print(f"  Mean: {cos_sims.mean():.6f}")
    print(f"  Std: {cos_sims.std():.6f}")
    print(f"  Frac > 0: {(cos_sims > 0).float().mean():.2%}")
    print(f"  Frac < 0: {(cos_sims < 0).float().mean():.2%}")
    print(f"  Values: {cos_sims.tolist()}")


if __name__ == "__main__":
    main()
