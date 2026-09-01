"""
Continuous-gradient PGD sweep — secondary check for Check #4.
Uses alpha * grad / ||grad|| instead of alpha * sign(grad).
Same 9 configs, both vulnerable and fixed bases, N=200.
"""
import sys, os, json, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shadow.defense import (
    TrainedSurrogateModel, midas_defense_forward,
    penetration_epsilon_windowed, rotate_manifold_fixed_basis,
)
from shadow.scratch.debug_gradients import midas_defense_forward_vulnerable
from shadow.attacks import project_Lp_ball

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shadow", "trained_weights")

D = 42
W = 10
GAMMA = 2.3061  # benign P95 (recalibrated)
N_SAMPLES = 200

CONFIG = {"W": W, "D": D, "gamma": GAMMA, "lambda": 1.0, "k": 2.0,
          "delta_theta_max_deg": 45.0, "tau": 0.75, "sla_budget_ms": 10, "channel_capacity": 4}


def load_real_surrogate():
    model = TrainedSurrogateModel(D, hidden=16)
    model.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, "trained_surrogate_real.pt"), weights_only=True))
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_manifold():
    with open("results/basis.json") as f:
        data = json.load(f)
    return (torch.tensor(data["basis"], dtype=torch.float32),
            torch.tensor(data["c_base"], dtype=torch.float32))


def load_real_data():
    return (torch.tensor(np.load(f"{DATA_DIR}/test_features.npy"), dtype=torch.float32),
            torch.tensor(np.load(f"{DATA_DIR}/test_labels.npy"), dtype=torch.float32))


def build_dataset_and_trajectories(test_X, test_y, N, W, classifier):
    benign_mask = test_y == 0
    benign_X = test_X[benign_mask]

    # Apply >0.95 confidence filter
    with torch.no_grad():
        probs = classifier(benign_X)
        nat_conf = torch.where(probs > 0.5, probs, 1.0 - probs)
        keep_mask = nat_conf.squeeze() <= 0.95
        filtered_X = benign_X[keep_mask]
        benign_indices = torch.where(benign_mask)[0]
        filtered_orig_indices = benign_indices[keep_mask]

    print(f"  Benign pool: {len(benign_X)} -> After >0.95 filter: {len(filtered_X)}")

    if len(filtered_X) < N:
        print(f"  WARNING: Only {len(filtered_X)} samples survive filter, need {N}. Using all.")
        N = len(filtered_X)

    dataset = filtered_X[:N]
    trajectories = []
    for i in range(N):
        orig_idx = filtered_orig_indices[i].item()
        start = max(0, orig_idx - (W - 1))
        traj = [test_X[j] for j in range(start, orig_idx)]
        while len(traj) < W - 1:
            traj.insert(0, filtered_X[0])
        trajectories.append(traj)
    return dataset, trajectories, N


def pgd_vuln_continuous(x0_init, y_target, classifier, traj, c_base, config,
                        alpha, epsilon, steps, naive, first_sample_flag):
    """Continuous-gradient PGD: uses alpha * normalized_grad instead of sign(grad)."""
    x_t = x0_init.clone().detach()
    x_t = x_t + torch.empty_like(x_t).uniform_(-epsilon, epsilon)
    x_t = project_Lp_ball(x_t, x0_init, epsilon)

    W_local = config.get("W", W)
    sliding_window = [w.clone().detach() for w in traj]

    for step in range(steps):
        x_t.requires_grad_(True)
        window_slice = sliding_window[-(W_local - 1):]
        x_def, theta_vuln = midas_defense_forward_vulnerable(
            x_t, window_slice, c_base, config, naive=naive
        )
        if not naive:
            theta_vuln.retain_grad()
        pred = classifier(x_def).clamp(1e-7, 1.0 - 1e-7)
        loss = F.binary_cross_entropy(pred, y_target.expand_as(pred))
        loss.backward()
        grad = x_t.grad

        if first_sample_flag and step < 5:
            if not naive and theta_vuln.grad is not None:
                gt = torch.norm(theta_vuln.grad).item()
            else:
                gt = 0.0
            tag = "CONT-VULN-ADAPT" if not naive else "CONT-VULN-NAIVE"
            gn = torch.norm(grad).item()
            print(f"    [{tag} Step {step:3d}] ||grad||={gn:.6f} ||d(loss)/d(theta)||={gt:.8f} loss={loss.item():.6f}")

        with torch.no_grad():
            grad_norm = torch.norm(grad)
            if grad_norm > 1e-8:
                x_t = x_t + alpha * grad / grad_norm
            x_t = project_Lp_ball(x_t, x0_init, epsilon)
            sliding_window.append(x_t.clone().detach())

    final_window = sliding_window[-(W_local - 1):]
    return x_t.detach(), final_window


def pgd_fixed_continuous(x0_init, y_target, classifier, traj, c_base, basis, config,
                         alpha, epsilon, steps, naive, first_sample_flag):
    """Continuous-gradient PGD with fixed basis."""
    x_t = x0_init.clone().detach()
    x_t = x_t + torch.empty_like(x_t).uniform_(-epsilon, epsilon)
    x_t = project_Lp_ball(x_t, x0_init, epsilon)

    W_local = config.get("W", W)
    sliding_window = [w.clone().detach() for w in traj]

    for step in range(steps):
        x_t.requires_grad_(True)
        window_slice = sliding_window[-(W_local - 1):]
        x_def, theta = midas_defense_forward(x_t, window_slice, c_base, basis, config, naive=naive)
        if not naive:
            theta.retain_grad()
        pred = classifier(x_def).clamp(1e-7, 1.0 - 1e-7)
        loss = F.binary_cross_entropy(pred, y_target.expand_as(pred))
        loss.backward()
        grad = x_t.grad

        if first_sample_flag and step < 5:
            if not naive and theta.grad is not None:
                gt = torch.norm(theta.grad).item()
            else:
                gt = 0.0
            tag = "CONT-FIXED-ADAPT" if not naive else "CONT-FIXED-NAIVE"
            gn = torch.norm(grad).item()
            print(f"    [{tag} Step {step:3d}] ||grad||={gn:.6f} ||d(loss)/d(theta)||={gt:.8f} loss={loss.item():.6f}")

        with torch.no_grad():
            grad_norm = torch.norm(grad)
            if grad_norm > 1e-8:
                x_t = x_t + alpha * grad / grad_norm
            x_t = project_Lp_ball(x_t, x0_init, epsilon)
            sliding_window.append(x_t.clone().detach())

    final_window = sliding_window[-(W_local - 1):]
    return x_t.detach(), final_window


def run_battery(classifier, dataset, trajectories, c_base, basis, config, attacks, N, mode="vuln"):
    results = {}
    for name, alpha, steps, eps in attacks:
        print(f"\n  {name}  [{mode.upper()}]")
        naive_succ = 0
        adaptive_succ = 0
        for i, (x0, traj) in enumerate(zip(dataset, trajectories)):
            is_first = (i == 0)
            with torch.no_grad():
                y_nat = (classifier(x0) > 0.5).float()

            if mode == "vuln":
                x_adv_n, win_n = pgd_vuln_continuous(x0, y_nat, classifier, traj, c_base, config,
                                                     alpha, eps, steps, naive=True, first_sample_flag=is_first)
                def_n, _ = midas_defense_forward_vulnerable(x_adv_n, win_n, c_base, config, naive=False)
                if (classifier(def_n) > 0.5).item() != y_nat.item():
                    naive_succ += 1
                if is_first:
                    print()
                x_adv_a, win_a = pgd_vuln_continuous(x0, y_nat, classifier, traj, c_base, config,
                                                     alpha, eps, steps, naive=False, first_sample_flag=is_first)
                def_a, _ = midas_defense_forward_vulnerable(x_adv_a, win_a, c_base, config, naive=False)
                if (classifier(def_a) > 0.5).item() != y_nat.item():
                    adaptive_succ += 1
            else:
                x_adv_n, win_n = pgd_fixed_continuous(x0, y_nat, classifier, traj, c_base, basis, config,
                                                      alpha, eps, steps, naive=True, first_sample_flag=is_first)
                def_n, _ = midas_defense_forward(x_adv_n, win_n, c_base, basis, config, naive=False)
                if (classifier(def_n) > 0.5).item() != y_nat.item():
                    naive_succ += 1
                if is_first:
                    print()
                x_adv_a, win_a = pgd_fixed_continuous(x0, y_nat, classifier, traj, c_base, basis, config,
                                                      alpha, eps, steps, naive=False, first_sample_flag=is_first)
                def_a, _ = midas_defense_forward(x_adv_a, win_a, c_base, basis, config, naive=False)
                if (classifier(def_a) > 0.5).item() != y_nat.item():
                    adaptive_succ += 1

        asr_n = naive_succ / N
        asr_a = adaptive_succ / N
        gap = asr_a - asr_n
        results[name] = {"naive": asr_n, "adaptive": asr_a, "gap": gap, "n": N}
        print(f"    Naive={asr_n:.2%}  Adaptive={asr_a:.2%}  Gap={gap:+.2%}")
    return results


def main():
    classifier = load_real_surrogate()
    basis, c_base = load_manifold()
    test_X, test_y = load_real_data()
    dataset, trajectories, actual_n = build_dataset_and_trajectories(test_X, test_y, N_SAMPLES, W, classifier)
    print(f"CONTINUOUS-GRADIENT SWEEP — d={D}, N={actual_n}, gamma={GAMMA}")

    attacks = [
        ("PGD T=20  eps=0.05", 0.01, 20, 0.05),
        ("PGD T=50  eps=0.05", 0.01, 50, 0.05),
        ("PGD T=100 eps=0.05", 0.01, 100, 0.05),
        ("PGD T=20  eps=0.1",  0.01, 20, 0.1),
        ("PGD T=50  eps=0.1",  0.01, 50, 0.1),
        ("PGD T=100 eps=0.1",  0.01, 100, 0.1),
        ("PGD T=20  eps=0.2",  0.01, 20, 0.2),
        ("PGD T=50  eps=0.2",  0.01, 50, 0.2),
        ("PGD T=100 eps=0.2",  0.01, 100, 0.2),
    ]

    print("\n" + "#"*60)
    print("# VULNERABLE BASIS — continuous gradient")
    print("#"*60)
    vuln = run_battery(classifier, dataset, trajectories, c_base, basis, CONFIG,
                       attacks, actual_n, mode="vuln")

    print("\n" + "#"*60)
    print("# FIXED BASIS — continuous gradient")
    print("#"*60)
    fixed = run_battery(classifier, dataset, trajectories, c_base, basis, CONFIG,
                        attacks, actual_n, mode="fixed")

    out = {"vuln": vuln, "fixed": fixed, "gradient_type": "continuous",
           "gamma": GAMMA, "d": D, "w": W, "n_samples": actual_n}
    os.makedirs("results", exist_ok=True)
    with open("results/check4_real_continuous.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to results/check4_real_continuous.json")


if __name__ == "__main__":
    main()
