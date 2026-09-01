"""
Train TrainedSurrogateModel on real UNSW-NB15 data (d=42).
Saves weights, meta, and runs verification chain.
"""
import sys, os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shadow.defense import (
    TrainedSurrogateModel, SmoothMockModel,
    midas_defense_forward, penetration_epsilon_windowed,
    rotation_angle, project_to_manifold, rotate_manifold_fixed_basis,
)
from shadow.attacks import project_Lp_ball

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shadow", "trained_weights")

def load_real_data():
    train_X = np.load(f"{DATA_DIR}/train_features.npy")
    train_y = np.load(f"{DATA_DIR}/train_labels.npy")
    test_X = np.load(f"{DATA_DIR}/test_features.npy")
    test_y = np.load(f"{DATA_DIR}/test_labels.npy")
    return (torch.tensor(train_X, dtype=torch.float32),
            torch.tensor(train_y, dtype=torch.float32),
            torch.tensor(test_X, dtype=torch.float32),
            torch.tensor(test_y, dtype=torch.float32))


def train_on_real_data(d=42, hidden=16, seed=42, n_epochs=500, lr=0.01,
                       batch_size=1024, save_dir=None):
    print("="*60)
    print("Training TrainedSurrogateModel on real UNSW-NB15 data")
    print(f"  d={d}, hidden={hidden}, seed={seed}, epochs={n_epochs}, lr={lr}")
    print("="*60)

    train_X, train_y, test_X, test_y = load_real_data()
    print(f"  Train: {len(train_X)} samples ({int(train_y.sum())} attack, {len(train_y)-int(train_y.sum())} benign)")
    print(f"  Test:  {len(test_X)} samples ({int(test_y.sum())} attack, {len(test_y)-int(test_y.sum())} benign)")

    torch.manual_seed(seed)
    model = TrainedSurrogateModel(d, hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    n_train = len(train_X)
    log = {"losses": [], "train_accs": [], "test_accs": []}

    t0 = time.time()
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            idx = perm[start:start+batch_size]
            X_batch = train_X[idx]
            y_batch = train_y[idx]

            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches

        model.eval()
        with torch.no_grad():
            train_pred = model(train_X)
            train_acc = ((train_pred > 0.5).float() == train_y).float().mean().item()
            test_pred = model(test_X)
            test_acc = ((test_pred > 0.5).float() == test_y).float().mean().item()

        log["losses"].append(avg_loss)
        log["train_accs"].append(train_acc)
        log["test_accs"].append(test_acc)

        if (epoch + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1:4d}/{n_epochs}: loss={avg_loss:.4f}  "
                  f"train_acc={train_acc:.2%}  test_acc={test_acc:.2%}  [{elapsed:.1f}s]")

    final_train_acc = log["train_accs"][-1]
    final_test_acc = log["test_accs"][-1]
    elapsed = time.time() - t0
    print(f"\n  Training complete in {elapsed:.1f}s")
    print(f"  Final train accuracy: {final_train_acc:.2%}")
    print(f"  Final test accuracy:  {final_test_acc:.2%}")

    for p in model.parameters():
        p.requires_grad_(False)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        weights_path = os.path.join(save_dir, "trained_surrogate_real.pt")
        torch.save(model.state_dict(), weights_path)
        meta = {
            "d": d, "hidden": hidden, "seed": seed,
            "n_epochs": n_epochs, "lr": lr, "batch_size": batch_size,
            "n_train": n_train, "final_train_accuracy": final_train_acc,
            "final_test_accuracy": final_test_acc,
            "arch": "TrainedSurrogateModel",
            "data_source": "UNSW-NB15 training-set.csv",
            "elapsed_seconds": elapsed,
        }
        meta_path = os.path.join(save_dir, "trained_surrogate_real_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  Saved weights to {weights_path}")
        print(f"  Saved meta to {meta_path}")

    return model, log


def verify_finite_difference(model, d=42, n_checks=20, eps=1e-4):
    print("\n" + "="*60)
    print("Verification 1: Finite-Difference Gradient Check")
    print("="*60)

    torch.manual_seed(1234)
    n_pass = 0
    max_err = 0.0

    for i in range(n_checks):
        x = torch.rand(d, requires_grad=True)
        pred = model(x)
        loss = pred.sum()

        loss.backward()
        analytic_grad = x.grad.clone()

        fd_grad = torch.zeros_like(x)
        for j in range(d):
            x_pert = x.clone().detach()
            x_pert[j] += eps
            f_p = model(x_pert).sum().item()

            x_pert[j] -= 2 * eps
            f_m = model(x_pert).sum().item()

            fd_grad[j] = (f_p - f_m) / (2 * eps)

        cos_sim = F.cosine_similarity(analytic_grad.unsqueeze(0), fd_grad.unsqueeze(0)).item()
        err = (analytic_grad - fd_grad).norm().item() / (analytic_grad.norm().item() + 1e-8)
        max_err = max(max_err, err)

        if cos_sim > 0.99:
            n_pass += 1

    print(f"  Passed: {n_pass}/{n_checks} (max relative error: {max_err:.6f})")
    if n_pass == n_checks:
        print("  ✓ All finite-difference checks passed")
    else:
        print(f"  ✗ {n_checks - n_pass} checks failed")
    return n_pass == n_checks


def verify_confidence_distribution(model, d=42, n_samples=500):
    print("\n" + "="*60)
    print("Verification 2: Confidence Distribution Check")
    print("="*60)

    train_X = torch.tensor(np.load(f"{DATA_DIR}/train_features.npy"), dtype=torch.float32)
    train_y = torch.tensor(np.load(f"{DATA_DIR}/train_labels.npy"), dtype=torch.float32)

    idx = torch.randperm(len(train_X))[:n_samples]
    samples = train_X[idx]
    labels = train_y[idx]

    with torch.no_grad():
        probs = model(samples)

    nat_probs = torch.where(probs > 0.5, probs, 1.0 - probs)
    nat_labels = (probs > 0.5).float()
    accuracy = (nat_labels == labels).float().mean().item()

    print(f"  Accuracy on {n_samples} real samples: {accuracy:.2%}")
    print(f"  Confidence distribution:")
    print(f"    min={nat_probs.min():.4f}  median={nat_probs.median():.4f}  "
          f"max={nat_probs.max():.4f}  mean={nat_probs.mean():.4f}")
    print(f"    frac>0.95={(nat_probs > 0.95).float().mean():.2%}")
    print(f"    frac<0.60={(nat_probs < 0.60).float().mean():.2%}")

    frac_near_boundary = ((nat_probs > 0.55) & (nat_probs < 0.95)).float().mean().item()
    print(f"    frac_near_boundary(0.55-0.95): {frac_near_boundary:.2%}")

    if frac_near_boundary > 0.3:
        print("  ✓ Good confidence distribution (many samples near boundary)")
    else:
        print("  ⚠ Low near-boundary fraction — classifier may be too confident")
    return accuracy


def verify_epsilon_p_distribution(model, d=42, W=10, n_trajectories=200, gamma=0.292):
    print("\n" + "="*60)
    print("Verification 3: Epsilon-p Phase Transition Check")
    print("="*60)

    train_X = torch.tensor(np.load(f"{DATA_DIR}/train_features.npy"), dtype=torch.float32)

    epsilon_p_values = []
    for i in range(n_trajectories):
        start_idx = torch.randint(0, len(train_X) - W, (1,)).item()
        window = [train_X[start_idx + j] for j in range(W)]
        eps_p = penetration_epsilon_windowed(window)
        epsilon_p_values.append(eps_p.item())

    epsilon_p_values = np.array(epsilon_p_values)
    p95 = np.percentile(epsilon_p_values, 95)
    p99 = np.percentile(epsilon_p_values, 99)

    print(f"  epsilon_p distribution ({n_trajectories} trajectories, W={W}):")
    print(f"    mean={epsilon_p_values.mean():.6f}  std={epsilon_p_values.std():.6f}")
    print(f"    p50={np.percentile(epsilon_p_values, 50):.6f}")
    print(f"    p95={p95:.6f}  p99={p99:.6f}")
    print(f"    max={epsilon_p_values.max():.6f}")
    print(f"  Current gamma: {gamma}")
    print(f"  Suggested gamma (95th percentile): {p95:.6f}")

    n_above = np.sum(epsilon_p_values > gamma)
    print(f"  Fractions above gamma: {n_above}/{n_trajectories} = {n_above/n_trajectories:.2%}")

    if p95 > gamma:
        print(f"  ⚠ gamma={gamma} is below p95={p95:.4f} — most trajectories trigger Logarithmic phase")
    else:
        print(f"  ✓ gamma={gamma} is above p95={p95:.4f} — mostly Archimedean phase")

    return {"p95": p95, "p99": p99, "mean": epsilon_p_values.mean(), "gamma": gamma}


def verify_basis_divergence(model, d=42, n_checks=10, eps=0.1):
    print("\n" + "="*60)
    print("Verification 4: Basis Divergence Check (naive vs adaptive)")
    print("="*60)

    basis = torch.tensor(json.load(open("results/basis.json"))["basis"], dtype=torch.float32)
    c_base = torch.tensor(json.load(open("results/basis.json"))["c_base"], dtype=torch.float32)

    train_X = torch.tensor(np.load(f"{DATA_DIR}/train_features.npy"), dtype=torch.float32)

    divergent = 0
    for i in range(n_checks):
        x0 = train_X[torch.randint(0, len(train_X), (1,)).item()]

        x_adv = x0.clone().detach().requires_grad_(True)
        pred = model(x_adv)
        loss = F.binary_cross_entropy(pred.clamp(1e-7, 1-1e-7), torch.tensor(1.0))
        loss.backward()
        with torch.no_grad():
            x_attacked = x0 + eps * torch.sign(x_adv.grad)
            x_attacked = project_Lp_ball(x_attacked, x0, eps)

        x_naive = x_attacked.clone().detach()
        window = [x_naive]

        theta_vuln = rotation_angle(
            penetration_epsilon_windowed(window + [x_naive]),
            gamma=0.292, lambda_=1.0, k=2.0,
            delta_theta_max=45.0 * 3.14159 / 180.0
        )

        basis_naive = basis.clone()
        basis_adaptive = compute_adaptive_basis(x_naive, c_base, d)

        dot_naive = torch.dot(basis_naive[0], basis_naive[1]).abs().item()
        dot_adaptive = torch.dot(basis_adaptive[0], basis_adaptive[1]).abs().item()

        diff = (basis_naive[0] - basis_adaptive[0]).norm().item()
        if diff > 0.01:
            divergent += 1

    print(f"  Naive vs adaptive basis divergence: {divergent}/{n_checks} samples show divergence")
    if divergent > 0:
        print(f"  ✓ Basis divergence detected (attacker-coupled basis differs from fixed)")
    else:
        print(f"  ⚠ No basis divergence — det() may not affect rotation plane")
    return divergent


def compute_adaptive_basis(x, c_base, d):
    centered = x - c_base
    norm = torch.norm(centered)
    if norm < 1e-7:
        b0 = torch.zeros(d)
        b0[0] = 1.0
    else:
        b0 = centered / norm

    b1 = torch.zeros(d)
    b1[1] = 1.0
    dot = torch.dot(b1, b0)
    b1 = b1 - dot * b0
    n1 = torch.norm(b1)
    if n1 > 1e-7:
        b1 = b1 / n1

    return torch.stack([b0, b1])


if __name__ == "__main__":
    d = 42

    model, log = train_on_real_data(
        d=d, hidden=16, seed=42, n_epochs=500, lr=0.01,
        batch_size=1024, save_dir=SAVE_DIR
    )

    verify_finite_difference(model, d=d)
    verify_confidence_distribution(model, d=d)
    eps_stats = verify_epsilon_p_distribution(model, d=d, W=10)
    verify_basis_divergence(model, d=d)

    print("\n" + "="*60)
    print("Suggested gamma for d=42 real data:")
    print(f"  gamma = {eps_stats['p95']:.6f} (95th percentile of benign epsilon_p)")
    print("="*60)
