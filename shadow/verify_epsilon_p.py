"""
Verify corrected trajectory generation: epsilon_p and M_t distributions.

After fixing trajectory vectors from torch.rand(D) to torch.randn(D),
check that:
1. Cosine similarity M_t averages near 0 (Clean Traffic Axiom)
2. epsilon_p distribution is reasonable
3. gamma=0.292 is well-placed (95th pct of corrected eps_p)
"""
import math
import torch
from shadow.defense import momentum, penetration_epsilon_windowed


def main():
    torch.manual_seed(42)
    D = 10
    W = 4
    N_TRAJECTORIES = 10000
    N_PAIRS_PER_TRAJ = W - 1  # 3 pairs per trajectory window

    print("=" * 70)
    print("VERIFICATION: Corrected trajectory generation (torch.rand(D) - 0.5)")
    print("=" * 70)

    # ---- Collect M_t and epsilon_p over many independent trajectories ----
    all_momentum_values = []
    all_epsilon_p_values = []

    for _ in range(N_TRAJECTORIES):
        window = [torch.rand(D) - 0.5 for _ in range(W)]
        eps_p = penetration_epsilon_windowed(window)
        all_epsilon_p_values.append(eps_p.item())

        for i in range(len(window) - 1):
            m = momentum(window[i + 1], window[i])
            all_momentum_values.append(m.item())

    all_momentum_values = torch.tensor(all_momentum_values)
    all_epsilon_p_values = torch.tensor(all_epsilon_p_values)

    # ---- M_t distribution ----
    print(f"\n--- M_t (cosine similarity) distribution ---")
    print(f"  N pairs:    {len(all_momentum_values)}")
    print(f"  Mean:       {all_momentum_values.mean():.6f}  (target: ~0)")
    print(f"  Std:        {all_momentum_values.std():.6f}")
    print(f"  Median:     {all_momentum_values.median():.6f}")
    print(f"  Frac > 0:   {(all_momentum_values > 0).float().mean():.2%}")
    print(f"  Frac < 0:   {(all_momentum_values < 0).float().mean():.2%}")
    print(f"  Min:        {all_momentum_values.min():.6f}")
    print(f"  Max:        {all_momentum_values.max():.6f}")

    # ---- epsilon_p distribution ----
    print(f"\n--- epsilon_p distribution ---")
    print(f"  N trajectories: {len(all_epsilon_p_values)}")
    print(f"  Mean:       {all_epsilon_p_values.mean():.6f}")
    print(f"  Std:        {all_epsilon_p_values.std():.6f}")
    print(f"  Median:     {all_epsilon_p_values.median():.6f}")
    print(f"  Min:        {all_epsilon_p_values.min():.6f}")
    print(f"  Max:        {all_epsilon_p_values.max():.6f}")

    for pct in [50, 75, 90, 95, 99]:
        val = torch.quantile(all_epsilon_p_values, pct / 100.0).item()
        print(f"  {pct}th percentile: {val:.6f}")

    # ---- Gamma placement analysis ----
    gamma = 0.292
    p95 = torch.quantile(all_epsilon_p_values, 0.95).item()
    p90 = torch.quantile(all_epsilon_p_values, 0.90).item()
    frac_below_gamma = (all_epsilon_p_values <= gamma).float().mean().item()

    print(f"\n--- Gamma placement analysis ---")
    print(f"  Current gamma:     {gamma}")
    print(f"  90th pct eps_p:    {p90:.6f}")
    print(f"  95th pct eps_p:    {p95:.6f}")
    print(f"  Frac eps_p <= gamma: {frac_below_gamma:.2%}")
    print(f"  Frac eps_p > gamma:  {1 - frac_below_gamma:.2%}")

    if p95 <= gamma:
        print(f"\n  OK: gamma={gamma} is above the 95th percentile ({p95:.4f}).")
        print(f"  Most attacks will start in the Archimedean phase.")
        suggested_gamma = gamma
    else:
        print(f"\n  WARNING: gamma={gamma} is BELOW the 95th percentile ({p95:.4f}).")
        print(f"  {1 - frac_below_gamma:.1%} of benign trajectories already exceed gamma.")
        print(f"  Paper's intent: gamma = 95th pct of benign epsilon_p.")
        suggested_gamma = p95
        print(f"  Suggested gamma: {p95:.4f} (95th percentile)")

    # ---- What theta values does this produce? ----
    lambda_ = 1.0
    k = 2.0
    delta_theta_max_deg = 45.0
    delta_theta_max_rad = math.radians(delta_theta_max_deg)

    print(f"\n--- theta distribution with gamma={suggested_gamma:.4f} ---")
    thetas_deg = []
    for eps_p in all_epsilon_p_values:
        eps_p_val = eps_p.item()
        if eps_p_val <= suggested_gamma:
            theta = lambda_ * eps_p_val
        else:
            theta = min(lambda_ * math.exp(k * (eps_p_val - suggested_gamma)), delta_theta_max_rad)
        thetas_deg.append(math.degrees(theta))

    thetas_deg = torch.tensor(thetas_deg)
    print(f"  Mean theta:    {thetas_deg.mean():.4f} deg")
    print(f"  Median theta:  {thetas_deg.median():.4f} deg")
    print(f"  Max theta:     {thetas_deg.max():.4f} deg")
    print(f"  Frac at ceiling: {(thetas_deg >= delta_theta_max_deg - 0.01).float().mean():.2%}")
    print(f"  95th pct theta: {torch.quantile(thetas_deg, 0.95):.4f} deg")

    # ---- Also test with the OLD generator for comparison ----
    print(f"\n--- COMPARISON: Old generator (torch.rand, no centering) ---")
    old_momentum = []
    old_epsilon_p = []
    for _ in range(1000):
        window = [torch.rand(D) for _ in range(W)]
        eps_p = penetration_epsilon_windowed(window)
        old_epsilon_p.append(eps_p.item())
        for i in range(len(window) - 1):
            m = momentum(window[i + 1], window[i])
            old_momentum.append(m.item())

    old_momentum = torch.tensor(old_momentum)
    old_epsilon_p = torch.tensor(old_epsilon_p)
    print(f"  Old M_t mean:      {old_momentum.mean():.6f}  (was ~0.75, violating Axiom 1)")
    print(f"  Old eps_p mean:    {old_epsilon_p.mean():.6f}")
    print(f"  Old eps_p 95th:    {torch.quantile(old_epsilon_p, 0.95):.6f}")
    print(f"  Old frac > gamma:  {(old_epsilon_p > gamma).float().mean():.2%}")

    print(f"\n--- COMPARISON: Previous fix (torch.randn, breaks clamping) ---")
    randn_momentum = []
    randn_epsilon_p = []
    for _ in range(1000):
        window = [torch.randn(D) for _ in range(W)]
        eps_p = penetration_epsilon_windowed(window)
        randn_epsilon_p.append(eps_p.item())
        for i in range(len(window) - 1):
            m = momentum(window[i + 1], window[i])
            randn_momentum.append(m.item())

    randn_momentum = torch.tensor(randn_momentum)
    randn_epsilon_p = torch.tensor(randn_epsilon_p)
    print(f"  randn M_t mean:    {randn_momentum.mean():.6f}  (zero-mean vectors)")
    print(f"  randn eps_p mean:  {randn_epsilon_p.mean():.6f}")
    print(f"  randn eps_p 95th:  {torch.quantile(randn_epsilon_p, 0.95):.6f}")
    print(f"  NOTE: randn breaks because project_Lp_ball clamps to [0,1],")
    print(f"        making x_t non-negative while traj vectors stay in R^D.")


if __name__ == "__main__":
    main()
