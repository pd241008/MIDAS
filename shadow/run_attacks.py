import json
import torch
import os
from shadow.defense import SmoothMockModel, midas_defense_forward
from shadow.attacks import naive_pgd_attack, adaptive_pgd_attack
from shadow.report import generate_report
from shadow.scratch.debug_gradients import midas_defense_forward_vulnerable

def run_all_attacks():
    print("Loading basis from results/basis.json...")
    if not os.path.exists("results/basis.json"):
        raise RuntimeError("results/basis.json not found! Run the Rust harness first to export the basis.")
        
    with open("results/basis.json", "r") as f:
        basis_data = json.load(f)
        
    c_base = torch.tensor(basis_data["c_base"], dtype=torch.float32)
    basis = torch.tensor(basis_data["basis"], dtype=torch.float32)
    
    config = {
        "W": 4,
        "D": 10,
        "gamma": 0.5,
        "lambda": 1.0,
        "k": 2.0,
        "delta_theta_max_deg": 45.0,
        "tau": 0.3,
        "sla_budget_ms": 10,
        "channel_capacity": 4
    }
    
    D = config["D"]
    classifier = SmoothMockModel(in_features=D)
    
    torch.manual_seed(42)
    N_SAMPLES = 50
    dataset = [torch.rand(D) for _ in range(N_SAMPLES)]
    
    # Mock trajectories (in practice they'd come from a time series)
    trajectories = [
        [torch.rand(D) for _ in range(config["W"] - 1)] for _ in range(N_SAMPLES)
    ]
    
    # The configs defined in Rust
    attacks = [
        ("PGD T=20 eps=0.1", 0.01, 20, 0.1),
        ("PGD T=50 eps=0.1", 0.01, 50, 0.1),
        ("PGD T=100 eps=0.1", 0.01, 100, 0.1),
    ]
    
    results = {}
    
    for name, alpha, steps, eps in attacks:
        print(f"Evaluating {name} (Fixed Basis)...")
        naive_successes = 0
        adaptive_successes = 0
        naive_succ_vuln = 0
        adaptive_succ_vuln = 0
        
        naive_loss_sum = 0.0
        adaptive_loss_sum = 0.0
        naive_loss_vuln_sum = 0.0
        adaptive_loss_vuln_sum = 0.0
        
        for x0, traj in zip(dataset, trajectories):
            # Naive (Fixed Basis)
            x_adv_naive = naive_pgd_attack(x0, None, classifier, traj, c_base, basis, config, alpha, eps, steps)
            naive_defended, _ = midas_defense_forward(x_adv_naive, traj, c_base, basis, config, naive=False)
            l_naive = classifier(naive_defended).item()
            naive_loss_sum += l_naive
            if l_naive > 2.0:
                naive_successes += 1
            
            # Adaptive (Fixed Basis)
            if x0 is dataset[0]: print("  --- Running Adaptive (Fixed Basis) ---")
            x_adv_adaptive = adaptive_pgd_attack(x0, None, classifier, traj, c_base, basis, config, alpha, eps, steps)
            adaptive_defended, _ = midas_defense_forward(x_adv_adaptive, traj, c_base, basis, config, naive=False)
            l_adapt = classifier(adaptive_defended).item()
            adaptive_loss_sum += l_adapt
            if l_adapt > 2.0:
                adaptive_successes += 1

            # --- Vulnerable Basis Test ---
            def naive_attack_vuln(x0, y, cls, tw, cb, bs, conf, a, e, s):
                # We monkeypatch defense forward in attacks.py to use the vulnerable version
                pass

            # Since the attack loops call midas_defense_forward directly, we need a custom attack loop or to inject it.
            # Instead of monkeypatching, let's just create inline attack loops for the vulnerable basis.
            def pgd_vuln(x0_init, naive):
                x_t = x0_init.clone().detach()
                x_t = x_t + torch.empty_like(x_t).uniform_(-eps, eps)
                diff = torch.clamp(x_t - x0_init, min=-eps, max=eps)
                x_t = torch.clamp(x0_init + diff, min=0.0, max=1.0)
                
                for _ in range(steps):
                    x_t.requires_grad_(True)
                    window_cloned = [w.clone().detach() for w in traj]
                    # use the vulnerable forward
                    x_def, theta_vuln = midas_defense_forward_vulnerable(x_t, window_cloned, c_base, config, naive=naive)
                    if not naive:
                        theta_vuln.retain_grad()
                    loss = classifier(x_def)
                    loss.backward()
                    grad = x_t.grad
                    
                    if not naive and _ < 3 and x0 is dataset[0]:
                        grad_theta_norm = torch.norm(theta_vuln.grad).item() if theta_vuln.grad is not None else 0.0
                        print(f"    [Vuln Basis Step {_}] ||d(loss)/d(delta_theta)|| = {grad_theta_norm:.6f}")
                        
                    with torch.no_grad():
                        grad_norm = torch.norm(grad) + 1e-8
                        x_t = x_t + alpha * (grad / grad_norm)
                        diff = torch.clamp(x_t - x0_init, min=-eps, max=eps)
                        x_t = torch.clamp(x0_init + diff, min=0.0, max=1.0)
                        x_t.grad = None
                return x_t.detach()
            
            x_adv_naive_vuln = pgd_vuln(x0, naive=True)
            naive_def_vuln, _ = midas_defense_forward_vulnerable(x_adv_naive_vuln, traj, c_base, config, naive=False)
            l_naive_v = classifier(naive_def_vuln).item()
            naive_loss_vuln_sum += l_naive_v
            if l_naive_v > 2.0:
                naive_succ_vuln += 1
                
            if x0 is dataset[0]: print("  --- Running Adaptive (Vulnerable Basis) ---")
            x_adv_adaptive_vuln = pgd_vuln(x0, naive=False)
            adaptive_def_vuln, _ = midas_defense_forward_vulnerable(x_adv_adaptive_vuln, traj, c_base, config, naive=False)
            l_adapt_v = classifier(adaptive_def_vuln).item()
            adaptive_loss_vuln_sum += l_adapt_v
            if l_adapt_v > 2.0:
                adaptive_succ_vuln += 1
                
        asr_naive = naive_successes / N_SAMPLES
        asr_adaptive = adaptive_successes / N_SAMPLES
        asr_naive_vuln = naive_succ_vuln / N_SAMPLES
        asr_adaptive_vuln = adaptive_succ_vuln / N_SAMPLES
        
        results[name] = {"naive": asr_naive, "adaptive": asr_adaptive}
        print(f"  Fixed Basis      | Naive ASR: {asr_naive:.2f} (Loss: {naive_loss_sum/N_SAMPLES:.2f}), Adaptive ASR: {asr_adaptive:.2f} (Loss: {adaptive_loss_sum/N_SAMPLES:.2f})")
        print(f"  Vulnerable Basis | Naive ASR: {asr_naive_vuln:.2f} (Loss: {naive_loss_vuln_sum/N_SAMPLES:.2f}), Adaptive ASR: {asr_adaptive_vuln:.2f} (Loss: {adaptive_loss_vuln_sum/N_SAMPLES:.2f})")

    print("Generating report...")
    try:
        generate_report(results, "results/defense_success_rate.json", "results/adaptive_defense_success_rate.json")
        print("Wrote results/adaptive_defense_success_rate.json!")
    except Exception as e:
        print(f"Skipping JSON report generation: {e}")

if __name__ == "__main__":
    run_all_attacks()
