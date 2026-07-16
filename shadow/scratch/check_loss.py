import torch
from midas_shadow.defense import *
from midas_shadow.scratch.debug_gradients import midas_defense_forward_vulnerable

class SmoothLoss(torch.nn.Module):
    def forward(self, x):
        # A perfectly smooth target: just try to maximize the sum of elements
        return torch.sum(x)

def test_attack():
    torch.manual_seed(42)
    D = 10
    config = {"W": 4, "D": D, "gamma": 0.5, "lambda": 1.0, "k": 2.0, "delta_theta_max_deg": 45.0}
    c_base = torch.zeros(D)
    
    # We create a random dataset
    dataset = [torch.rand(D) for _ in range(50)]
    trajectories = [[torch.randn(D) for _ in range(3)] for _ in range(50)]
    
    eps = 0.2
    alpha = 0.02
    steps = 50
    classifier = SmoothLoss()
    
    print("Evaluating Smooth Loss with Vulnerable Basis...")
    
    def pgd_vuln(x0_init, naive):
        x_t = x0_init.clone().detach()
        x_t = x_t + torch.empty_like(x_t).uniform_(-eps, eps)
        diff = torch.clamp(x_t - x0_init, min=-eps, max=eps)
        x_t = torch.clamp(x0_init + diff, min=0.0, max=1.0)
        
        for _ in range(steps):
            x_t.requires_grad_(True)
            window_cloned = [w.clone().detach() for w in traj]
            x_def, _ = midas_defense_forward_vulnerable(x_t, window_cloned, c_base, config, naive=naive)
            loss = classifier(x_def)
            grad = torch.autograd.grad(loss, x_t, retain_graph=False)[0]
            with torch.no_grad():
                x_t = x_t + alpha * torch.sign(grad)
                diff = torch.clamp(x_t - x0_init, min=-eps, max=eps)
                x_t = torch.clamp(x0_init + diff, min=0.0, max=1.0)
        return x_t.detach()

    naive_loss_sum = 0
    adaptive_loss_sum = 0

    for x0, traj in zip(dataset, trajectories):
        x_adv_naive = pgd_vuln(x0, naive=True)
        naive_def, _ = midas_defense_forward_vulnerable(x_adv_naive, traj, c_base, config, naive=False)
        naive_loss_sum += classifier(naive_def).item()
        
        x_adv_adaptive = pgd_vuln(x0, naive=False)
        adaptive_def, _ = midas_defense_forward_vulnerable(x_adv_adaptive, traj, c_base, config, naive=False)
        adaptive_loss_sum += classifier(adaptive_def).item()
        
    print(f"Mean Naive Loss (higher is better for attacker): {naive_loss_sum/50:.4f}")
    print(f"Mean Adaptive Loss (higher is better for attacker): {adaptive_loss_sum/50:.4f}")

if __name__ == "__main__":
    test_attack()
