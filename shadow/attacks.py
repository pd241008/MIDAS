import torch
import torch.nn.functional as F
from shadow.defense import midas_defense_forward

def project_Lp_ball(x: torch.Tensor, x0: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    Projects x onto the L-infinity ball around x0 with radius epsilon.
    Also clamps to [0, 1] assuming normalized feature space.
    """
    diff = x - x0
    diff = torch.clamp(diff, min=-epsilon, max=epsilon)
    res = x0 + diff
    return torch.clamp(res, min=0.0, max=1.0)

def pgd_attack(x0: torch.Tensor, y: torch.Tensor, classifier: torch.nn.Module,
               trajectory_window: list[torch.Tensor], c_base: torch.Tensor, basis: torch.Tensor,
               config: dict, alpha: float, epsilon: float, steps: int, naive: bool,
               first_sample: bool = False) -> torch.Tensor:
    """
    PGD adaptive/naive attack loop with BCE loss against SmoothMockModel.
    If naive=True, the rotation angle `delta_theta` is detached from the computation graph.
    If naive=False, gradients flow backward entirely through delta_theta's magnitude.

    Attack objective: maximize BCE(classifier(x_defended), y) where y is the natural label.
    Success = model prediction flips after defense.

    When first_sample=True, logs ||d(loss)/d(delta_theta)|| at every step for diagnostics.
    """
    x_t = x0.clone().detach()
    # Apply initial random perturbation
    x_t = x_t + torch.empty_like(x_t).uniform_(-epsilon, epsilon)
    x_t = project_Lp_ball(x_t, x0, epsilon)

    W = config.get("W", 4)
    sliding_window = [w.clone().detach() for w in trajectory_window]

    for step in range(steps):
        x_t.requires_grad_(True)

        window_slice = sliding_window[-(W - 1):]
        x_defended, theta = midas_defense_forward(x_t, window_slice, c_base, basis, config, naive=naive)

        # We explicitly track theta to log its gradient
        if not naive:
            theta.retain_grad()

        pred = classifier(x_defended).clamp(1e-7, 1.0 - 1e-7)
        loss = F.binary_cross_entropy(pred, y.expand_as(pred))

        loss.backward()
        grad = x_t.grad

        # Gradient-aliveness logging: ||d(loss)/d(delta_theta)||
        if first_sample:
            if not naive and theta.grad is not None:
                grad_theta_norm = torch.norm(theta.grad).item()
            else:
                grad_theta_norm = 0.0
            tag = "ADAPTIVE" if not naive else "NAIVE   "
            print(f"    [{tag} Step {step:3d}] ||d(loss)/d(delta_theta)|| = {grad_theta_norm:.8f}  loss = {loss.item():.6f}")

        with torch.no_grad():
            x_t = x_t + alpha * torch.sign(grad)
            x_t = project_Lp_ball(x_t, x0, epsilon)
            sliding_window.append(x_t.clone().detach())

    final_window = sliding_window[-(W - 1):]
    return x_t.detach(), final_window

def adaptive_pgd_attack(x0: torch.Tensor, y: torch.Tensor, classifier: torch.nn.Module,
                        trajectory_window: list[torch.Tensor], c_base: torch.Tensor, basis: torch.Tensor,
                        config: dict, alpha: float, epsilon: float, steps: int,
                        first_sample: bool = False) -> torch.Tensor:
    return pgd_attack(x0, y, classifier, trajectory_window, c_base, basis, config, alpha, epsilon, steps, naive=False, first_sample=first_sample)

def naive_pgd_attack(x0: torch.Tensor, y: torch.Tensor, classifier: torch.nn.Module,
                     trajectory_window: list[torch.Tensor], c_base: torch.Tensor, basis: torch.Tensor,
                     config: dict, alpha: float, epsilon: float, steps: int,
                     first_sample: bool = False) -> torch.Tensor:
    return pgd_attack(x0, y, classifier, trajectory_window, c_base, basis, config, alpha, epsilon, steps, naive=True, first_sample=first_sample)
