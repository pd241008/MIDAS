import torch
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
               config: dict, alpha: float, epsilon: float, steps: int, naive: bool) -> torch.Tensor:
    """
    PGD adaptive/naive attack loop. 
    If naive=True, the rotation angle `delta_theta` is detached from the computation graph.
    If naive=False, gradients flow backward entirely through delta_theta's magnitude.
    """
    x_t = x0.clone().detach()
    # Apply initial random perturbation
    x_t = x_t + torch.empty_like(x_t).uniform_(-epsilon, epsilon)
    x_t = project_Lp_ball(x_t, x0, epsilon)
    
    for step in range(steps):
        x_t.requires_grad_(True)
        
        # Clone window objects so we don't accidentally leak gradients across steps
        # The window must not be differentiated through across steps (attacker doesn't control past history directly in this single-step PGD, though in a sequential attack they might. The harness tests single-step attack against a fixed window).
        window_cloned = [w.clone().detach() for w in trajectory_window]
        
        x_defended, theta = midas_defense_forward(x_t, window_cloned, c_base, basis, config, naive=naive)
        
        # We explicitly track theta to log its gradient
        if not naive:
            theta.retain_grad()
            
        loss = classifier(x_defended)
        
        loss.backward()
        grad = x_t.grad
        
        # Log theta grad for the first few steps to verify adaptive path is alive
        if not naive and step < 3:
            grad_theta_norm = torch.norm(theta.grad).item() if theta.grad is not None else 0.0
            print(f"    [Step {step}] ||d(loss)/d(delta_theta)|| = {grad_theta_norm:.6f}")
            
        with torch.no_grad():
            x_t = x_t + alpha * torch.sign(grad)
            x_t = project_Lp_ball(x_t, x0, epsilon)
            
    return x_t.detach()

def adaptive_pgd_attack(x0: torch.Tensor, y: torch.Tensor, classifier: torch.nn.Module, 
                        trajectory_window: list[torch.Tensor], c_base: torch.Tensor, basis: torch.Tensor, 
                        config: dict, alpha: float, epsilon: float, steps: int) -> torch.Tensor:
    return pgd_attack(x0, y, classifier, trajectory_window, c_base, basis, config, alpha, epsilon, steps, naive=False)

def naive_pgd_attack(x0: torch.Tensor, y: torch.Tensor, classifier: torch.nn.Module, 
                     trajectory_window: list[torch.Tensor], c_base: torch.Tensor, basis: torch.Tensor, 
                     config: dict, alpha: float, epsilon: float, steps: int) -> torch.Tensor:
    return pgd_attack(x0, y, classifier, trajectory_window, c_base, basis, config, alpha, epsilon, steps, naive=True)
