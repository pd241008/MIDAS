import torch

def momentum(v_t: torch.Tensor, v_prev: torch.Tensor) -> torch.Tensor:
    """
    Computes cosine similarity between v_t and v_prev.
    Matches Rust edge-core/src/trajectory.rs `compute_momentum`.
    """
    if v_t.shape != v_prev.shape:
        return torch.tensor(0.0, device=v_t.device)

    norm_t = torch.norm(v_t, p=2)
    norm_tm1 = torch.norm(v_prev, p=2)

    if norm_t < 1e-7 or norm_tm1 < 1e-7:
        return torch.tensor(0.0, device=v_t.device)

    dot = torch.sum(v_t * v_prev)
    m = dot / (norm_t * norm_tm1)
    return torch.clamp(m, min=-1.0, max=1.0)


def penetration_epsilon_windowed(window: list[torch.Tensor]) -> torch.Tensor:
    """
    epsilon_p = (1 / |W|) * sum_{i in W} ||v_i||_2 * max(0, M_i)
    Matches Rust edge-core/src/trajectory.rs `penetration_epsilon_windowed`.
    """
    if len(window) < 2:
        return torch.tensor(0.0, device=window[0].device if window else torch.device('cpu'))

    sum_val = torch.tensor(0.0, device=window[0].device)
    pairs = 0

    for i in range(len(window) - 1):
        v_prev = window[i]
        v_cur = window[i + 1]
        m = momentum(v_cur, v_prev)
        norm_cur = torch.norm(v_cur, p=2)
        # max(0, m)
        m_clamped = torch.clamp(m, min=0.0)
        sum_val = sum_val + norm_cur * m_clamped
        pairs += 1

    return sum_val / pairs


def rotation_angle(epsilon_p: torch.Tensor, gamma: float, lambda_: float, k: float, delta_theta_max: float) -> torch.Tensor:
    """
    Matches Rust edge-core/src/rotation.rs `rotation_angle`.
    Archimedean: lambda * epsilon_p
    Logarithmic: min(lambda * exp(k * (epsilon_p - gamma)), delta_theta_max)
    """
    if epsilon_p <= gamma:
        return lambda_ * epsilon_p
    else:
        val = lambda_ * torch.exp(k * (epsilon_p - gamma))
        return torch.clamp(val, max=delta_theta_max)


def rotate_manifold_fixed_basis(x: torch.Tensor, basis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Matches Rust edge-core/src/rotation.rs `rotate_manifold_givens_fixed_basis`.
    basis is a [2, D] tensor containing the two orthonormal PCA vectors.
    """
    if x.shape[0] < 2 or basis.shape[0] < 2:
        return x.clone()

    c = torch.cos(theta)
    s = torch.sin(theta)

    b0 = basis[0]
    b1 = basis[1]

    coord0 = torch.sum(x * b0)
    coord1 = torch.sum(x * b1)

    rot0 = c * coord0 - s * coord1
    rot1 = s * coord0 + c * coord1

    rotated = x.clone()
    rotated = rotated + (rot0 - coord0) * b0 + (rot1 - coord1) * b1
    return rotated


def project_to_manifold(x_adv: torch.Tensor, c_base: torch.Tensor) -> torch.Tensor:
    """
    Matches Rust edge-core/src/rotation.rs `project_to_manifold`.
    """
    if x_adv.shape != c_base.shape:
        return x_adv.clone()

    diff = torch.norm(x_adv - c_base, p=2)
    if diff <= 1e-7:
        return x_adv.clone()

    return x_adv - 0.5 * (x_adv - c_base)


def midas_defense_forward(x_t: torch.Tensor, trajectory_window: list[torch.Tensor], c_base: torch.Tensor, basis: torch.Tensor, config: dict, naive: bool = False):
    """
    The end-to-end differentiable forward pass of the MIDAS-Edge defense.
    `trajectory_window` includes previous x vectors up to x_{t-1}.
    Returns (x_defended, theta) so we can inspect gradients on theta.
    """
    gamma = config.get("gamma", 0.5)
    lambda_ = config.get("lambda", 1.0)
    k = config.get("k", 2.0)
    delta_theta_max_deg = config.get("delta_theta_max_deg", 45.0)
    delta_theta_max = delta_theta_max_deg * (3.141592653589793 / 180.0)

    full_window = trajectory_window + [x_t]

    epsilon_p = penetration_epsilon_windowed(full_window)
    theta = rotation_angle(epsilon_p, gamma, lambda_, k, delta_theta_max)
    
    if naive:
        theta = theta.detach()
        
    x_defended = rotate_manifold_fixed_basis(x_t, basis, theta)

    return x_defended, theta


class MockClassifierLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Matches Rust `MockModel`: label = (sum(x).abs() as u32) % 10. 
        The Rust naive finite-difference attacker uses the label directly as a continuous scalar for gradients.
        To mirror this exactly, we return fmod(abs(sum(x)), 10.0) as the continuous loss target.
        """
        s = torch.abs(torch.sum(x))
        return torch.fmod(s, 10.0)

class SmoothMockModel(torch.nn.Module):
    def __init__(self, in_features=10):
        super().__init__()
        # Use a fixed, stable random seed for the weights so it's consistent across runs
        torch.manual_seed(42)
        self.w = torch.nn.Parameter(torch.randn(in_features))
        self.b = torch.nn.Parameter(torch.randn(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Logistic loss against a fixed target. 
        Returns a scalar loss that the attacker wants to maximize.
        """
        logits = torch.sum(self.w * x, dim=-1) + self.b
        # Let's say the target is always class 1, and the attacker is maximizing the logit for class 1
        # To make it a smooth loss that can be maximized endlessly (or up to saturation),
        # we can just return the raw logit, or a sigmoid
        return logits
