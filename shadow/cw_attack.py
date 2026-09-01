import torch
import torch.nn.functional as F
from shadow.defense import midas_defense_forward
from shadow.scratch.debug_gradients import midas_defense_forward_vulnerable


def cw_l2_attack(x0, y, classifier, trajectory_window, c_base, basis,
                 config, initial_c=1.0, c_steps=5, lr=0.01, iters=200,
                 naive=False, coupled_basis=False, first_sample=False):
    """
    Carlini-Wagner L2 (untargeted, binary sigmoid output).

    Minimizes   ||x' - x0||_2^2 + c * BCE(pred(x'), y)
    over w via x' = 0.5*(tanh(w)+1), using Adam. Smaller ||x'-x0||_2 is
    preferred; c is binary-searched upward so each perturbation must actually
    flip the defended prediction.

    Mirrors the PGD naive/adaptive split:
      - naive=True        : rotation theta AND (if coupled) the basis are detached
      - naive=False       : gradients flow through the entire defense
      - coupled_basis=True: use the attacker-coupled basis (midas_defense_forward_vulnerable)
      - coupled_basis=False: use the fixed PCA-2 basis passed in `basis`

    Success (returned) = defended prediction flips relative to natural label y.
    """
    W = config.get("W", 4)
    sliding_window = [w.clone().detach() for w in trajectory_window]
    x0c = x0.clone().detach()
    yc = y.clone().detach().float()

    # tanh-space parametrization maps any w to [0,1]; initialize near x0.
    w = (torch.atanh(2.0 * x0c.clamp(0.05, 0.95) - 1.0)).clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([w], lr=lr)

    def _defense_forward(x_t):
        window_slice = sliding_window[-(W - 1):]
        if coupled_basis:
            return midas_defense_forward_vulnerable(x_t, window_slice, c_base, config, naive=naive)
        else:
            return midas_defense_forward(x_t, window_slice, c_base, basis, config, naive=naive)

    def _map():
        return 0.5 * (torch.tanh(w) + 1.0)

    best = None
    best_dist = None
    best_c = initial_c

    for cstep in range(c_steps):
        c = initial_c * (4.0 ** cstep)
        # re-init w each c step from a clean start for fair per-c comparison
        w.data = (torch.atanh(2.0 * x0c.clamp(0.05, 0.95) - 1.0)).clone()
        optimizer = torch.optim.Adam([w], lr=lr)

        for it in range(iters):
            optimizer.zero_grad()
            x_t = _map()
            x_defended, theta = _defense_forward(x_t)
            pred = classifier(x_defended).clamp(1e-7, 1.0 - 1e-7)
            l2 = torch.sum((x_t - x0c) ** 2)
            ce = F.binary_cross_entropy(pred, yc.expand_as(pred))
            loss = l2 + c * ce
            loss.backward()
            optimizer.step()

        # evaluate resulting x
        x_adv = _map().detach()
        window_slice = sliding_window[-(W - 1):]
        if coupled_basis:
            x_defended, _ = midas_defense_forward_vulnerable(x_adv, window_slice, c_base, config, naive=False)
        else:
            x_defended, _ = midas_defense_forward(x_adv, window_slice, c_base, basis, config, naive=False)
        with torch.no_grad():
            pred = classifier(x_defended)
        try_success = (pred > 0.5).item() != yc.item()
        dist = torch.norm(x_adv - x0c).item()

        if first_sample:
            print(f"    [cw naive={naive} coupled={coupled_basis} c={c:.4g}] "
                  f"dist={dist:.6f} success={try_success} pred={pred.item():.4f}")

        if try_success and (best_dist is None or dist < best_dist):
            best = x_adv
            best_dist = dist
            best_c = c

        # stop early if found a low-distortion success and c is large enough
        if try_success and cstep >= 1:
            break

    if best is None:
        # no success found; return the best-effort (last) point
        best = _map().detach()
        best_dist = torch.norm(best - x0c).item()
        best_c = None

    # final success decision on the chosen perturbation
    window_slice = sliding_window[-(W - 1):]
    if coupled_basis:
        x_defended, _ = midas_defense_forward_vulnerable(best, window_slice, c_base, config, naive=False)
    else:
        x_defended, _ = midas_defense_forward(best, window_slice, c_base, basis, config, naive=False)
    with torch.no_grad():
        pred = classifier(x_defended)
    success = (pred > 0.5).item() != yc.item()

    return success, best, best_dist, best_c
