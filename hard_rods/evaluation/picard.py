"""
Picard iteration of the Euler-Lagrange equation rho = exp(-u + c1(rho)) (Appendix A), used for
every functional. Adaptive mixing: a step that lowers the residual is accepted and alpha grows;
otherwise it is rejected and alpha shrinks, down to alpha_min, where worse steps are accepted.
"""

import numpy as np
import torch


def picard_minimise(c1_of_rho, u, alpha, maxiter, tol, device, alpha_min=None, alpha_max=0.5,
                    grow=1.1, shrink=0.9, S_initial=None):
    """
    Iterate to the fixed point, starting from rho = 0.01 (0 at hard walls, where u = +inf).

    c1_of_rho: function rho -> c1 on torch tensors [2, H, W]; u: numpy array [2, H, W];
    tol: bound on the mean squared residual over the allowed sites; S_initial: start from
    rho_i = 0.01 (1 +/- S_initial), needed to reach an ordered bulk state.
    Returns (rho, steps); steps = maxiter if not converged (rho is always finite).
    """
    with torch.no_grad():
        u = torch.from_numpy(np.float32(u)).to(device)
        inf_mask = torch.isposinf(u)
        good = torch.logical_not(inf_mask)
        if alpha_min is None:
            alpha_min = alpha

        def G(x):
            """Euler-Lagrange map, with the hard-wall sites pinned to zero density."""
            out = torch.exp(-u + c1_of_rho(x))
            out[inf_mask] = 0
            return out

        rho = torch.ones_like(u) * 0.01
        if S_initial is not None:
            # same total density, split asymmetrically between the species
            rho[0] *= 1.0 + S_initial
            rho[1] *= 1.0 - S_initial
        rho[inf_mask] = 0

        rho_EL = G(rho)
        residual = torch.abs(rho - rho_EL)[good]
        residual_squared = torch.mean(residual ** 2)

        for i in range(maxiter):
            if residual_squared < tol:
                return rho, i
            if not torch.isfinite(residual_squared):
                # only reachable from the initial state, since bad steps are never accepted
                return rho, maxiter

            # trial step, built out of place so that the current iterate survives a rejection
            rho_trial = rho + alpha * (rho_EL - rho)
            rho_trial[inf_mask] = 0
            rho_trial.clamp_(0, float('inf'))

            rho_EL_trial = G(rho_trial)
            residual = torch.abs(rho_trial - rho_EL_trial)[good]
            max_trial, residual_squared_trial = residual.max(), torch.mean(residual ** 2)
            finite = bool(torch.isfinite(max_trial))

            improved = finite and bool(residual_squared_trial <= residual_squared)
            if finite and (improved or alpha <= alpha_min):
                # accept; the map value at the new point is reused in the next iteration
                rho, rho_EL = rho_trial, rho_EL_trial
                residual_squared = residual_squared_trial
                if alpha_max > 0 and improved:
                    alpha = min(alpha * grow, alpha_max)
            else:
                if alpha <= alpha_min:
                    # non-finite step at the floor: keep the last finite iterate
                    return rho, maxiter
                alpha = max(alpha * shrink, alpha_min)

        return rho, maxiter
