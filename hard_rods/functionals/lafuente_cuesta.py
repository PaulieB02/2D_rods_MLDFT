"""
The Lafuente-Cuesta (LC) lattice functional (Sec. 2.2), the analytical reference.

Written in torch, so that it is minimised by the same Picard iteration as the networks.
"""

import torch


def lafuente_cuesta_excess(rho, k):
    """
    beta F_ex of the LC functional on a periodic box; rho is [2, H, W] with indices [y, x].

        beta F_ex = sum_{i,j} Phi(A) - Phi(B) - Phi(C),   Phi(n) = (1 - n) ln(1 - n) + n,
        A[i, j] = sum_{s=0}^{k-1} ( rho0[i + k-1-s, j] + rho1[i + k-1, j - s] )
        B[i, j] = sum_{s=1}^{k-1}   rho0[i + k-1-s, j]
        C[i, j] = sum_{s=1}^{k-1}   rho1[i, j + k-1-s]

    A, B and C are the weighted densities n_0, n_1^(0) and n_1^(1), with the site labels
    shifted by k - 1 (which leaves the periodic sums unchanged).
    """
    rho0, rho1 = rho[0], rho[1]

    A = torch.zeros_like(rho0)
    for s in range(k):
        A = A + torch.roll(rho0, -(k - 1 - s), dims=0) \
              + torch.roll(torch.roll(rho1, s, dims=1), -(k - 1), dims=0)

    B = torch.zeros_like(rho0)
    C = torch.zeros_like(rho0)
    for s in range(1, k):
        B = B + torch.roll(rho0, -(k - 1 - s), dims=0)
        C = C + torch.roll(rho1, -(k - 1 - s), dims=1)

    def Phi(x):
        return (1 - x) * torch.log(1 - x) + x

    return torch.sum(Phi(A) - Phi(B) - Phi(C))


def c1_lafuente_cuesta(k):
    """The function rho -> c1 = -d(beta F_ex)/d rho (detached), for the Picard iteration."""
    def c1_of_rho(rho):
        rho_leaf = rho.detach().requires_grad_(True)
        with torch.enable_grad():
            F_exc = lafuente_cuesta_excess(rho_leaf, k)
            grad = torch.autograd.grad(F_exc, rho_leaf)[0]
        return -grad.detach()
    return c1_of_rho
