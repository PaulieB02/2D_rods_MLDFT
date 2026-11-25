"""
c1 of the trained networks (Sec. 3.2 and 3.3), and loading of the checkpoints.

Both networks impose the orientation-exchange symmetry T (swap the species, transpose x and y).
A single profile is [2, H, W]; a batch [N, 2, H, W] keeps its batch dimension.
"""

import torch

from ..config import ARCHITECTURE_OF
from .networks import ARCHITECTURES


def phi_field(net, rho, local_learning_mode=False):
    """
    Local excess free-energy density phi of the Phi-network.

    Evaluation: phi averaged over rho and T rho, [2, H, W] -> [1, H, W] (square box).
    Training (local_learning_mode, padding off): one plain pass on a local window.
    """
    assert rho.shape[-1] == rho.shape[-2], "both orientations share one input tensor, so the grid must be square"

    single_sample = rho.dim() == 3
    rho_batch = rho.unsqueeze(0) if single_sample else rho     # [N, 2, H, W]
    n = rho_batch.shape[0]
    if not local_learning_mode:
        # symmetry partner T rho: swap the two species and transpose x and y
        rho_flipped = torch.flip(rho_batch, dims=[1]).permute(0, 1, 3, 2)
        phi_both = net(torch.cat((rho_batch, rho_flipped), dim=0))                 # [2N, 1, H, W]
        phi_combined = (phi_both[:n] + phi_both[n:].permute(0, 1, 3, 2)) / 2
    else:
        phi_combined = net(rho_batch)                                              # [N, 1, h, w]

    return phi_combined.squeeze(0) if single_sample else phi_combined


def c1_direct(net, rho):
    """c1 of both species, [2, H, W]; species 1 is the network on T rho, transposed back."""
    single_sample = rho.dim() == 3
    rho_batch = rho.unsqueeze(0) if single_sample else rho     # [N, 2, H, W]

    # symmetry partner T rho: swap the two species and transpose x and y
    rho_other = torch.flip(rho_batch.transpose(2, 3), dims=[1])
    c1 = torch.cat((net(rho_batch), net(rho_other).transpose(2, 3)), dim=1)

    return c1.squeeze(0) if single_sample else c1


def c1_from_phi(net, rho, local_learning_mode=False, create_graph=None):
    """
    c1 = -d(sum phi)/d rho of the Phi-network by automatic differentiation (rho must require grad).

    Evaluation: c1 on every site, [2, H, W]. Training (local_learning_mode): only the complete
    central value of each (2k+1) window, [N, 2]; create_graph keeps the graph for training.
    """
    phi = phi_field(net, rho, local_learning_mode)

    if create_graph is None:
        create_graph = local_learning_mode

    # one backward pass gives both species
    grad_rho = torch.autograd.grad(phi.sum(), rho, create_graph=create_graph)[0]

    if local_learning_mode:
        return -grad_rho[..., grad_rho.shape[-2] // 2, grad_rho.shape[-1] // 2]
    return -grad_rho


def load_network(cfg, network_type, device):
    """
    A trained network of the run cfg in eval mode, with circular padding. TF32 is switched off,
    since the bulk free-energy differences are below its precision.
    """
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    state = torch.load(cfg.model_path(network_type), map_location=device)
    architecture = ARCHITECTURES[ARCHITECTURE_OF[network_type]]
    net = architecture(state["conv1.weight"].shape[-1]).to(device)
    net.load_state_dict(state)
    net.enable_padding()
    net.eval()
    return net


def c1_for_picard(net, network_type):
    """The function rho -> c1 (detached) of a loaded network, for the Picard iteration."""
    if network_type == "phi":
        def c1_of_rho(rho):
            rho_leaf = rho.detach().requires_grad_(True)
            with torch.enable_grad():
                c1 = c1_from_phi(net, rho_leaf, local_learning_mode=False)
            return c1.detach()
        return c1_of_rho
    if network_type in ("c1", "c1_regularised"):
        def c1_of_rho(rho):
            return c1_direct(net, rho).detach()
        return c1_of_rho
    raise ValueError(f"unknown network type '{network_type}'")
