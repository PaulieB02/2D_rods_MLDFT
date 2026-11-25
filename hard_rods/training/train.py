"""
Losses and the training loop (Sec. 3.1 and Appendix B). The regularised direct-c1 network
minimises 0.8 L_c1 + 0.2 L_c2, with the mixed-derivative penalty L_c2.
"""

import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.func import jvp
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..functionals.neural import c1_direct, c1_from_phi
from .dataset import WindowDataset

# weight of the c2 penalty for the regularised network
lambda_c2 = 0.2


def training_window_sizes(k, network_type):
    """(density window, c1 window) for training a network of this type."""
    if network_type == "c1_regularised":
        # c2_loss needs c1 on the (2k+1) neighbourhood of the centre
        return 4 * k + 1, 1
    return 2 * k + 1, 1


def c2_loss(net, rho_windows):
    """
    Mixed-derivative penalty: mean squared violation of
    d c1_i(centre)/d rho_j(r') = d c1_j(r')/d rho_i(centre), summed over the species pairs.
    The left side is taken in reverse mode, the right side in forward mode (jvp).
    """
    rho = rho_windows.detach().requires_grad_(True)
    c1 = c1_direct(net, rho)              # [N, 2, h, w]: c1 of both species on the output field

    # centre of the c1 field, and centre of the (wider) density window that produced it
    out_y, out_x = c1.shape[-2] // 2, c1.shape[-1] // 2
    in_y, in_x = rho.shape[-2] // 2, rho.shape[-1] // 2

    # reverse mode: d c1_i(centre)/d rho_j(r'); the sum keeps each sample's own gradient
    d_c10_d_rho = torch.autograd.grad(c1[:, 0, out_y, out_x].sum(), rho, create_graph=True)[0]
    d_c11_d_rho = torch.autograd.grad(c1[:, 1, out_y, out_x].sum(), rho, create_graph=True)[0]

    # trim to the sites r' of the c1 field, so both sides are indexed alike
    pad = (rho.shape[-1] - c1.shape[-1]) // 2
    if pad > 0:
        d_c10_d_rho = d_c10_d_rho[:, :, pad:-pad, pad:-pad]      # [N, 2, h, w]
        d_c11_d_rho = d_c11_d_rho[:, :, pad:-pad, pad:-pad]

    # forward mode: d c1_j(r') / d rho_i(centre), with a unit tangent on one central bin
    tangent_rho0 = torch.zeros_like(rho)
    tangent_rho0[:, 0, in_y, in_x] = 1.0
    tangent_rho1 = torch.zeros_like(rho)
    tangent_rho1[:, 1, in_y, in_x] = 1.0

    def c1_of(x):
        return c1_direct(net, x)

    d_c1_d_rho0_centre = jvp(c1_of, (rho,), (tangent_rho0,))[1]     # [N, 2, h, w]
    d_c1_d_rho1_centre = jvp(c1_of, (rho,), (tangent_rho1,))[1]

    # same species: d c1_0(centre)/d rho_0(r') == d c1_0(r')/d rho_0(centre), and 1 <-> 1
    loss_00 = F.mse_loss(d_c10_d_rho[:, 0], d_c1_d_rho0_centre[:, 0])
    loss_11 = F.mse_loss(d_c11_d_rho[:, 1], d_c1_d_rho1_centre[:, 1])

    # cross species: d c1_0(centre)/d rho_1(r') == d c1_1(r')/d rho_0(centre), and 1 <-> 0
    loss_01 = F.mse_loss(d_c10_d_rho[:, 1], d_c1_d_rho0_centre[:, 1])
    loss_10 = F.mse_loss(d_c11_d_rho[:, 0], d_c1_d_rho1_centre[:, 0])

    return loss_00 + loss_11 + loss_01 + loss_10


def c1_loss(net, network_type, rho_windows, c1_target_values, create_graph=True):
    """
    Mean squared c1 error at the window centres, over the finite targets only. For the
    Phi-network rho_windows must require grad; create_graph=False for validation.
    """
    if network_type == "phi":
        c1_predicted = c1_from_phi(net, rho_windows, local_learning_mode=True,
                                   create_graph=create_graph)                       # [N, 2]
    else:
        c1_predicted = c1_direct(net, rho_windows)                                   # [N, 2, h, w]
        # only the centre is compared (h = w = 1 for the plain direct-c1 network)
        cy, cx = c1_predicted.shape[-2] // 2, c1_predicted.shape[-1] // 2
        c1_predicted = c1_predicted[:, :, cy:cy + 1, cx:cx + 1]                      # [N, 2, 1, 1]
    c1_target = c1_target_values.reshape(c1_predicted.shape)

    # infinite targets (hard walls) are zeroed first: masking afterwards would give 0 * inf = nan
    finite = torch.isfinite(c1_target)
    c1_target = torch.where(finite, c1_target, torch.zeros_like(c1_target))

    squared_error = (c1_predicted - c1_target) ** 2 * finite
    loss = squared_error.sum() / finite.sum().clamp(min=1)
    return loss


def train_model(net, network_type, cfg, device):
    """
    Train net with cfg.training[network_type] on the training profiles, monitoring the c1 loss
    on the validation profiles; the weights are saved to cfg.model_path every 10 epochs.
    """
    settings = cfg.training[network_type]
    k = cfg.k
    model_path = cfg.model_path(network_type)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    rho_window_size, c1_window_size = training_window_sizes(k, network_type)
    dataset_training = WindowDataset(cfg.profile_dir("training"), cfg.n_training,
                                     rho_window_size, c1_window_size,
                                     cfg.max_packing_fraction_training, k)
    dataset_validation = WindowDataset(cfg.profile_dir("validation"), cfg.n_validation,
                                       rho_window_size, c1_window_size,
                                       cfg.max_packing_fraction_training, k)

    writer = SummaryWriter(log_dir=cfg.tensorboard_dir(network_type))
    dataloader_training = DataLoader(dataset=dataset_training, batch_size=settings.batch_size,
                                     shuffle=True, num_workers=16, pin_memory=True,
                                     prefetch_factor=5, persistent_workers=True)
    dataloader_validation = DataLoader(dataset=dataset_validation, batch_size=settings.batch_size,
                                       shuffle=False, num_workers=2)

    optimizer = optim.Adam(net.parameters(), lr=settings.learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.8)
    net.disable_padding()

    # the Phi-network's c1 is a derivative with respect to the density windows
    needs_rho_grad = network_type == "phi"

    for epoch in range(settings.num_epochs):
        net.train()
        for rho_windows, c1_values in dataloader_training:
            rho_windows = rho_windows.to(device, non_blocking=True).requires_grad_(needs_rho_grad)
            c1_values = c1_values.to(device, non_blocking=True)

            loss = c1_loss(net, network_type, rho_windows, c1_values)
            if network_type == "c1_regularised":
                loss = (1 - lambda_c2) * loss + lambda_c2 * c2_loss(net, rho_windows)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # validation (the Phi-network needs gradients here too)
        net.eval()
        val_data_loss = 0.0
        with torch.set_grad_enabled(needs_rho_grad):
            for rho_windows, c1_values in dataloader_validation:
                rho_windows = rho_windows.to(device, non_blocking=True).requires_grad_(needs_rho_grad)
                c1_values = c1_values.to(device, non_blocking=True)
                val_data_loss += c1_loss(net, network_type, rho_windows, c1_values,
                                         create_graph=False).item()
        val_data_loss /= len(dataloader_validation)

        print(f"epoch {epoch+1}/{settings.num_epochs}, val_data={val_data_loss:.4f}")
        writer.add_scalar('val/data_loss', val_data_loss, epoch)
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            torch.save(net.state_dict(), model_path)
