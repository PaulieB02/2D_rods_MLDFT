"""
The two network architectures (Sec. 3.2 and 3.3): Diomedes, the direct-c1 network (window
2k + 1), and Sarpedon, the Phi-network (window k + 1).

Both map the densities [N, 2, H, W] to one output channel [N, 1, H, W]: a window x window
convolution to 20 channels (conv1), then four 1x1 convolutions of width 256 with SiLU.
conv1 pads circularly for periodic boxes; for training on local windows the padding is off.
"""

import torch.nn as nn


class Diomedes(nn.Module):
    """Direct-c1 network: the output is c1 of species 0."""

    def __init__(self, window_size):
        super().__init__()
        # kept to restore the padding after training
        self.original_padding = window_size // 2
        self.original_padding_mode = "circular"

        hidden_layer_size = 256
        weighted_density_num = 20
        self.conv1 = nn.Conv2d(2, weighted_density_num, window_size, 1,   # 2 input channels: rho_0, rho_1
                               padding=self.original_padding,
                               padding_mode=self.original_padding_mode,
                               bias=False)

        self.rest_of_stack = nn.Sequential(
            nn.Conv2d(weighted_density_num, hidden_layer_size, 1, padding=0),
            nn.SiLU(),
            nn.Conv2d(hidden_layer_size, hidden_layer_size, 1, padding=0),
            nn.SiLU(),
            nn.Conv2d(hidden_layer_size, hidden_layer_size, 1, padding=0),
            nn.SiLU(),
            nn.Conv2d(hidden_layer_size, hidden_layer_size, 1, padding=0),
            nn.SiLU(),
            nn.Conv2d(hidden_layer_size, 1, 1, padding=0)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.rest_of_stack(x)
        return x

    def disable_padding(self):
        """Switch off the padding of conv1 (training on local windows)."""
        self.conv1.padding = (0, 0)
        # 'circular' is incompatible with padding 0
        self.conv1.padding_mode = 'zeros'

    def enable_padding(self):
        """Restore the circular padding of conv1 (evaluation on periodic boxes)."""
        self.conv1.padding = (self.original_padding, self.original_padding)
        self.conv1.padding_mode = self.original_padding_mode


class Sarpedon(nn.Module):
    """Phi-network: the output is the local excess free-energy density phi."""

    def __init__(self, window_size):
        super().__init__()
        act = nn.SiLU
        # kept to restore the padding after training
        self.original_padding = window_size // 2
        self.original_padding_mode = "circular"

        hidden_layer_size = 256
        weighted_density_num = 20
        # no bias: the weighted densities vanish together with the densities
        self.conv1 = nn.Conv2d(2, weighted_density_num, window_size, 1,   # 2 input channels: rho_0, rho_1
                               padding=self.original_padding,
                               padding_mode=self.original_padding_mode,
                               bias=False)

        self.rest_of_stack = nn.Sequential(
            nn.Conv2d(weighted_density_num, hidden_layer_size, 1, padding=0),
            act(),
            nn.Conv2d(hidden_layer_size, hidden_layer_size, 1, padding=0),
            act(),
            nn.Conv2d(hidden_layer_size, hidden_layer_size, 1, padding=0),
            act(),
            nn.Conv2d(hidden_layer_size, hidden_layer_size, 1, padding=0),
            act(),
            # no bias: a constant shift of phi does not change c1 and receives no gradient
            nn.Conv2d(hidden_layer_size, 1, 1, padding=0, bias=False)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.rest_of_stack(x)
        return x

    def disable_padding(self):
        """Switch off the padding of conv1 (training on local windows)."""
        self.conv1.padding = (0, 0)
        # 'circular' is incompatible with padding 0
        self.conv1.padding_mode = 'zeros'

    def enable_padding(self):
        """Restore the circular padding of conv1 (evaluation on periodic boxes)."""
        self.conv1.padding = (self.original_padding, self.original_padding)
        self.conv1.padding_mode = self.original_padding_mode


# architecture name -> class; config.ARCHITECTURE_OF assigns one to every network type
ARCHITECTURES = {
    "Sarpedon": Sarpedon,
    "Diomedes": Diomedes,
}
