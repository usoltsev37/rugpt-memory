import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseNetwork(nn.Module):
    def __init__(
        self,
        n_hid_layers: int = 1,
        input_dim: int = 768,
        hidden_dim: int = 768 * 2,
        out_dim: int = 768,
        dtype: torch.dtype = torch.float32,
        dropout: float = 0.0,
        initialize_with_zeros=False,
    ):
        """
        Dense network with adjustable number of hidden layers.
        :param n_hid_layers: Number of hidden layers
        :param input_dim: Size of each input sample
        :param hidden_dim: Size of each hidden sample
        :param out_dim: Size of each output sample
        :param dtype: Data type of module parameters
        :param dropout: Dropout rate to apply after each layer
        """
        super().__init__()
        self.act = F.gelu

        if n_hid_layers:
            layers = (
                [nn.Linear(input_dim, hidden_dim), nn.Dropout(dropout)]
                if dropout > 0
                else [nn.Linear(input_dim, hidden_dim)]
            )

            for _ in range(n_hid_layers - 1):
                layers += (
                    [nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout)]
                    if dropout > 0
                    else [nn.Linear(hidden_dim, hidden_dim)]
                )

            layers.append(nn.Linear(hidden_dim, out_dim))

        else:
            layers = [nn.Linear(input_dim, out_dim)]

        self.layers = nn.Sequential(*layers)

        if initialize_with_zeros:
            for param in self.layers[-1].parameters():
                nn.init.zeros_(param)
                

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # to do: check activation function
        if len(self.layers) > 1:
            for layer in self.layers[:-1]:
                x = self.act(layer(x)) if isinstance(layer, nn.Linear) else layer(x)
            return self.layers[-1](x)
        else:
            return self.act(self.layers[0](x))
