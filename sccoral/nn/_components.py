from collections.abc import Iterable
from typing import Literal, Optional

import torch
from scvi.nn import FCLayers
from torch import Tensor, nn
from torch.distributions import Normal


def _no_grad_absolute(tensor: Tensor) -> Tensor:
    """Return absolute value of tensor"""
    with torch.no_grad():
        return torch.absolute(tensor)
    
def _no_grad_zero(tensor: Tensor) -> Tensor:
    """Return absolute value of tensor"""
    with torch.no_grad():
        return torch.zeros_like(tensor)


class LinearEncoder(nn.Module):
    """LinearEncoder for covariates

    Parameters
    ----------
    n_input
        Number of input dimensions
    n_output
        Number of output dimensions
    latent_distribution
        Normal distribution `normal` or lognormal `ln` (:cite:Svensson2020)
    return_dist
        Whether to return the distribution or samples
    mean_bias
        Whether to add bias term to linear layer of mean encoding
    var_bias
        Whether to add bias term to linear layer of variance encoding
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        latent_distribution: Literal["ln", "normal"] = "ln",
        return_dist: bool = False,
        mean_bias: bool = True,
        var_bias: bool = True,
        var_eps: float = 1e-4,
        init_positive: bool = True,
    ):
        super().__init__()

        self.mean = nn.Linear(n_input, n_output, bias=mean_bias)

        # Implement expected behaviour for positive class to be initialized
        # as the class with "high" factor activity.
        if init_positive:
            self.mean.weight.data = _no_grad_absolute(self.mean.weight.data)
            self.mean.bias.data = torch.nn.init.zeros_(self.mean.bias.data)

        self.var = nn.Linear(n_input, n_output, bias=var_bias)

        self.var_eps = var_eps

        if latent_distribution == "ln":
            self.z_transformation = nn.Sigmoid()
        else:
            # Identity function
            self.z_transformation = lambda x: x

        self.return_dist = return_dist

    def forward(self, x):
        q_m = self.mean(x)
        # Non-zero, positive variance (add var_eps for numerical stability)
        q_v = torch.exp(self.var(x)) + self.var_eps

        dist = Normal(q_m, torch.sqrt(q_v))
        latent = self.z_transformation(dist.rsample())

        if self.return_dist:
            return dist, latent

        return q_m, q_v, latent


class LinearDecoder(nn.Module):
    """Linear Decoder :cite:p:~Svensson20"""

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Optional[Iterable[int]] = None,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        bias: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.factor_loading = FCLayers(
            n_in=n_input,
            n_out=n_output,
            n_cat_list=n_cat_list,
            n_layers=1,
            use_activation=False,
            use_batch_norm=use_batch_norm,  # None
            use_layer_norm=use_layer_norm,
            bias=bias,
            dropout_rate=0,
            **kwargs,
        )

        self.px_dropout_decoder = FCLayers(
            n_in=n_input,
            n_out=n_output,
            n_cat_list=n_cat_list,  # None
            n_layers=1,
            use_activation=False,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            bias=bias,
            dropout_rate=0,
            **kwargs,
        )

    def forward(self, dispersion: str, z: torch.Tensor, library: torch.Tensor):
        raw_px_scale = self.factor_loading(z)
        px_scale = torch.softmax(raw_px_scale, dim=-1)
        px_dropout = self.px_dropout_decoder(z)
        px_rate = torch.exp(library) * px_scale
        px_r = None

        return px_scale, px_r, px_rate, px_dropout
