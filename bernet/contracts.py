# ------------------------------------------------------------------------------------ #
import torch

from dataclasses import dataclass
from typing import Optional, Sequence

# ------------------------------------------------------------------------------------ #
@dataclass(frozen=True, slots=True)
class Direction():
    # -------------------------------------------------------------------------------- #
    latent: torch.Tensor
    xi: torch.Tensor
    phi_0: Optional[torch.Tensor] = None
    phi_1: Optional[torch.Tensor] = None
    dphi_dxi_0: Optional[torch.Tensor] = None
    dphi_dxi_1: Optional[torch.Tensor] = None
    # -------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------ #
@dataclass(frozen=True, slots=True)
class Head():
    # -------------------------------------------------------------------------------- #
    input_dim: int
    output_dim: int
    hidden_dims: Optional[Sequence[int]] = None
    activation: Optional[torch.nn.Module] = None
    # -------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------ #
