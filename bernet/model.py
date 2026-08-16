# ------------------------------------------------------------------------------------ #
import torch

from typing import Sequence

from bernet.contracts import Head, Direction
from bernet.functions import (
    create_sequential_network,
    outer_product,
    insert,
    select,
    bernstein_basis,
)

# ------------------------------------------------------------------------------------ #
class Bernstein(torch.nn.Module):
    """
    Neural network based on Bernstein polynomials.

    Each parametric direction xi_k has its own Head (MLP), which maps a
    latent state to that direction's free control points. `forward`
    combines the m directions' free control points via outer product into
    a single (batch, n_1 + 1, ..., n_m + 1) tensor, injects boundary
    conditions (Dirichlet and/or Neumann, per direction/edge) directly
    into that tensor, and contracts it against the Bernstein basis to get
    the scalar output phi.

    Boundary conditions are structural (hard constraints), not penalized
    in a loss: a Dirichlet value replaces its control point outright; a
    Neumann derivative sets the adjacent control point via a closed-form
    formula. Both work by inserting a slice that's constant across every
    OTHER direction (via the Bernstein partition-of-unity property), so a
    condition on one direction doesn't get scaled by the others' free
    points.

    Limitation: if two or more directions each fix a point that lands on
    the same tensor position (e.g. both fixing xi_k = 0 — a shared
    corner), there's no way to satisfy both exactly — a genuine
    discontinuity this single rank-1 tensor-product structure can't
    represent. Whichever direction is listed LAST in `forward` wins at
    that position; nearby points blend toward it, decaying over a region
    whose width depends on that direction's Bernstein degree. Prescribe
    compatible values at shared corners to avoid this.
    """

    def __init__(self, heads: Sequence[Head]) -> None:
        super().__init__()
        self.settings = heads
        self.networks = [
            create_sequential_network(
                input_dim=setting.input_dim,
                output_dim=setting.output_dim,
                hidden_dims=setting.hidden_dims,
                activation=setting.activation,
            ) for setting in self.settings
        ]

    def forward(self, directions: Sequence[Direction]) -> torch.Tensor:

        # Step 1: Head -> free control points, one (batch, N_k) per direction
        ctrl_points = [
            self.networks[i](directions[i].latent)
            for i in range(len(directions))
        ]

        # Step 2: control points tensor, (batch, N_1, ..., N_m)
        tensor = outer_product(ctrl_points)

        # Step 3: inject boundary conditions
        for dim, direction in enumerate(directions):
            axis = dim + 1  # dim 0 of `tensor` is batch

            # Degree n_k: free points so far, plus one per condition below.
            n = tensor.shape[axis] - 1
            n += sum(
                x is not None for x in (
                    direction.phi_0, direction.phi_1,
                    direction.dphi_dxi_0, direction.dphi_dxi_1,
                )
            )
            view_shape = (tensor.shape[0],) + (1,) * (tensor.ndim - 2)

            if direction.phi_0 is not None:
                tensor = insert(tensor, axis, 0)
                select(tensor, axis, 0).mul_(direction.phi_0.view(view_shape))

            if direction.phi_1 is not None:
                tensor = insert(tensor, axis, tensor.shape[axis])
                select(tensor, axis, tensor.shape[axis] - 1).mul_(direction.phi_1.view(view_shape))

            if direction.dphi_dxi_0 is not None:
                p_0 = select(tensor, axis, 0)
                tensor = insert(tensor, axis, 1)
                select(tensor, axis, 1).copy_(p_0 + direction.dphi_dxi_0.view(view_shape) / n)

            if direction.dphi_dxi_1 is not None:
                p_n = select(tensor, axis, tensor.shape[axis] - 1)
                insert_at = tensor.shape[axis] - 1
                tensor = insert(tensor, axis, insert_at)
                select(tensor, axis, insert_at).copy_(p_n - direction.dphi_dxi_1.view(view_shape) / n)

        # Step 4: Bernstein basis per direction, using the final degree n_k
        basis = [
            bernstein_basis(directions[i].xi, tensor.shape[i + 1] - 1)
            for i in range(len(directions))
        ]

        # Step 5: complete tensor = control points * basis, elementwise
        basis_tensor = outer_product(basis)
        complete_tensor = tensor * basis_tensor

        # Step 6: reduce every direction dim, (batch, N_1+1, ..., N_m+1) -> (batch,)
        phi = complete_tensor.sum(dim=tuple(range(1, complete_tensor.ndim)))

        return phi

# ------------------------------------------------------------------------------------ #
