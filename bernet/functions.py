# ------------------------------------------------------------------------------------ #
import math
import torch

from typing import Optional, Sequence

# ------------------------------------------------------------------------------------ #
def create_sequential_network(
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[Sequence[int]],
        activation: Optional[torch.nn.Module],
    ) -> torch.nn.Module:
    """
    Builds a direction's internal MLP.
    
    Parameters:
    -----------
        input_dim : int
            Dimension of the input latent vector.
        output_dim : int
            Dimension of the output vector (e.g., 1 for scalar outputs).
        hidden_dims : Optional[Sequence[int]]
            Sequence of integers specifying the number of neurons in each hidden layer.
        activation : Optional[torch.nn.Module]
            Activation function to use between layers (default is Tanh).

    Returns:
    --------
        torch.nn.Module
            A sequential model representing the MLP.
    """
    # -------------------------------------------------------------------------------- #
    # Define activation
    activation = activation or torch.nn.Tanh

    # -------------------------------------------------------------------------------- #
    # Check if there are hidden layers
    if hidden_dims is None:
        # ---------------------------------------------------------------------------- #
        # There are no hidden layers
        layers = [torch.nn.Linear(input_dim, output_dim)]

    # -------------------------------------------------------------------------------- #
    else:
        # ---------------------------------------------------------------------------- #
        # First layer
        layers = [torch.nn.Linear(input_dim, hidden_dims[0]), activation()]

        # ---------------------------------------------------------------------------- #
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(torch.nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(activation())

        # ---------------------------------------------------------------------------- #
        # Output layer
        layers.append(torch.nn.Linear(hidden_dims[-1], output_dim))

    # -------------------------------------------------------------------------------- #
    return torch.nn.Sequential(*layers)


# ------------------------------------------------------------------------------------ #
def outer_product(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """
    Computes the batched outer product of a sequence of (batch, N_i) tensors.

    Parameters:
    -----------
        tensors : Sequence[torch.Tensor]
            A sequence of (batch, N_i) tensors to compute the outer product.
            The batch dimension (dim 0) is preserved, not multiplied into.

    Returns:
    --------
        torch.Tensor
            Tensor of shape (batch, N_1, N_2, ..., N_m).
    """
    # -------------------------------------------------------------------------------- #
    result = tensors[0]

    for tensor in tensors[1:]:
        # Reshape `tensor` to broadcast against every non-batch dim already
        # in `result`, then add its own dim at the end. Plain
        # `result.unsqueeze(-1) * tensor` does NOT do this: it relies on
        # ordinary broadcasting, which aligns shapes from the right and
        # collides `tensor`'s own N_i against `result`'s last accumulated
        # dimension instead of the batch dimension.
        view_shape = (tensor.shape[0],) + (1,) * (result.ndim - 1) + (tensor.shape[1],)
        result = result.unsqueeze(-1) * tensor.view(view_shape)

    # -------------------------------------------------------------------------------- #
    return result


# ------------------------------------------------------------------------------------ #
def insert(
        tensor: torch.Tensor,
        dim: int,
        index: int,
    ) -> torch.Tensor:
    """
    Insert a slice filled with ones along a dimension.
    
    Parameters:
    -----------
        tensor : torch.Tensor
            The input tensor to modify.
        dim : int
            The dimension along which to insert the slice.
        index : int
            The index at which to insert the slice of ones.

    Returns:
    --------
        torch.Tensor
            The modified tensor with a slice of ones inserted.
    """

    if dim < 0:
        dim += tensor.ndim

    shape = list(tensor.shape)
    shape[dim] = 1

    ones = torch.ones(
        shape,
        dtype=tensor.dtype,
        device=tensor.device,
    )

    before = tensor.narrow(dim, 0, index)
    after = tensor.narrow(dim, index, tensor.shape[dim] - index)

    return torch.cat((before, ones, after), dim=dim)


# ------------------------------------------------------------------------------------ #
def select(tensor: torch.Tensor, dim: int, index: int) -> torch.Tensor:
    """
    Selects a single index along `dim`, keeping all other dims. Returns a
    view (not a copy), so it can be mutated in place (e.g. `.mul_()`,
    `.copy_()`) to write into `tensor`.

    Parameters:
    -----------
        tensor : torch.Tensor
            The input tensor.
        dim : int
            Dimension to index into.
        index : int
            Index to select along `dim`.

    Returns:
    --------
        torch.Tensor
            View of `tensor` with `dim` collapsed to `index`.
    """
    idx = [slice(None)] * tensor.ndim
    idx[dim] = index
    return tensor[tuple(idx)]


# ------------------------------------------------------------------------------------ #
def bernstein_basis(xi: torch.Tensor, n: int) -> torch.Tensor:
    """
    Computes the degree-n Bernstein basis polynomials at xi.

    w_i(xi) = C(n, i) * xi^i * (1 - xi)^(n - i),  for i = 0, ..., n

    Parameters:
    -----------
        xi : torch.Tensor
            Parametric coordinate in [0, 1], shape (batch,) or (batch, 1).
        n : int
            Degree of the basis (number of control points is n + 1).

    Returns:
    --------
        torch.Tensor
            Shape (batch, n + 1), column i holds w_i(xi) per batch sample.
    """
    # -------------------------------------------------------------------------------- #
    # Ensure shape (batch, 1) so it broadcasts against the (n + 1,) exponents
    xi = xi.reshape(-1, 1)

    i = torch.arange(n + 1, device=xi.device, dtype=xi.dtype)
    binom = torch.tensor(
        [math.comb(n, k) for k in range(n + 1)], device=xi.device, dtype=xi.dtype
    )

    # -------------------------------------------------------------------------------- #
    return binom * xi.pow(i) * (1 - xi).pow(n - i)

# ------------------------------------------------------------------------------------ #
