import torch

from typing import List, Literal, Optional

BCType = Literal["free", "dirichlet"]

class BoundaryCondition():

    def __init__(self, name: BCType, value: Optional[float | int] = None):
        self.name = name
        self.value = value
        return

    @property
    def remove(self) -> int:
        if self.name == "free":
            return 0
        elif self.name == "dirichlet":
            return 1
        return 0
    
    @property
    def points(self) -> List[torch.Tensor]:
        return [self.value]

class Neuron(torch.nn.Module):

    def __init__(
            self,
            degree: int,
            input_dim: int,
            boundary_conditions: List[BoundaryCondition],
            *,
            activation: torch.nn.Module = None,
            bias: bool = True,
            device: str | torch.device = "cpu",
            dtype: torch.dtype = torch.float32,
        ) -> None:
        super().__init__()
        
        #-- Inputs
        self.degree = degree
        self.input_dim = input_dim
        self.activation = activation
        self.bc_0, self.bc_1 = boundary_conditions

        #-- Trainable parameters
        n_out = self.degree - self.bc_0.remove - self.bc_1.remove
        self.linear = torch.nn.Linear(self.input_dim, n_out, bias, device, dtype)
        
        #-- Bernstien polynomial
        self.power = torch.arange(0, self.degree + 1, 1)
        self.binom = torch.tensor([torch.math.comb(i, self.degree) for i in range(self.degree + 1)])

        return

    def foward(self, x: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:

        # Linear transformation
        x = self.linear(x)

        # Activation function
        if self.activation:
            x = self.activation(x)

        # Construct control points
        x = torch.concat([self.bc_0.points, x, self.bc_1.points])

        # Bernstein polynomial
        y = x * self.binom * ( xi ** self.power ) * ( (1 - xi) ** (1 - self.power) )        

        return y