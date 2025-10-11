#####################################################################################
import torch

from typing import Mapping, Iterable, Literal, get_args

from bernet.interface.typing.aliases import Model, Tensor
from bernet.utils.validation import TypeCheck, ValueCheck

#####################################################################################
IO = Literal["in", "out", "none"]
Source = Literal["residual", "boundary", "initial", "observational"]

class Operator():

    def __init__(self):
        self.parameters = {} # {"tensor": {...}, "input": {...}, "output": {...}, "auxiliary": {...}}
        self._operators = {}
        return
    
    #-- Getters and setters [parameters]
    def set_parameter(
            self,
            name: str,
            *,
            axis: int = 0,
            dim: int = 1,
            io: IO = "in",
            losses: Iterable[Source] = [],
            requires_grad: bool = False,
        ) -> None:
        """
        Set a new parameter.

        Parameters
        ----------
        name : str
            Parameter name.
        axis : int
            Column axis defining the beginning of the parameter.
        dim : int
            Dimension of the parameter.
        io : IO
            Define if the parameter is an input or output.
        losses : LossType
            Defines at which loss the parameter is required.
        requires_grad : bool
            True if the parameter will be used in autodiff.
        """

        #-- Validation
        TypeCheck.str(name)
        TypeCheck.int(axis)
        TypeCheck.int(dim)
        TypeCheck.str(io)
        TypeCheck.sequence(losses)
        TypeCheck.bool(requires_grad)

        ValueCheck.on_iterable(io, get_args(IO))
        for loss in losses: ValueCheck.on_iterable(loss, get_args(Source))

        #-- Store data
        loc = "input" if io == "in" else "output" if io == "out" else "auxiliary"

        #-- Create dict
        if self.parameters.get(loc, None) is None:
            self.parameters[loc] = {}
            
        #-- Add parameter
        self.parameters[loc][name] = {
            "axis": axis,
            "dim": dim,
            "io": io,
            "losses": losses,
            "requires_grad": requires_grad,
        }

        return
    
    def get_parameter(self, name: str) -> Tensor:
        """
        Get the parameter based on its name.

        Parameters
        ----------
        name : str
            Parameter name.
        
        Returns
        -------
        Tensor
            Current tensor associated with 'name'.
        """

        #-- Validation
        TypeCheck.str(name)

        #-- Get tensor
        tensor = self.parameters["tensor"][name]

        return tensor
    
    #--
    def predict(self, model: Model, batch: Tensor, *, source: Source) -> None:
        """
        Compute residual loss.

        Parameters
        ----------
        model : torch.nn.Module
            PyTorch Module.
        batch : Mapping[str, Tensor]
            Data for loss computation.
        source : Source
            Loss source.
        """
        
        #-- Empty current parameters and operators
        self.parameters["tensor"] = {}
        self._operators = {}

        #-- Add inputs to parameters
        for k, v in self.parameters["input"].items():
            
            #-- Separate tensor
            tensor = batch[:, v["axis"]:v["axis"] + v["dim"]].detach()

            # Add requires_grad if necessary
            if v["requires_grad"]:
                self.parameters["tensor"][k] = tensor.requires_grad_(True)
            else:
                self.parameters["tensor"][k] = tensor

        #-- Create input
        x = torch.cat([v for _, v in self.parameters["tensor"].items()], dim=1)

        #-- Compute output
        y = model(x)

        #-- Add output to parameters
        for k, v in self.parameters["output"].items():
            tensor = y[:, v["axis"]:v["axis"] + v["dim"]]
            self.parameters["tensor"][k] = tensor

        #-- Add extra data to parameters
        if self.parameters.get("auxiliary", None) is not None:
            for k, v in self.parameters["auxiliary"].items():
                if source in v["losses"]:
                    tensor = batch[:, v["axis"]:v["axis"] + v["dim"]]
                    self.parameters["tensor"][k] = tensor
        
        return
    
    #-- Operators
    def partial(self, y: Tensor, x: Tensor) -> Tensor:
        """
        Compute the first partial derivative.

        Parameters
        ----------
        y : Tensor
            Output parameter.
        x : Tensor
            Input parameter.
        
        Returns
        -------
        Tensor
            First derivative.
        """

        #-- Operator name
        name = f"partial({id(y)}, {id(x)})"

        #-- Check if is already computed
        partial_y_x = self._operators.get(name, None)

        #-- Compute derivative if necessary
        if partial_y_x is None:

            #-- Compute operator
            partial_y_x = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]

            #-- Save it
            self._operators[name] = partial_y_x
        
        return partial_y_x
    
    def partial2(self, y: Tensor, x: Tensor) -> Tensor:
        """
        Compute the first partial derivative.

        Parameters
        ----------
        y : Tensor
            Output parameter.
        x : Tensor
            Input parameter.
        
        Returns
        -------
        Tensor
            First derivative.
        """

        #-- Operator name
        name_p1 = f"partial({id(y)}, {id(x)})"
        name_p2 = f"partial2({id(y)}, {id(x)})"

        #-- Check if is already computed
        partial2_y_x = self._operators.get(name_p2, None)

        if partial2_y_x is not None:
            return partial2_y_x
        
        partial_y_x = self._operators.get(name_p1, None)

        #-- Compute first derivative if necessary
        if partial_y_x is None:

            #-- Compute operator
            partial_y_x = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]

            #-- Save it
            self._operators[name_p1] = partial_y_x
        
        #-- Compute second derivative
        partial2_y_x = torch.autograd.grad(partial_y_x, x, grad_outputs=torch.ones_like(partial_y_x), create_graph=True)[0]

        #-- Save it
        self._operators[name_p2] = partial2_y_x
        
        return partial2_y_x

    def dot(self, y: Tensor, x: Tensor) -> Tensor:
        """
        Compute the dot product.

        Parameters
        ----------
        y : Tensor
            Output parameter.
        x : Tensor
            Input parameter.
        
        Returns
        -------
        Tensor
            Dot product.
        """

        #-- Operator name
        name = f"dot({id(y)}, {id(x)})"

        #-- Check if is already computed
        dot_x_y = self._operators.get(name, None)

        #-- Compute derivative if necessary
        if dot_x_y is None:

            #-- Compute operator
            dot_x_y = torch.sum(x * y, dim=1)

            #-- Save it
            self._operators[name] = dot_x_y
        
        return dot_x_y

#####################################################################################