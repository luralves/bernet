#####################################################################################
import torch

from typing import Mapping, Iterable, Literal, get_args

from bernet.contracts import ILoss

from bernet.utils.validation import TypeCheck, ValueCheck

#####################################################################################
class Tensor:
    """Wrapper around torch.Tensor that returns torch.Tensor on math ops."""

    def __init__(self, data: torch.Tensor, *, name: str):
        if not isinstance(data, torch.Tensor):
            raise TypeError("data must be a torch.Tensor")
        if not isinstance(name, str):
            raise TypeError("name must be a str")
        self.data = data
        self.name = name

    # Convenience
    def __repr__(self):
        return f"Tensor(name={self.name!r}, data={self.data!r})"

    def as_tensor(self) -> torch.Tensor:
        return self.data

    # Delegate attribute access to the underlying tensor.
    # Methods like .mean(), .sum() will be fetched from self.data and
    # will naturally return torch.Tensor.
    def __getattr__(self, attr):
        return getattr(self.data, attr)

    # ---- Binary ops: return torch.Tensor ----
    @staticmethod
    def _unwrap(x):
        return x.data if isinstance(x, Tensor) else x

    def __add__(self, other):
        return self.data + self._unwrap(other)

    def __radd__(self, other):
        return self._unwrap(other) + self.data

    def __sub__(self, other):
        return self.data - self._unwrap(other)

    def __rsub__(self, other):
        return self._unwrap(other) - self.data

    def __mul__(self, other):
        return self.data * self._unwrap(other)

    def __rmul__(self, other):
        return self._unwrap(other) * self.data

    def __truediv__(self, other):
        return self.data / self._unwrap(other)

    def __rtruediv__(self, other):
        return self._unwrap(other) / self.data

    def __pow__(self, other):
        return self.data ** self._unwrap(other)

    def __rpow__(self, other):
        return self._unwrap(other) ** self.data

    def __neg__(self):
        return -self.data

    # ---- torch.* functions support: unwrap inputs; return raw result ----
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        # Replace any Tensor wrappers in args/kwargs with their .data
        def un(x):
            if isinstance(x, Tensor): return x.data
            if isinstance(x, (list, tuple)):
                t = type(x)
                return t(un(i) for i in x)
            if isinstance(x, dict):
                return {k: un(v) for k, v in x.items()}
            return x

        result = func(*(un(a) for a in args), **un(kwargs))
        # IMPORTANT: we return the raw torch result (tensor / tuple of tensors)
        return result



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
        TypeCheck.iterable(losses)
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
    
    def get_parameter(self, name: str) -> torch.Tensor:
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
    def predict(self,
            model: torch.nn.Module,
            batch: Mapping[str, torch.Tensor],
            *,
            source: Source,
        ) -> None:
        """
        Compute residual loss.

        Parameters
        ----------
        model : torch.nn.Module
            PyTorch Module.
        batch : Mapping[str, torch.Tensor]
            Data for loss computation.
        source : Source
            Loss source.
        """

        #-- Validation
        TypeCheck.abc(model, torch.nn.Module)
        TypeCheck.mapping_tensor(batch)
        TypeCheck.str(source)

        ValueCheck.on_iterable(source, get_args(Source))

        #-- Empty current parameters and operators
        self.parameters["tensor"] = {}
        self._operators = {}

        #-- Add inputs to parameters
        for k, v in self.parameters["input"].items():
            
            #-- Separate tensor
            tensor = batch[:, v["axis"]:v["axis"] + v["dim"]].detach()

            # Add requires_grad if necessary
            if v["requires_grad"]:
                self.parameters["tensor"][k] = Tensor(tensor.requires_grad_(True), name=k)
            else:
                self.parameters["tensor"][k] = Tensor(tensor, name=k)

        #-- Create input
        x = torch.cat([v.data for _, v in self.parameters["tensor"].items()], dim=1)

        #-- Compute output
        y = model(x)

        #-- Add output to parameters
        for k, v in self.parameters["output"].items():
            tensor = y[:, v["axis"]:v["axis"] + v["dim"]]
            self.parameters["tensor"][k] = Tensor(tensor, name=k)

        #-- Add extra data to parameters
        if self.parameters.get("auxiliary", None) is not None:
            for k, v in self.parameters["auxiliary"].items():
                if source in v["losses"]:
                    tensor = batch[:, v["axis"]:v["axis"] + v["dim"]]
                    self.parameters["tensor"][k] = tensor
        
        return
    
    #-- Operators
    def partial(
            self,
            y: Tensor,
            x: Tensor,
        ) -> torch.Tensor:
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
        name = f"partial({y.name}, {x.name})"

        #-- Check if is already computed
        partial_y_x = self._operators.get(name, None)

        #-- Compute derivative if necessary
        if partial_y_x is None:

            #-- Compute operator
            partial_y_x = torch.autograd.grad(y.data, x.data, grad_outputs=torch.ones_like(y.data), create_graph=True)[0]

            #-- Save it
            self._operators[name] = partial_y_x
        
        return partial_y_x
    
    def partial2(self,
            y: Tensor,
            x: Tensor,
        ) -> torch.Tensor:
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
        name_p1 = f"partial({y.name}, {x.name})"
        name_p2 = f"partial2({y.name}, {x.name})"

        #-- Check if is already computed
        partial2_y_x = self._operators.get(name_p2, None)

        if partial2_y_x is not None:
            return partial2_y_x
        
        partial_y_x = self._operators.get(name_p1, None)

        #-- Compute first derivative if necessary
        if partial_y_x is None:

            #-- Compute operator
            partial_y_x = torch.autograd.grad(y.data, x.data, grad_outputs=torch.ones_like(y.data), create_graph=True)[0]

            #-- Save it
            self._operators[name_p1] = partial_y_x
        
        #-- Compute second derivative
        partial2_y_x = torch.autograd.grad(partial_y_x, x, grad_outputs=torch.ones_like(partial_y_x), create_graph=True)[0]

        #-- Save it
        self._operators[name_p2] = partial2_y_x
        
        return partial2_y_x

    def dot(
            self,
            y: Tensor,
            x: Tensor,
        ) -> torch.Tensor:
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
        name = f"dot({y.name}, {x.name})"

        #-- Check if is already computed
        dot_x_y = self._operators.get(name, None)

        #-- Compute derivative if necessary
        if dot_x_y is None:

            #-- Compute operator
            dot_x_y = torch.sum(x * y, dim=1)

            #-- Save it
            self._operators[name] = dot_x_y
        
        return dot_x_y

class LossBASE(ILoss):
    
    def __init__(self, operator: Operator):
        super().__init__()
        self._operator = operator
        return
    
    def get_parameter(self, name: str) -> torch.Tensor:
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
        tensor = self._operator.parameters["tensor"][name]

        return tensor

    def get_operator(self) -> Operator:
        """
        Get operator.
        """
        return self._operator
    
    #-- Partial implementation
    def residual(
            self,
            model: torch.nn.Module,
            batch: Mapping[str, torch.Tensor],
        ) -> torch.Tensor | None:
        super().residual(model, batch)
        self._operator.predict(model, batch, source="residual")
    
    def boundary(self,
            model: torch.nn.Module,
            batch: Mapping[str, torch.Tensor],
        ) -> torch.Tensor | None:
        super().boundary(model, batch)
        self._operator.predict(model, batch, source="boundary")
    
    def initial(self,
            model: torch.nn.Module,
            batch: Mapping[str, torch.Tensor],
        ) -> torch.Tensor | None:
        super().initial(model, batch)
        self._operator.predict(model, batch, source="initial")
    
    def observational(self,
            model: torch.nn.Module,
            batch: Mapping[str, torch.Tensor],
        ) -> torch.Tensor | None:
        super().observational(model, batch)
        self._operator.predict(model, batch, source="observational")

#####################################################################################