#####################################################################################
import torch

from typing import Callable

from bernet.interface.abstract.loss import ILoss
from bernet.interface.typing.aliases import Model, Tensor, TensorData
from bernet.interface.typing.structs import Batch
from bernet.interface.typing.dataclass import Losses
from bernet.utils.processing.error import Error
from bernet.utils.validation import TypeCheck

#####################################################################################
class LossBASE(ILoss):

    def __init__(self, func: Callable = Error.mse):
        """
        Parameters
        ----------
        func : Callable
            Function used to compute the loss based on expected
            predicted values.
        """
        super().__init__()

        #-- Verification
        TypeCheck.callable(func)

        #-- Inputs
        self._func = func

        return

    #-- Needs to be implemented
    def residual(self, model: Model, batch: TensorData) -> Tensor:
        ...
    
    #-- Override
    def boundary(self, model: Model, batch: TensorData) -> Tensor:

        if TypeCheck.sequence(batch, stop=False):
            y_hat = model(batch[0])
            loss = self._func(y_hat, batch[1])

        elif TypeCheck.mapping(batch, stop=False):

            #-- Get default parameters
            x = batch.get("x", None)
            y = batch.get("y", None)
            
            #-- Check if inputs/outputs are tensors
            TypeCheck.generic(x, [Tensor])
            TypeCheck.generic(y, [Tensor])
            
            #-- Compute output
            y_hat = model(batch["x"])
            loss = self._func(y_hat, batch["y"])

        else:
            raise TypeError("TypeError: Batch is not structured correctly.")

        return loss

    #-- Override
    def initial(self, model: Model, batch: TensorData) -> Tensor:

        if TypeCheck.sequence(batch, stop=False):
            y_hat = model(batch[0])
            loss = self._func(y_hat, batch[1])

        elif TypeCheck.mapping(batch, stop=False):

            #-- Get default parameters
            x = batch.get("x", None)
            y = batch.get("y", None)
            
            #-- Check if inputs/outputs are tensors
            TypeCheck.generic(x, [Tensor])
            TypeCheck.generic(y, [Tensor])
            
            #-- Compute output
            y_hat = model(batch["x"])
            loss = self._func(y_hat, batch["y"])

        else:
            raise TypeError("TypeError: Batch is not structured correctly.")
        
        return loss
    
    #-- Override
    def observational(self, model: Model, batch: TensorData) -> Tensor:

        if TypeCheck.sequence(batch, stop=False):
            y_hat = model(batch[0])
            loss = self._func(y_hat, batch[1])

        elif TypeCheck.mapping(batch, stop=False):

            #-- Get default parameters
            x = batch.get("x", None)
            y = batch.get("y", None)
            
            #-- Check if inputs/outputs are tensors
            TypeCheck.generic(x, [Tensor])
            TypeCheck.generic(y, [Tensor])
            
            #-- Compute output
            y_hat = model(batch["x"])
            loss = self._func(y_hat, batch["y"])

        else:
            raise TypeError("TypeError: Batch is not structured correctly.")
        
        return loss
    
    def compute(self, model: Model, batch: Batch) -> Losses:
        """
        Compute the losses based on the methods implemented.

        Parameters
        ----------
        model : Model
            PyTorch Module.
        batch : Batch
            Data for loss computation.
        
        Returns
        -------
        Losses
            Dataclass with the losses computed.
        """

        #-- Validation
        TypeCheck.generic(model, [Model])
        TypeCheck.generic(batch, [Batch])

        #-- Losses
        losses = Losses(
            residual=torch.tensor(0.0),
            boundary=torch.tensor(0.0),
            initial=torch.tensor(0.0),
            observational=torch.tensor(0.0),
        )

        #-- Add losses
        if batch.residual is not None:
            losses.residual = self.residual(model, batch.residual)
        
        if batch.boundary is not None:
            losses.boundary = self.boundary(model, batch.boundary)
        
        if batch.initial is not None:
            losses.initial = self.initial(model, batch.initial)
        
        if batch.observational is not None:
            losses.observational = self.observational(model, batch.observational)
        
        return losses

#####################################################################################