#####################################################################################
import torch

from bernet.interface.abstract.loss import ILoss
from bernet.interface.typing.aliases import Model, Tensor, TensorData
from bernet.interface.typing.dataclass import Losses
from bernet.interface.typing.structs import Batch
from bernet.utils.validation.type_check import TypeCheck
from bernet.core.operator.operator import Operator

#####################################################################################
class LossBASE(ILoss):
    
    def __init__(self, operator: Operator):
        super().__init__()
        self._operator = operator
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
        tensor = self._operator.parameters["tensor"][name]

        return tensor

    def get_operator(self) -> Operator:
        return self._operator
    
    #-- Partial implementation
    def residual(self, model: Model, batch: TensorData) -> Tensor | None:
        super().residual(model, batch)
        self._operator.predict(model, batch, source="residual")
    
    def boundary(self, model: Model, batch: TensorData) -> Tensor | None:
        super().boundary(model, batch)
        self._operator.predict(model, batch, source="boundary")
    
    def initial(self, model: Model, batch: TensorData) -> Tensor | None:
        super().initial(model, batch)
        self._operator.predict(model, batch, source="initial")
    
    def observational(self, model: Model, batch: TensorData) -> Tensor | None:
        super().observational(model, batch)
        self._operator.predict(model, batch, source="observational")
    
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