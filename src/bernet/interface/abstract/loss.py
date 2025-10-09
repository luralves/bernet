#####################################################################################
from abc import ABC, abstractmethod

from bernet.interface.typing.aliases import Tensor, Model, TensorData
from bernet.interface.typing.dataclass import Losses
from bernet.interface.typing.structs import Batch
from bernet.utils.validation.type_check import TypeCheck

#####################################################################################
class ILoss(ABC):
    """
    Compute loss terms based on sampled data
    
    Notes
    -----
    The method 'compute(...)' is already implemented and can be
    called from a 'Trainer(ITrainer)' to obtain the losses.
    """

    @abstractmethod
    def residual(self, model: Model, batch: TensorData) -> Tensor:
        """
        Compute residual loss.

        Parameters
        ----------
        model : Model
            PyTorch Module.
        batch : TensorData
            Data for loss computation.
        
        Returns
        -------
        Tensor
            Tensor loss.
        """
        ...
    
    @abstractmethod
    def boundary(self, model: Model, batch: TensorData) -> Tensor:
        """
        Compute boundary loss.

        Parameters
        ----------
        model : Model
            PyTorch Module.
        batch : TensorData
            Data for loss computation.
        
        Returns
        -------
        Tensor
            Tensor loss.
        """
        ...

    @abstractmethod
    def initial(self, model: Model, batch: TensorData) -> Tensor:
        """
        Compute initial loss.

        Parameters
        ----------
        model : Model
            PyTorch Module.
        batch : TensorData
            Data for loss computation.
        
        Returns
        -------
        Tensor
            Tensor loss.
        """
        ...
    
    @abstractmethod
    def observational(self, model: Model, batch: TensorData) -> Tensor:
        """
        Compute observational loss.

        Parameters
        ----------
        model : Model
            PyTorch Module.
        batch : TensorData
            Data for loss computation.
        
        Returns
        -------
        Tensor
            Tensor loss.
        """
        ...
    
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
        TypeCheck.abc(model, Model)
        TypeCheck.abc(batch, Batch)

        #-- Losses
        losses = Losses(None, None, None, None)

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