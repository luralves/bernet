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

#####################################################################################