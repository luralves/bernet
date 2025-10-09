#####################################################################################
from abc import ABC, abstractmethod
from typing import Any

from bernet.interface.abstract.sampler import ISampler
from bernet.interface.abstract.loss import ILoss
from bernet.interface.typing.aliases import Model, Optimizer
from bernet.utils.validation import TypeCheck

#####################################################################################
class ITrainer(ABC):
    """
    Trainer class for training neural networks.
    """

    def __init__(
            self,
            model: Model,
            sampler: ISampler,
            loss: ILoss,
            optimizer: Optimizer) -> None:
        """
        Parameters
        ----------
        model : nn.Module
            The model to be trained.
        sampler : ISampler
            The sampler to be used for data loading.
        loss : ILoss
            The loss function to be used during training.
        optimizer : torch.optim.Optimizer
            The optimizer to be used for updating model weights.
        """
        super().__init__()

        #-- Validation
        TypeCheck.abc(model, Model)
        TypeCheck.abc(sampler, ISampler)
        TypeCheck.abc(loss, ILoss)
        TypeCheck.abc(optimizer, Optimizer)

        #-- Inputs
        self.model = model
        self.sampler = sampler
        self.loss = loss
        self.optimizer = optimizer

        return


    @abstractmethod
    def fit(self, num_epochs: int, verbose: bool) -> Any:
        """
        Perform training.

        Parameters
        ----------
        num_epochs : int
            Number of epochs to train the model for.
        verbose : bool
            Show training progress.
        
        Returns
        -------
        Any
            Training information.
        """
        ...
    
#####################################################################################