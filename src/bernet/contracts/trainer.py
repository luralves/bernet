#####################################################################################
import torch
import numpy.typing as npt

from abc import ABC, abstractmethod
from typing import Mapping

from bernet.contracts.sampler import ISampler
from bernet.contracts.loss import ILoss

from bernet.utils.validation import TypeCheck

#####################################################################################
class ITrainer(ABC):
    """
    Trainer class for training neural networks.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            sampler: ISampler,
            loss: ILoss,
            optimizer: torch.optim.Optimizer,
        ) -> None:
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

        #-- Verification
        TypeCheck.abc(model, torch.nn.Module)
        TypeCheck.abc(sampler, ISampler)
        TypeCheck.abc(loss, ILoss)
        TypeCheck.abc(optimizer, torch.optim.Optimizer)

        #-- Inputs
        self._model = model
        self._sampler = sampler
        self._loss = loss
        self._optimizer = optimizer

        return


    @abstractmethod
    def fit(
            self,
            num_epochs: int,
            verbose: bool,
        ) -> Mapping[str, npt.NDArray]:
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
        Mapping[str, npt.NDArray]
            Dict containing training data, such as losses, test,
            and validation data.
        """
        ...
    
#####################################################################################