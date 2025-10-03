#####################################################################################
import torch

from typing import Mapping, List
from abc import ABC, abstractmethod

from bernet.contracts.loss import BatchLoss

#####################################################################################
class LoggerABC(ABC):
    """
    Stores a summary about the model and training data.
    """

    @abstractmethod
    def train_start(
            self,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
        ) -> None:
        """
        Called when training starts.

        Parameters
        ----------
        model : torch.nn.Module
            PyTorch Module.
        """
        ...
    
    @abstractmethod
    def epoch_end(
            self,
            losses: BatchLoss,
            metrics: List[float],
        ) -> None:
        """
        Called at the end of each epoch.

        Parameters
        ----------
        losses : Losses
            Dataclass with residual, boundary, initial and observational loss.
        metrics : Mapping[str, float]
            Dictionary with metrics data.
        """
        ...

    @abstractmethod
    def exception(
            self,
            e: BaseException,
        ) -> None:
        """
        Called when an exception occurs during training.

        Parameters
        ----------
        e : BaseException
            The exception that was raised.
        """
        ...
    
    @abstractmethod
    def training_end(
            self,
            stopped: bool,
        ) -> None:
        """
        Called when training ends.

        Parameters
        ----------
        stopped : bool
            True if stopped by EarlyStop
        """
        ...
    
    @abstractmethod
    def save(
            self,
            file: str,
        ) -> None:
        """
        Parameters
        ----------
        file : str
            The name of the file to save the data to.
        """
        ...
    
#####################################################################################