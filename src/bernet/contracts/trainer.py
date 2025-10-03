#####################################################################################
import torch

from abc import ABC, abstractmethod
from typing import Optional, List

from bernet.contracts.sampler import SamplerABC
from bernet.contracts.loss import LossABC
from bernet.contracts.metrics import MetricsABC
from bernet.contracts.callbacks import CallbacksABC
from bernet.contracts.logger import LoggerABC
from bernet.contracts.early_stop import EarlyStopABC

#####################################################################################
class TrainerABC(ABC):
    """
    Trainer class for training neural networks.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            sampler: SamplerABC,
            loss: LossABC,
            optimizer: torch.optim.Optimizer,
            metrics: Optional[List[MetricsABC]] = None,
            callbacks: Optional[CallbacksABC] = None,
            logger: Optional[LoggerABC] = None,
            early_stop: Optional[EarlyStopABC] = None,
            device: Optional[str | torch.device] = "cpu",
        ) -> None:
        """
        Parameters
        ----------
        model : nn.Module
            The model to be trained.
        sampler : SamplerABC
            The sampler to be used for data loading.
        loss : LossABC
            The loss function to be used during training.
        optimizer : torch.optim.Optimizer
            The optimizer to be used for updating model weights.
        metrics : Optional[MetricsABC]
            The metrics to be used for evaluation.
        callback : Optional[CallbackABC]
            The callback for training events.
        logger : Optional[LoggerABC]
            The logger for logging training progress.
        early_stop : Optional[IEarlyStop]
            The early stop functionality.
        device : device
            The device to run the training on. If "auto", selects GPU if available,
        """
        super().__init__()

        #-- Inputs
        self.model = model
        self.sampler = sampler
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.callbacks = callbacks
        self.early_stop = early_stop
        self.logger = logger

        #-- Computed parameters
        self.device = None

        #-- Select device
        if device == "auto":
            if torch.cuda.is_available():
                self.device =  torch.device("cuda")
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self.device =  torch.device("mps")
            else:
                self.device =  torch.device("cpu")
        else:
            self.device = torch.device(device) if isinstance(device, str) else device

        #-- Move model to device
        self.model.to(self.device)

        return


    @abstractmethod
    def fit(
            self,
            num_epochs: int,
            verbose: bool,
        ) -> None:
        """
        Perform training.

        Parameters
        ----------
        num_epochs : int
            Number of epochs to train the model for.
        verbose : bool
            Show training progress.
        """
        ...
    
#####################################################################################