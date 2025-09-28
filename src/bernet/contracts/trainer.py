#####################################################################################
import torch

from abc import ABC, abstractmethod
from typing import Optional

from bernet.contracts.sampler import ISampler
from bernet.contracts.loss import ILoss
from bernet.contracts.metrics import IMetrics
from bernet.contracts.callbacks import ICallbacks
from bernet.contracts.logger import ILogger
from bernet.contracts.early_stop import IEarlyStop

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
            metrics: Optional[IMetrics] = None,
            callbacks: Optional[ICallbacks] = None,
            logger: Optional[ILogger] = None,
            early_stop: Optional[IEarlyStop] = None,
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