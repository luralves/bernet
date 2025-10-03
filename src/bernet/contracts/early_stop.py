#####################################################################################
from typing import List, Optional
from abc import ABC, abstractmethod

from bernet.contracts.loss import BatchLoss

#####################################################################################
class EarlyStopABC(ABC):
    """
    Provides early stopping functionality during training.
    """

    @abstractmethod
    def evaluate(
            self,
            losses: BatchLoss,
            metrics: Optional[List[float]]
        ) -> bool:
        """
        Evaluate if training should stops.

        Parameters
        ----------
        losses : Losses
            Dataclass with residual, boundary, initial and observational loss.
        metrics : Mapping[str, float]
            Dictionary with metrics data.
        
        Returns
        -------
        bool
            Return True if training should stop.
        """
        ...
    
#####################################################################################