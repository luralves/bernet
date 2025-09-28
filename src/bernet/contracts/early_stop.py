#####################################################################################
from typing import Mapping, Optional
from abc import ABC, abstractmethod

from bernet.contracts.loss import Losses

#####################################################################################
class IEarlyStop(ABC):
    """
    Provides early stopping functionality during training.
    """

    @abstractmethod
    def evaluate(
            self,
            losses: Losses,
            metrics: Optional[Mapping[str, float]]
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