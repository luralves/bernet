#####################################################################################
from abc import ABC, abstractmethod

from bernet.interface.typing.aliases import TensorData
from bernet.interface.typing.dataclass import Losses
from bernet.interface.typing.structs import Signal

#####################################################################################
class IEarlyStop(ABC):
    """Provides early stopping functionality during training"""

    @abstractmethod
    def evaluate(self, losses: Losses, metrics: TensorData) -> Signal:
        """
        Evaluate if training should stops.

        Parameters
        ----------
        losses : Losses
            Dataclass with residual, boundary, initial and observational loss.
        metrics : TensorData
            Metrics data evaluated during training.
        
        Returns
        -------
        Signal
            Return the value of the signal (True or False) and a message.
        """
        ...
    
#####################################################################################