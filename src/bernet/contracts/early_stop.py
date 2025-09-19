#####################################################################################
from typing import Mapping
from abc import ABC, abstractmethod

#####################################################################################
class EarlyStopBASE(ABC):
    """
    Provides early stopping functionality for the Trainer.

    > Necessary methods:
    - __call__() -> bool
    """

    @abstractmethod
    def __call__(
        self,
        data: Mapping[str, float],
        ) -> bool:
        """
        Check if early stopping criteria are met.
        """
        ...
    
#####################################################################################