#####################################################################################
from abc import ABC, abstractmethod
from typing import Optional

from bernet.interface.typing.aliases import TensorData
from bernet.interface.typing.structs import Batch

#####################################################################################
class ISampler(ABC):
    """
    Provides data for training and validating.
    """

    @abstractmethod
    def generate(self) -> int:
        """
        Generate the batches for current epoch and returns
        its length.

        Returns
        -------
        int
            Batches length.
        """
        ...
    
    @abstractmethod
    def batch(self, index: int) -> Batch:
        """
        Receives the batch index and returns the batch with
        trianing parameters.

        Parameters
        ----------
        index : int
            Current batch index.
        
        Returns
        -------
        Batch
            Data for loss computation.
        """
        ...
    
    @abstractmethod
    def test(self) -> Optional[TensorData]:
        """
        Returns the data necessary for testing.

        Returns:
        TensorData
            Dictionary structured data.
        """
        ...
    
    @abstractmethod
    def validate(self) -> Optional[TensorData]:
        """
        Returns the data necessary for validation.

        Returns:
        TensorData
            Dictionary structured data.
        """
        ...
    
#####################################################################################