#####################################################################################
import torch

from abc import ABC, abstractmethod
from typing import Mapping
from dataclasses import dataclass

#####################################################################################
#-- Auxiliary class
@dataclass
class Batch():
    """
    Batch data for loss computation.
    """
    residual: Mapping[str, torch.Tensor]
    boundary: Mapping[str, torch.Tensor] | None = None
    initial: Mapping[str, torch.Tensor] | None = None
    observational: Mapping[str, torch.Tensor] | None = None

    def to_device(
            self,
            device: torch.device,
        ) -> None:
        """
        Move tensors to device.

        Parameters
        ----------
        device : torch.device
            Model device
        """

        #-- Required dataset
        self.residual = {k: v.to(device) for k, v in self.residual.items()}
        
        #-- Optional datasets
        if self.boundary is not None:
            self.boundary = {k: v.to(device) for k, v in self.boundary.items()}

        if self.initial is not None:
            self.initial = {k: v.to(device) for k, v in self.initial.items()}
        
        if self.observational is not None:
            self.observational = {k: v.to(device) for k, v in self.observational.items()}

        return

#-- Main class
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
    def batch(
            self,
            index: int,
        ) -> Batch:
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
    def metrics(self) -> Mapping[str, torch.Tensor]:
        """
        Returns the data necessary for metrics computation.

        Returns:
        Mapping[str, torch.Tensor]
            Dictionary structured data.
        """
        ...
    
#####################################################################################