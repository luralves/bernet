#####################################################################################
import torch

from abc import ABC, abstractmethod
from typing import Mapping

#####################################################################################
class SamplerBASE(ABC):
    """
    Provides batches to the Trainer.
    
    > Necessary parameters:
    - sampler.num_batches: int
    
    > Necessary methods:
    - sampler.batch(epoch, step) -> Mapping[str, torch.Tensor]

    > Optional methods:
    - sampler.metrics() -> Mapping[str, torch.Tensor]
    """
    num_batches: int = None

    @abstractmethod
    def batch(self) -> Mapping[str, torch.Tensor]:
        """
        Return a single training batch as a flat mapping of tensors.
        """
        ...
    
    def metrics(self) -> Mapping[str, torch.Tensor]:
        """
        Return evaluation data for metrics computation.
        """
        ...
    
#####################################################################################