#####################################################################################
import torch

from abc import ABC, abstractmethod
from typing import Mapping

#####################################################################################
class MetricsBASE(ABC):
    """
    Compute metrics based on sampler data.

    > Necessary methods:
    - metrics(model, data) -> Mapping[str, torch.Tensor]
    """

    @abstractmethod
    def __call__(
        self,
        model: torch.nn.Module,
        data: Mapping[str, torch.Tensor],
        ) -> Mapping[str, float]:
        """
        Return the metrics computed based on sampler data.

        Parameters:
        ----------
        - model: torch.nn.Module
          > The model to evaluate.
        - data: Mapping[str, torch.Tensor]
          > The data to evaluate on.

        Returns:
        -------
        - Mapping[str, torch.Tensor]
          > The computed metrics.
        """
        ...
    
#####################################################################################