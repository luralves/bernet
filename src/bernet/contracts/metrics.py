#####################################################################################
import torch

from abc import ABC, abstractmethod
from typing import Mapping

#####################################################################################
class MetricsABC(ABC):
    """
    Compute metrics based on sampled data.
    """

    @abstractmethod
    def evaluate(
            self,
            model: torch.nn.Module,
            data: Mapping[str, torch.Tensor],
        ) -> Mapping[str, torch.Tensor]:
        """
        Compute metrics.

        Parameters
        ----------
        model : torch.nn.Module
            PyTorch Module.
        batch : Mapping[str, torch.Tensor]
            Data for loss computation containing "x" and "y_ref"
            as parameters.
        
        Returns
        -------
        Mapping[str, float]
            Dictionary structured data.
        """
        ...
    
#####################################################################################