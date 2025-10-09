#####################################################################################
from abc import ABC, abstractmethod

from bernet.interface.typing.aliases import Model, TensorData

#####################################################################################
class IMetrics(ABC):
    """
    Compute metrics based on sampled data.
    """

    @abstractmethod
    def evaluate(self, model: Model, data: TensorData) -> TensorData:
        """
        Compute metrics.

        Parameters
        ----------
        model : torch.nn.Module
            PyTorch Module.
        data : TensorData
            Data for loss computation.
        
        Returns
        -------
        TensorData
            Metrics data.
        """
        ...
    
#####################################################################################