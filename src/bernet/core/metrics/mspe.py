#####################################################################################
import torch

from typing import Mapping

from bernet.contracts import IMetrics
from bernet.utils.analysis import Losses

#####################################################################################
class MSPE(IMetrics):
    """Mean Squared Percentage Error"""
    
    #-- Override
    def evaluate(self, model: torch.Tensor, data: Mapping[str, float]) -> Mapping[str, torch.Tensor]:
        super().evaluate(model, data)
        y_hat = model(data["x"])
        mse = Losses.mspe(y_hat, data["y"])
        return {"mse": mse}

#####################################################################################