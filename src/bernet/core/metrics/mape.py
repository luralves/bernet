#####################################################################################
import torch

from typing import Mapping

from bernet.contracts import IMetrics
from bernet.utils.processing import Losses

#####################################################################################
class MAPE(IMetrics):
    """Mean Absolute Percentage Error"""

    #-- Override
    def evaluate(self, model: torch.Tensor, data: Mapping[str, float]) -> Mapping[str, torch.Tensor]:
        super().evaluate(model, data)
        y_hat = model(data["x"])
        mse = Losses.mape(y_hat, data["y"])
        return {"mape": mse}

#####################################################################################