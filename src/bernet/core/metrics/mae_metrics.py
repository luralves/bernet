#####################################################################################
import torch

from typing import Mapping

from bernet.contracts import MetricsABC
from bernet.utils.analysis import Losses

#####################################################################################
class MAEMetrics(MetricsABC):
    
    #-- Override
    def evaluate(self, model: torch.Tensor, data: Mapping[str, float]) -> Mapping[str, torch.Tensor]:
        super().evaluate(model, data)
        y_hat = model(data["x"])
        mae = Losses.mae(y_hat, data["y"])
        return {"mae": mae}

#####################################################################################