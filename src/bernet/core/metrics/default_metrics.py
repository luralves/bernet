#####################################################################################
from typing import Mapping

from bernet.contracts import IMetrics

#####################################################################################
class DFLTMetrics(IMetrics):
    
    def evaluate(self, model, data) -> Mapping[str, float]:
        super().evaluate(model, data)

        #-- Compute L1 norm
        l1_norm = 

        return

#####################################################################################