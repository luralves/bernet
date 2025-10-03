#####################################################################################
import torch

from typing import Mapping, Callable

from bernet.contracts.loss import ILoss

from bernet.utils.analysis import Losses
from bernet.utils.validation import TypeCheck

#####################################################################################
class LossBASE(ILoss):

    def __init__(
            self,
            func_bc: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = Losses.mse,
            func_in: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = Losses.mse,
            func_ob: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = Losses.mse,
        ):
        super().__init__()

        #-- Verification
        TypeCheck.callable(func_bc)
        TypeCheck.callable(func_in)
        TypeCheck.callable(func_ob)

        #-- Inputs
        self._func_bc = func_bc
        self._func_in = func_in
        self._func_ob = func_ob

        return

    #-- Needs to be implemented
    def residual(
            self,
            model: torch.nn.Module,
            batch: Mapping[str, torch.Tensor]
        ) -> torch.Tensor:
        ...
    
    #-- Override
    def boundary(
            self,
            model: torch.nn.Module,
            batch: Mapping[str, torch.Tensor]
        ) -> torch.Tensor:
        y_hat = model(batch["x"])
        loss = self._func_bc(y_hat, batch["y"])
        return loss

    #-- Override
    def initial(
            self,
            model: torch.nn.Module,
            batch: Mapping[str, torch.Tensor]
        ) -> torch.Tensor:
        y_hat = model(batch["x"])
        loss = self._func_in(y_hat, batch["y"])
        return loss
    
    #-- Override
    def observational(
            self,
            model: torch.nn.Module,
            batch: Mapping[str, torch.Tensor]
        ) -> torch.Tensor:
        y_hat = model(batch["x"])
        loss = self._func_ob(y_hat, batch["y"])
        return loss

#####################################################################################