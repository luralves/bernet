#####################################################################################
import torch

from typing import Mapping, Tuple
from abc import ABC

from bernet.utils import Validation

#####################################################################################
class LossBASE(ABC):
    """
    Compute the loss based on sampler data.

    There is a loss term for each type of data:
    - Residual points: _residual(model, batch) -> Tensor
    - Boundary points: _boundary(model, batch) -> Tensor
    - Initial points: _initial(model, batch) -> Tensor
    - Data points: _data(model, batch) -> Tensor

    The user must implement only the necessary internal methods.
    """

    def _residual(
            self,
            model: torch.Tensor,
            batch: Mapping[str, torch.Tensor]
            ) -> torch.Tensor:
        """
        Compute the residual loss.

        Parameters:
        ----------
        - model: torch.Tensor
          > The model to evaluate.
        - batch: Mapping[str, torch.Tensor]
          > The data batch.
        """
        ...
    
    def _boundary(
            self,
            model: torch.Tensor,
            batch: Mapping[str, torch.Tensor]
            ) -> torch.Tensor:
        """
        Compute the boundary condition loss.

        Parameters:
        ----------
        - model: torch.Tensor
          > The model to evaluate.
        - batch: Mapping[str, torch.Tensor]
          > The data batch.
        """
        ...
    
    def _initial(
            self,
            model: torch.Tensor,
            batch: Mapping[str, torch.Tensor]
            ) -> torch.Tensor:
        """
        Compute the initial condition loss.

        Parameters:
        ----------
        - model: torch.Tensor
          > The model to evaluate.
        - batch: Mapping[str, torch.Tensor]
          > The data batch.
        """
        ...
    
    def _data(
            self,
            model: torch.Tensor,
            batch: Mapping[str, torch.Tensor]
            ) -> torch.Tensor:
        """
        Compute the data loss.

        Parameters:
        ----------
        - model: torch.Tensor
          > The model to evaluate.
        - batch: Mapping[str, torch.Tensor]
          > The data batch.
        """
        ...
    
    def __call__(
            self,
            model: torch.nn.Module,
            batch: Mapping[str, torch.Tensor]
            ) -> Tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        """
        Compute the loss.

        Parameters
        ----------
        - model: nn.Module
          > Neural network model.
        - batch: Mapping[str, Mapping[str, Tensor]]
          > Batch for each term in the loss function.

        Returns
        -------
        - loss: Tensor
          > Scalar loss value.
        - terms: Optional[Mapping[str, Tensor]]
          > Optional dictionary of per-term losses for logging.
        """
        
        #-- Compute loss components
        loss_rs = self._residual(model=model, batch=batch["residual"])
        loss_bc = self._boundary(model=model, batch=batch["boundary"])
        loss_ic = self._initial(model=model, batch=batch.get("initial", None))
        loss_dt = self._data(model=model, batch=batch["data"])

        #-- Validate loss components
        device = next(model.parameters()).device

        loss_rs = Validation.loss(x=loss_rs, device=device)
        loss_bc = Validation.loss(x=loss_bc, device=device)
        loss_ic = Validation.loss(x=loss_ic, device=device)
        loss_dt = Validation.loss(x=loss_dt, device=device)

        #-- Compute total loss
        loss = loss_rs + loss_bc + loss_ic + loss_dt

        #-- Loss componentes
        terms = {"residual": loss_rs, "boundary": loss_bc, "initial": loss_ic, "data": loss_dt,}

        return loss, terms