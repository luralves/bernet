#####################################################################################
import torch

from typing import Mapping, Optional, Tuple
from abc import ABC, abstractmethod

#####################################################################################
class BASELoss(ABC):
    """
    Abstract base class for PINN losses.

    Subclasses must implement `_compute`, which returns a scalar loss tensor and an
    optional dictionary of per-term values for logging.
    """

    @abstractmethod
    def _compute(
        self,
        model: torch.nn.Module,
        batch: Mapping[str, torch.Tensor],
        *,
        epoch: int,
        global_step: int,
    ) -> Tuple[torch.Tensor, Optional[Mapping[str, torch.Tensor]]]:
        """
        Internal method to compute the loss.

        Parameters
        ----------
        - model: nn.Module
          > Neural network model.
        - batch: Mapping[str, Tensor]
          > Input batch used to compute the loss.
        - epoch: int
          > Current training epoch provided by the Trainer.
        - global_step: int
          > Current training step (number of batches processed).

        Returns
        -------
        - loss: Tensor
          > Scalar loss value.
        - terms: Optional[Mapping[str, Tensor]]
          > Optional dictionary of per-term losses for logging.
        """
        ...

    def __call__(
        self,
        model: torch.nn.Module,
        batch: Mapping[str, torch.Tensor],
        *,
        epoch: int,
        global_step: Optional[int],
    ) -> Tuple[torch.Tensor, Optional[Mapping[str, torch.Tensor]]]:
        """
        Public API to compute the loss. Wraps `_compute` and verifies that
        the returned loss is a scalar tensor on the same device as the model.
        """

        #--- Loss computation
        res = self._compute(model, batch, epoch=epoch, global_step=global_step)

        #--- Verification
        if len(res) != 2:
            raise TypeError("[BASELoss] _compute must return tuble of len = 2.")
        
        if not torch.is_tensor(res[0]):
            raise TypeError("[BASELoss] Loss must be a torch.Tensor.")
        
        if res[0].ndim != 0:
            raise TypeError(f"[BASELoss] Loss must be scalar (0-dim), got shape {res[0].shape}.")

        if res[0].device != next(model.parameters()).device:
            raise TypeError("[BASELoss] Loss must be on the same device as the model")

        if not torch.isfinite(res[0]):
            raise TypeError("[BASELoss] Loss is not finite (NaN/Inf).")

        return res

#####################################################################################
