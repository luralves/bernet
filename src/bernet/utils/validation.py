#####################################################################################
import torch

from typing import Any

#####################################################################################
class Validation():
    """
    Implement validation utilities.
    """
    
    @staticmethod
    def loss(
        x: Any,
        device: torch.device,
        ) -> torch.Tensor:
        """
        Validade the loss.

        Parameters:
        ----------
        - x: Any
          > The loss to validate.
        - device: torch.device
          > The device where the loss should be.

        Returns:
        -------
        - torch.Tensor
          > The validated loss (0.0 if x is None).
        """

        #-- Check if x is None
        if x is None:
            return torch.tensor(0.0, device=device)

        #-- Check if x is a tensor
        if not torch.is_tensor(x):
            raise TypeError("[Loss validation] Loss must be a torch.Tensor.")
        
        #-- Check if x is a scalar
        if x.ndim != 0:
            raise TypeError(f"[Loss validation] Loss must be scalar (0-dim), got shape {x.shape}.")

        #-- Check if x is on the correct device
        if x.device != device:
            raise TypeError(f"[Loss validation] Loss must be on device {device}, got {x.device}.")

        #-- Check if x contain nan or inf numbers
        if not torch.isfinite(x):
            raise TypeError("[Loss validation] Loss is not finite (NaN/Inf).")

        return x

#####################################################################################