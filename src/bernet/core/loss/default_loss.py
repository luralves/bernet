#####################################################################################
import torch

from typing import Mapping

from bernet.contracts.loss import ILoss

#####################################################################################
#-- Auxiliary function
def _mse_default(
        model: torch.nn.Module,
        batch: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
    """
    Compute the Mean Squared Error for a default batch.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch Module.
    batch : Mapping[str, torch.Tensor]
        Data for loss computation containing "x" and "y_ref"
        as parameters.
        
    Returns
    -------
    torch.Tensor
        Tensor loss.
    """

    #-- Batch parameters
    x = batch["x"]
    y_ref = batch["y"]
        
    #-- Compute model output
    y_pred = model(x)

    #-- Compute loss from MSE
    loss = torch.mean((y_pred - y_ref) ** 2)

    return loss

#-- Loss implementation
class DFLTLoss(ILoss):
    """
    Helper class that already implements:
        1- compute_boundary_loss(...)
        2- compute_initial_loss(...)
        3- compute_observational_loss(...)
    using a mean squared error loss function.

    It is necessary to implement:
        1- compute_residual_loss(...)
    
    In order to implement the Batch, at least the ones
    related to boundary, initial and observational loss
    is composed on a Mapping with the following structure:
        batch = {
            "x": Tensor,
            "y": Tensor,
        }
    where, "x" are the inputs and "y" represents the
    expected output.
    """

    #-- Needs to be implemented
    def compute_residual_loss(
            self,
            model: torch.nn.Module,
            batch: Mapping[str, torch.Tensor]
        ) -> torch.Tensor:
        ...
    
    #-- Override
    def compute_boundary_loss(
            self,
            model: torch.nn.Module,
            batch: Mapping[str, torch.Tensor]
        ) -> torch.Tensor:
        return _mse_default(model=model, batch=batch)

    #-- Override
    def compute_initial_loss(
            self,
            model: torch.nn.Module,
            batch: Mapping[str, torch.Tensor]
        ) -> torch.Tensor:
        return _mse_default(model=model, batch=batch)
    
    #-- Override
    def compute_observational_loss(
            self,
            model: torch.nn.Module,
            batch: Mapping[str, torch.Tensor]
        ) -> torch.Tensor:
        return _mse_default(model=model, batch=batch)

    