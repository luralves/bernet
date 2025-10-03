#####################################################################################
from __future__ import annotations

import torch

from typing import Mapping, Optional, List
from abc import ABC, abstractmethod
from dataclasses import dataclass

from bernet.contracts.sampler import BatchSample
from bernet.utils.validation import TypeCheck

#####################################################################################
#-- Auxiliary class
@dataclass
class BatchLoss():
    residual: torch.Tensor | float
    boundary: Optional[torch.Tensor | float]
    initial: Optional[torch.Tensor | float]
    observational: Optional[torch.Tensor | float]

    def sum(self) -> torch.Tensor | float:
        """
        Compute total loss.

        Returns
        -------
        torch.Tensor | float
            Sum of all loss terms.
        """

        #-- Initiate with the only required parameter
        if isinstance(self.residual, torch.Tensor):
            loss = self.residual.clone()
        else:
            loss = float(self.residual)
        
        #-- Add optional parameters
        if self.boundary is not None:
            loss = loss + self.boundary
        
        if self.initial is not None:
            loss += self.initial
        
        if self.observational is not None:
            loss += self.observational

        return loss
    
    def to_float(self) -> BatchLoss:
        """
        Convert Tensor to float.

        Returns
        -------
        Losses
            Terms with data in float format.
        """
        return BatchLoss(
            residual=self.residual.item(),
            boundary=self.boundary.item() if self.boundary is not None else 0.0,
            initial=self.initial.item() if self.initial is not None else 0.0,
            observational=self.observational.item() if self.observational is not None else 0.0,
        )

    def to_list(self) -> List[torch.Tensor | float]:
        """
        Convert the terms to a list.

        Returns
        -------
        List[float]
            List with loss terms.
        """

        #-- Initiate with the only required parameter
        out = [self.residual]

        #-- Add optional parameters
        if self.boundary is not None:
            out.append(self.boundary)
        else:
            out.append(.0)
        
        if self.initial is not None:
            out.append(self.initial)
        else:
            out.append(.0)
        
        if self.observational is not None:
            out.append(self.observational)
        else:
            out.append(.0)

        return out
    
    #-- Override
    def __add__(
            self,
            other: BatchLoss,
        ) -> BatchLoss:
        """
        Override the __add__ method.
        """
        return BatchLoss(
            residual=self.residual + other.residual,
            boundary=self.boundary + other.boundary if (self.boundary is not None) and (other.boundary is not None) else 0.0,
            initial=self.initial + other.initial if (self.initial is not None) and (other.initial is not None) else 0.0,
            observational=self.observational + other.observational if (self.observational is not None) and (other.observational is not None) else 0.0,
        )
    
    #-- Override
    def __truediv__(self, value: float) -> BatchLoss:
        """
        Override the __truediv__ method.
        Terms can only be divided by a float value.
        """
        TypeCheck.number(value)
        return BatchLoss(
            residual=self.residual / value,
            boundary=self.boundary / value if self.boundary is not None else 0.0,
            initial=self.initial / value if self.initial is not None else 0.0,
            observational=self.observational / value if self.observational is not None else 0.0,
        )

#-- Main class
class ILoss(ABC):
    """
    Compute loss terms based on sampled data.
    """

    @abstractmethod
    def residual(
            self,
            model: torch.nn.Module,
            batch: Mapping[str, torch.Tensor]
        ) -> torch.Tensor:
        """
        Compute residual loss.

        Parameters
        ----------
        model : torch.nn.Module
            PyTorch Module.
        batch : Mapping[str, torch.Tensor]
            Data for loss computation.
        
        Returns
        -------
        torch.Tensor
            Tensor loss.
        """
        ...
    
    @abstractmethod
    def boundary(
            self,
            model: torch.nn.Module,
            batch: Mapping[str, torch.Tensor]
        ) -> torch.Tensor:
        """
        Compute residual loss.

        Parameters
        ----------
        model : torch.nn.Module
            PyTorch Module.
        batch : Mapping[str, torch.Tensor]
            Data for loss computation.
        
        Returns
        -------
        torch.Tensor
            Tensor loss.
        """
        ...

    @abstractmethod
    def initial(
            self,
            model: torch.nn.Module,
            batch: Mapping[str, torch.Tensor]
        ) -> torch.Tensor:
        """
        Compute initial loss.

        Parameters
        ----------
        model : torch.nn.Module
            PyTorch Module.
        batch : Mapping[str, torch.Tensor]
            Data for loss computation.
        
        Returns
        -------
        torch.Tensor
            Tensor loss.
        """
        ...
    
    @abstractmethod
    def observational(
            self,
            model: torch.nn.Module,
            batch: Mapping[str, torch.Tensor]
        ) -> torch.Tensor:
        """
        Compute initial loss.

        Parameters
        ----------
        model : torch.nn.Module
            PyTorch Module.
        batch : Mapping[str, torch.Tensor]
            Data for loss computation.
        
        Returns
        -------
        torch.Tensor
            Tensor loss.
        """
        ...


    def compute(
            self,
            model: torch.nn.Module,
            batch: BatchSample,
        ) -> BatchLoss:
        """
        Return loss terms
        """

        #-- Required loss    
        loss_r = self.residual(model=model, batch=batch.residual)

        #-- Optional losses
        if batch.boundary is not None:
            loss_b = self.boundary(model=model, batch=batch.boundary)
        else:
            loss_b = None
        
        if batch.initial is not None:
            loss_i = self.initial(model=model, batch=batch.initial)
        else:
            loss_i = None
        
        if batch.observational is not None:
            loss_o = self.observational(model=model, batch=batch.observational)
        else:
            loss_o = None

        #-- Create output
        losses = BatchLoss(
            residual=loss_r,
            boundary=loss_b,
            initial=loss_i,
            observational=loss_o,
        )

        return losses

