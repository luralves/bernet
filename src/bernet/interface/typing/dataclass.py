#####################################################################################
from __future__ import annotations

from dataclasses import dataclass

from bernet.interface.typing.aliases import Tensor
from bernet.utils.validation import TypeCheck

#####################################################################################
@dataclass
class Losses:
    residual: Tensor
    boundary: Tensor
    initial: Tensor
    observational: Tensor

    def sum(self) -> Tensor:
        """Sum all losses"""
        return self.residual + self.boundary + self.initial + self.observational
    
    #-- Override
    def __add__(self, other: Losses) -> Losses:
        """Losses can only be added to other losses"""
        TypeCheck.abc(other, Losses)
        return Losses(
            residual=self.residual + other.residual,
            boundary=self.boundary + other.boundary,
            initial=self.initial + other.initial,
            observational=self.observational + other.observational,
        )
    
    #-- Override
    def __truediv__(self, value: float | int) -> Losses:
        """Losses can only be divided by a float value"""
        TypeCheck.number(value)
        return Losses(
            residual=self.residual / value,
            boundary=self.boundary / value,
            initial=self.initial / value,
            observational=self.observational / value,
        )

#####################################################################################