#####################################################################################
from __future__ import annotations
from typing import NamedTuple, Optional

from bernet.interface.typing.aliases import TensorData

#####################################################################################
class Batch(NamedTuple):
    residual: Optional[TensorData]
    boundary: Optional[TensorData]
    initial: Optional[TensorData]
    observational: Optional[TensorData]

class Signal(NamedTuple):
    value: bool
    message: str

#####################################################################################