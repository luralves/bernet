#####################################################################################
import torch
import numpy.typing as npt

from typing import Union, Iterable, Mapping

#####################################################################################
Tensor = torch.Tensor
Model = torch.nn.Module             # full model definition
Layer = torch.nn.Module             # layer in a model
Optimizer = torch.optim.Optimizer

NDArray = npt.NDArray

TensorData = Union[Tensor, Iterable[Tensor], Mapping[str, Tensor]]

#####################################################################################