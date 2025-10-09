#####################################################################################
import torch

from typing import Union, Iterable, Mapping

#####################################################################################
Tensor = torch.Tensor
Model = torch.nn.Module
Optimizer = torch.optim.Optimizer
TensorData = Union[Tensor, Iterable[Tensor], Mapping[str, Tensor]]

#####################################################################################