#####################################################################################
# tests/test_base_loss.py
import pytest
import torch

from typing import Mapping, Optional, Tuple
from bernet.abstract import BASELoss

#####################################################################################
class TinyModel(torch.nn.Module):
    """Dummy model & batches"""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3, 1)  # ensures the model has parameters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)

def make_batch(device: torch.device = torch.device("cpu")) -> Mapping[str, torch.Tensor]:
    """Define a batch"""
    return {"x": torch.randn(8, 3, device=device)}

#####################################################################################
class GoodLoss(BASELoss):
    """Returns a scalar tensor on the same device as the model"""
    def _compute(
        self,
        model: torch.nn.Module,
        batch: Mapping[str, torch.Tensor],
        *,
        epoch: int,
        global_step: int,
    ) -> Tuple[torch.Tensor, Optional[Mapping[str, torch.Tensor]]]:
        dev = next(model.parameters()).device
        loss = torch.tensor(0.123, device=dev)
        terms = {"dummy": loss.clone()}
        return loss, terms


class WrongTupleLenLoss(BASELoss):
    """Returns a tuple of length 1 -> should raise a TypeError in the wrapper"""
    def _compute(
        self,
        model: torch.nn.Module,
        batch: Mapping[str, torch.Tensor],
        *,
        epoch: int,
        global_step: int,
    ):
        dev = next(model.parameters()).device
        loss = torch.tensor(0.5, device=dev)
        return (loss,)  # wrong length


class NonTensorLoss(BASELoss):
    """Returns a non-tensor loss -> should raise a TypeError"""
    def _compute(
        self,
        model: torch.nn.Module,
        batch: Mapping[str, torch.Tensor],
        *,
        epoch: int,
        global_step: int,
    ):
        return 1.0, None  # not a torch.Tensor


class NonScalarLoss(BASELoss):
    """Returns a tensor with dim != 0 -> should raise a TypeError"""
    def _compute(
        self,
        model: torch.nn.Module,
        batch: Mapping[str, torch.Tensor],
        *,
        epoch: int,
        global_step: int,
    ):
        dev = next(model.parameters()).device
        loss = torch.ones(1, device=dev)  # shape (1,) not scalar
        return loss, None


class NonFiniteLoss(BASELoss):
    """Returns NaN -> should raise a TypeError"""
    def _compute(
        self,
        model: torch.nn.Module,
        batch: Mapping[str, torch.Tensor],
        *,
        epoch: int,
        global_step: int,
    ):
        dev = next(model.parameters()).device
        loss = torch.tensor(float("nan"), device=dev)
        return loss, None

#####################################################################################
def test_good_loss_cpu():
    model = TinyModel().cpu()
    batch = make_batch(torch.device("cpu"))
    loss_fn = GoodLoss()
    _, _ = loss_fn(model, batch, epoch=0, global_step=0)

def test_wrong_tuple_length():
    model = TinyModel()
    batch = make_batch()
    loss_fn = WrongTupleLenLoss()
    with pytest.raises(TypeError, match=r"_compute must return"):
        _ = loss_fn(model, batch, epoch=0, global_step=0)

def test_non_tensor_loss():
    model = TinyModel()
    batch = make_batch()
    loss_fn = NonTensorLoss()
    with pytest.raises(TypeError, match=r"Loss must be a torch\.Tensor"):
        _ = loss_fn(model, batch, epoch=0, global_step=0)

def test_non_scalar_loss():
    model = TinyModel()
    batch = make_batch()
    loss_fn = NonScalarLoss()
    with pytest.raises(TypeError, match=r"Loss must be scalar"):
        _ = loss_fn(model, batch, epoch=0, global_step=0)

def test_non_finite_loss():
    model = TinyModel()
    batch = make_batch()
    loss_fn = NonFiniteLoss()
    with pytest.raises(TypeError, match=r"Loss is not finite"):
        _ = loss_fn(model, batch, epoch=0, global_step=0)
