"""
This file contains two different types of implementations:
    1) A class that implements some error functions, such as MSE,
    as static methods.
    2) Decorators that can be used directly by the methods residual,
    boundary, initila and observational in the Loss implementation.
"""
#####################################################################################
import torch
import functools

from typing import Union, Iterable, Callable

from bernet.interface.typing import Tensor
from bernet.utils.validation.type_check import TypeCheck

#####################################################################################
class Losses:
    """
    Losses for continuous outputs.
    """

    @staticmethod
    def mse(y_hat: Tensor, y_ref: Tensor) -> Tensor:
        """
        Mean Squared Error.
        
        Parameters
        ----------
        y_hat : Tensor
            Predicted value.
        y_ref : Tensor
            Reference value.
        
        Returns
        -------
        Tensor
            Loss value
        """
        return torch.mean((y_hat - y_ref) ** 2)

    @staticmethod
    def mae(y_hat: Tensor, y_ref: Tensor) -> Tensor:
        """
        Mean Absolute Error.
        
        Parameters
        ----------
        y_hat : Tensor
            Predicted value.
        y_ref : Tensor
            Reference value.
        
        Returns
        -------
        Tensor
            Loss value
        """
        return torch.mean((y_hat - y_ref).abs())
    
    @staticmethod
    def log_cosh(y_hat: Tensor, y_ref: Tensor) -> Tensor:
        """
        Log(cosh(x)) error.
        
        Parameters
        ----------
        y_hat : Tensor
            Predicted value.
        y_ref : Tensor
            Reference value.
        
        Returns
        -------
        Tensor
            Loss value
        """
        return torch.mean(torch.log(torch.cosh(y_hat - y_ref)))
    
    @staticmethod
    def mape(y_hat: Tensor, y_ref: Tensor, *, eps: float = 1e-6) -> Tensor:
        """
        Mean Absolute Percentage Error.
        
        Parameters
        ----------
        y_hat : Tensor
            Predicted value.
        y_ref : Tensor
            Reference value.
        
        Returns
        -------
        Tensor
            Loss value
        """
        return torch.mean((y_hat - y_ref).abs() / y_ref.abs().clamp_min(min=eps))

    @staticmethod
    def smape(y_hat: Tensor, y_ref: Tensor, *, eps: float = 1e-6) -> Tensor:
        """
        Simmetric Mean Absolute Percentage Error.
        
        Parameters
        ----------
        y_hat : Tensor
            Predicted value.
        y_ref : Tensor
            Reference value.
        
        Returns
        -------
        Tensor
            Loss value
        """
        return 2.0 * torch.mean((y_hat - y_ref).abs() / (y_hat.abs() + y_ref.abs()).clamp_min(min=eps))

    @staticmethod
    def mspe(y_hat: Tensor, y_ref: Tensor, *, eps: float = 1e-6) -> Tensor:
        """
        Mean Squared Percentage Error.
        
        Parameters
        ----------
        y_hat : Tensor
            Predicted value.
        y_ref : Tensor
            Reference value.
        
        Returns
        -------
        Tensor
            Loss value
        """
        return torch.mean((y_hat - y_ref) ** 2 / (y_ref ** 2).clamp_min(min=eps))

    @staticmethod
    def smspe(y_hat: Tensor, y_ref: Tensor, *, eps: float = 1e-6) -> Tensor:
        """
        Simmetric Mean Squared Percentage Error.
        
        Parameters
        ----------
        y_hat : Tensor
            Predicted value.
        y_ref : Tensor
            Reference value.
        
        Returns
        -------
        Tensor
            Loss value
        """
        return 4.0 * torch.mean((y_hat - y_ref) ** 2 / ((y_hat ** 2) + (y_ref ** 2)).clamp_min(min=eps))

#####################################################################################
#-- Function output
Output = Union[Tensor, Iterable[Tensor], Iterable[Iterable[Tensor]]]

#-- Mean Squared Error
def mse() -> Callable[[Callable[..., Output]], Callable[..., Tensor]]:
    """
    Decorator factory that turns a function returning Output into an MSE loss.

    Notes
    -----
    Supports:
      - Tensor -> uses zeros_like as target
      - Iterable[Tensor] -> sum of mspe(t, 0) for each
      - Iterable[Iterable[Tensor]] -> assumes pairs (pred, target)
    """
    def decorator(fn: Callable[..., Output]) -> Callable[..., Tensor]:
        
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Tensor:
            out = fn(*args, **kwargs)

            # Case 1: single tensor
            if TypeCheck.generic(out, [Tensor], stop=False):
                return Losses.mse(out, torch.zeros_like(out))

            # We may need to iterate more than once, so snapshot
            if isinstance(out, Iterable):
                items = list(out)

                # Case 2: iterable of tensors
                if items and all(TypeCheck.generic(t, [Tensor], stop=False) for t in items):
                    return sum(Losses.mse(t, torch.zeros_like(t)) for t in items)

                # Case 3: iterable of iterable-of-tensors (pairs)
                def is_pair(elem) -> bool:
                    if not TypeCheck.iterable(elem, stop=False):
                        return False
                    pair = list(elem)
                    if len(pair) != 2:
                        return False
                    a, b = pair
                    return TypeCheck.generic(a, [Tensor], stop=False) and TypeCheck.generic(b, [Tensor], stop=False)

                if items and all(is_pair(p) for p in items):
                    return sum(Losses.mse(p[0], p[1]) for p in map(list, items))

            # If nothing matched, raise a helpful error
            raise TypeError(
                "mse-decorated function must return one of: "
                "Tensor; Iterable[Tensor]; Iterable[Iterable[Tensor]] where inner iterables are (pred, target) pairs."
            )
        return wrapper
    
    return decorator

#-- Mean Absolute Error
def mae() -> Callable[[Callable[..., Output]], Callable[..., Tensor]]:
    """
    Decorator factory that turns a function returning Output into an MAE loss.

    Notes
    -----
    Supports:
      - Tensor -> uses zeros_like as target
      - Iterable[Tensor] -> sum of mspe(t, 0) for each
      - Iterable[Iterable[Tensor]] -> assumes pairs (pred, target)
    """
    def decorator(fn: Callable[..., Output]) -> Callable[..., Tensor]:
        
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Tensor:
            out = fn(*args, **kwargs)

            # Case 1: single tensor
            if TypeCheck.generic(out, [Tensor], stop=False):
                return Losses.mae(out, torch.zeros_like(out))

            # We may need to iterate more than once, so snapshot
            if isinstance(out, Iterable):
                items = list(out)

                # Case 2: iterable of tensors
                if items and all(TypeCheck.generic(t, [Tensor], stop=False) for t in items):
                    return sum(Losses.mae(t, torch.zeros_like(t)) for t in items)

                # Case 3: iterable of iterable-of-tensors (pairs)
                def is_pair(elem) -> bool:
                    if not TypeCheck.iterable(elem, stop=False):
                        return False
                    pair = list(elem)
                    if len(pair) != 2:
                        return False
                    a, b = pair
                    return TypeCheck.generic(a, [Tensor], stop=False) and TypeCheck.generic(b, [Tensor], stop=False)

                if items and all(is_pair(p) for p in items):
                    return sum(Losses.mae(p[0], p[1]) for p in map(list, items))

            # If nothing matched, raise a helpful error
            raise TypeError(
                "mae-decorated function must return one of: "
                "Tensor; Iterable[Tensor]; Iterable[Iterable[Tensor]] where inner iterables are (pred, target) pairs."
            )
        return wrapper
    
    return decorator

#-- Log cosh Error
def log_cosh() -> Callable[[Callable[..., Output]], Callable[..., Tensor]]:
    """
    Decorator factory that turns a function returning Output into an LOG_COSH loss.

    Notes
    -----
    Supports:
      - Tensor -> uses zeros_like as target
      - Iterable[Tensor] -> sum of mspe(t, 0) for each
      - Iterable[Iterable[Tensor]] -> assumes pairs (pred, target)
    """
    def decorator(fn: Callable[..., Output]) -> Callable[..., Tensor]:
        
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Tensor:
            out = fn(*args, **kwargs)

            # Case 1: single tensor
            if TypeCheck.generic(out, [Tensor], stop=False):
                return Losses.log_cosh(out, torch.zeros_like(out))

            # We may need to iterate more than once, so snapshot
            if isinstance(out, Iterable):
                items = list(out)

                # Case 2: iterable of tensors
                if items and all(TypeCheck.generic(t, [Tensor], stop=False) for t in items):
                    return sum(Losses.log_cosh(t, torch.zeros_like(t)) for t in items)

                # Case 3: iterable of iterable-of-tensors (pairs)
                def is_pair(elem) -> bool:
                    if not TypeCheck.iterable(elem, stop=False):
                        return False
                    pair = list(elem)
                    if len(pair) != 2:
                        return False
                    a, b = pair
                    return TypeCheck.generic(a, [Tensor], stop=False) and TypeCheck.generic(b, [Tensor], stop=False)

                if items and all(is_pair(p) for p in items):
                    return sum(Losses.log_cosh(p[0], p[1]) for p in map(list, items))

            # If nothing matched, raise a helpful error
            raise TypeError(
                "log_cosh-decorated function must return one of: "
                "Tensor; Iterable[Tensor]; Iterable[Iterable[Tensor]] where inner iterables are (pred, target) pairs."
            )
        return wrapper
    
    return decorator

#-- Mean Absolute Percentage Error
def mape(*, eps: float = 1e-6) -> Callable[[Callable[..., Output]], Callable[..., Tensor]]:
    """
    Decorator factory that turns a function returning Output into an MAPE loss.

    Parameters
    ----------
    eps : float
        Zero approximation. Avoids division by zero.

    Notes
    -----
    Supports:
      - Tensor -> uses zeros_like as target
      - Iterable[Tensor] -> sum of mspe(t, 0) for each
      - Iterable[Iterable[Tensor]] -> assumes pairs (pred, target)
    """
    def decorator(fn: Callable[..., Output]) -> Callable[..., Tensor]:
        
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Tensor:
            out = fn(*args, **kwargs)

            # Case 1: single tensor
            if TypeCheck.generic(out, [Tensor], stop=False):
                return Losses.mape(out, torch.zeros_like(out), eps=eps)

            # We may need to iterate more than once, so snapshot
            if isinstance(out, Iterable):
                items = list(out)

                # Case 2: iterable of tensors
                if items and all(TypeCheck.generic(t, [Tensor], stop=False) for t in items):
                    return sum(Losses.mape(t, torch.zeros_like(t), eps=eps) for t in items)

                # Case 3: iterable of iterable-of-tensors (pairs)
                def is_pair(elem) -> bool:
                    if not TypeCheck.iterable(elem, stop=False):
                        return False
                    pair = list(elem)
                    if len(pair) != 2:
                        return False
                    a, b = pair
                    return TypeCheck.generic(a, [Tensor], stop=False) and TypeCheck.generic(b, [Tensor], stop=False)

                if items and all(is_pair(p) for p in items):
                    return sum(Losses.mape(p[0], p[1], eps=eps) for p in map(list, items))

            # If nothing matched, raise a helpful error
            raise TypeError(
                "mape-decorated function must return one of: "
                "Tensor; Iterable[Tensor]; Iterable[Iterable[Tensor]] where inner iterables are (pred, target) pairs."
            )
        return wrapper
    
    return decorator

#-- Symmetric Mean Absolute PErcentage Error
def smape(*, eps: float = 1e-6) -> Callable[[Callable[..., Output]], Callable[..., Tensor]]:
    """
    Decorator factory that turns a function returning Output into an SMAPE loss.

    Parameters
    ----------
    eps : float
        Zero approximation. Avoids division by zero.

    Notes
    -----

    Supports:
      - Tensor -> uses zeros_like as target
      - Iterable[Tensor] -> sum of mspe(t, 0) for each
      - Iterable[Iterable[Tensor]] -> assumes pairs (pred, target)
    """
    def decorator(fn: Callable[..., Output]) -> Callable[..., Tensor]:
        
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Tensor:
            out = fn(*args, **kwargs)

            # Case 1: single tensor
            if TypeCheck.generic(out, [Tensor], stop=False):
                return Losses.smape(out, torch.zeros_like(out), eps=eps)

            # We may need to iterate more than once, so snapshot
            if isinstance(out, Iterable):
                items = list(out)

                # Case 2: iterable of tensors
                if items and all(TypeCheck.generic(t, [Tensor], stop=False) for t in items):
                    return sum(Losses.smape(t, torch.zeros_like(t), eps=eps) for t in items)

                # Case 3: iterable of iterable-of-tensors (pairs)
                def is_pair(elem) -> bool:
                    if not TypeCheck.iterable(elem, stop=False):
                        return False
                    pair = list(elem)
                    if len(pair) != 2:
                        return False
                    a, b = pair
                    return TypeCheck.generic(a, [Tensor], stop=False) and TypeCheck.generic(b, [Tensor], stop=False)

                if items and all(is_pair(p) for p in items):
                    return sum(Losses.smape(p[0], p[1], eps=eps) for p in map(list, items))

            # If nothing matched, raise a helpful error
            raise TypeError(
                "smape-decorated function must return one of: "
                "Tensor; Iterable[Tensor]; Iterable[Iterable[Tensor]] where inner iterables are (pred, target) pairs."
            )
        return wrapper
    
    return decorator

#-- Mean Squared PErcentage Error
def mspe(*, eps: float = 1e-6) -> Callable[[Callable[..., Output]], Callable[..., Tensor]]:
    """
    Decorator factory that turns a function returning Output into an MSPE loss.

    Parameters
    ----------
    eps : float
        Zero approximation. Avoids division by zero.

    Notes
    -----
    Supports:
      - Tensor -> uses zeros_like as target
      - Iterable[Tensor] -> sum of mspe(t, 0) for each
      - Iterable[Iterable[Tensor]] -> assumes pairs (pred, target)
    """
    def decorator(fn: Callable[..., Output]) -> Callable[..., Tensor]:
        
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Tensor:
            out = fn(*args, **kwargs)

            # Case 1: single tensor
            if TypeCheck.generic(out, [Tensor], stop=False):
                return Losses.mspe(out, torch.zeros_like(out), eps=eps)

            # We may need to iterate more than once, so snapshot
            if isinstance(out, Iterable):
                items = list(out)

                # Case 2: iterable of tensors
                if items and all(TypeCheck.generic(t, [Tensor], stop=False) for t in items):
                    return sum(Losses.mspe(t, torch.zeros_like(t), eps=eps) for t in items)

                # Case 3: iterable of iterable-of-tensors (pairs)
                def is_pair(elem) -> bool:
                    if not TypeCheck.iterable(elem, stop=False):
                        return False
                    pair = list(elem)
                    if len(pair) != 2:
                        return False
                    a, b = pair
                    return TypeCheck.generic(a, [Tensor], stop=False) and TypeCheck.generic(b, [Tensor], stop=False)

                if items and all(is_pair(p) for p in items):
                    return sum(Losses.mspe(p[0], p[1], eps=eps) for p in map(list, items))

            # If nothing matched, raise a helpful error
            raise TypeError(
                "mspe-decorated function must return one of: "
                "Tensor; Iterable[Tensor]; Iterable[Iterable[Tensor]] where inner iterables are (pred, target) pairs."
            )
        return wrapper
    
    return decorator

#-- Symmetric Mean Squared Percentage Error
def smspe(*, eps: float = 1e-6) -> Callable[[Callable[..., Output]], Callable[..., Tensor]]:
    """
    Decorator factory that turns a function returning Output into an SMSPE loss.

    Parameters
    ----------
    eps : float
        Zero approximation. Avoids division by zero.

    Notes
    -----
    Supports:
      - Tensor -> uses zeros_like as target
      - Iterable[Tensor] -> sum of mspe(t, 0) for each
      - Iterable[Iterable[Tensor]] -> assumes pairs (pred, target)
    """
    def decorator(fn: Callable[..., Output]) -> Callable[..., Tensor]:
        
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Tensor:
            out = fn(*args, **kwargs)

            # Case 1: single tensor
            if TypeCheck.generic(out, [Tensor], stop=False):
                return Losses.smspe(out, torch.zeros_like(out), eps=eps)

            # We may need to iterate more than once, so snapshot
            if isinstance(out, Iterable):
                items = list(out)

                # Case 2: iterable of tensors
                if items and all(TypeCheck.generic(t, [Tensor], stop=False) for t in items):
                    return sum(Losses.smspe(t, torch.zeros_like(t), eps=eps) for t in items)

                # Case 3: iterable of iterable-of-tensors (pairs)
                def is_pair(elem) -> bool:
                    if not TypeCheck.iterable(elem, stop=False):
                        return False
                    pair = list(elem)
                    if len(pair) != 2:
                        return False
                    a, b = pair
                    return TypeCheck.generic(a, [Tensor], stop=False) and TypeCheck.generic(b, [Tensor], stop=False)

                if items and all(is_pair(p) for p in items):
                    return sum(Losses.smspe(p[0], p[1], eps=eps) for p in map(list, items))

            # If nothing matched, raise a helpful error
            raise TypeError(
                "smspe-decorated function must return one of: "
                "Tensor; Iterable[Tensor]; Iterable[Iterable[Tensor]] where inner iterables are (pred, target) pairs."
            )
        return wrapper
    
    return decorator

#####################################################################################