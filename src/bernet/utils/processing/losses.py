#####################################################################################
import torch
import functools

from typing import Union, Iterable, Callable

from bernet.utils.validation.type_check import TypeCheck

#####################################################################################
#-- Class
class Losses:
    """
    Losses for continuous outputs.
    """

    @staticmethod
    def mse(y_hat: torch.Tensor, y_ref: torch.Tensor) -> torch.Tensor:
        """
        Mean Squared Error.
        
        Parameters
        ----------
        y_hat : torch.Tensor
            Predicted value.
        y_ref : torch.Tensor
            Reference value.
        
        Returns
        -------
        torch.Tensor
            Loss value
        """
        return torch.mean((y_hat - y_ref) ** 2)

    @staticmethod
    def mae(y_hat: torch.Tensor, y_ref: torch.Tensor) -> torch.Tensor:
        """
        Mean Absolute Error.
        
        Parameters
        ----------
        y_hat : torch.Tensor
            Predicted value.
        y_ref : torch.Tensor
            Reference value.
        
        Returns
        -------
        torch.Tensor
            Loss value
        """
        return torch.mean((y_hat - y_ref).abs())
    
    @staticmethod
    def log_cosh(y_hat: torch.Tensor, y_ref: torch.Tensor) -> torch.Tensor:
        """
        Log(cosh(x)) error.
        
        Parameters
        ----------
        y_hat : torch.Tensor
            Predicted value.
        y_ref : torch.Tensor
            Reference value.
        
        Returns
        -------
        torch.Tensor
            Loss value
        """
        return torch.mean(torch.log(torch.cosh(y_hat - y_ref)))
    
    @staticmethod
    def mape(y_hat: torch.Tensor, y_ref: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
        """
        Mean Absolute Percentage Error.
        
        Parameters
        ----------
        y_hat : torch.Tensor
            Predicted value.
        y_ref : torch.Tensor
            Reference value.
        
        Returns
        -------
        torch.Tensor
            Loss value
        """
        return torch.mean((y_hat - y_ref).abs() / y_ref.abs().clamp_min(min=eps))

    @staticmethod
    def smape(y_hat: torch.Tensor, y_ref: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
        """
        Simmetric Mean Absolute Percentage Error.
        
        Parameters
        ----------
        y_hat : torch.Tensor
            Predicted value.
        y_ref : torch.Tensor
            Reference value.
        
        Returns
        -------
        torch.Tensor
            Loss value
        """
        return 2.0 * torch.mean((y_hat - y_ref).abs() / (y_hat.abs() + y_ref.abs()).clamp_min(min=eps))

    @staticmethod
    def mspe(y_hat: torch.Tensor, y_ref: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
        """
        Mean Squared Percentage Error.
        
        Parameters
        ----------
        y_hat : torch.Tensor
            Predicted value.
        y_ref : torch.Tensor
            Reference value.
        
        Returns
        -------
        torch.Tensor
            Loss value
        """
        return torch.mean((y_hat - y_ref) ** 2 / (y_ref ** 2).clamp_min(min=eps))

    @staticmethod
    def smspe(y_hat: torch.Tensor, y_ref: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
        """
        Simmetric Mean Squared Percentage Error.
        
        Parameters
        ----------
        y_hat : torch.Tensor
            Predicted value.
        y_ref : torch.Tensor
            Reference value.
        
        Returns
        -------
        torch.Tensor
            Loss value
        """
        return 4.0 * torch.mean((y_hat - y_ref) ** 2 / ((y_hat ** 2) + (y_ref ** 2)).clamp_min(min=eps))

#-- Decorators
OperatorLoss = Union[torch.Tensor, Iterable[torch.Tensor], Iterable[Iterable[torch.Tensor]]]

def mse() -> Callable[[Callable[..., OperatorLoss]], Callable[..., torch.Tensor]]:
    """
    Decorator factory that turns a function returning OperatorLoss into an MSE loss.
    Supports:
      - Tensor -> uses zeros_like as target
      - Iterable[Tensor] -> sum of mspe(t, 0) for each
      - Iterable[Iterable[Tensor]] -> assumes pairs (pred, target)
    """
    def decorator(fn: Callable[..., OperatorLoss]) -> Callable[..., torch.Tensor]:
        
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> torch.Tensor:
            out = fn(*args, **kwargs)

            # Case 1: single tensor
            if TypeCheck.abc(out, torch.Tensor, stop=False):
                return Losses.mse(out, torch.zeros_like(out))

            # We may need to iterate more than once, so snapshot
            if isinstance(out, Iterable):
                items = list(out)

                # Case 2: iterable of tensors
                if items and all(TypeCheck.abc(t, torch.Tensor, stop=False) for t in items):
                    return sum(Losses.mse(t, torch.zeros_like(t)) for t in items)

                # Case 3: iterable of iterable-of-tensors (pairs)
                def is_pair(elem) -> bool:
                    if not TypeCheck.iterable(elem, stop=False):
                        return False
                    pair = list(elem)
                    if len(pair) != 2:
                        return False
                    a, b = pair
                    return TypeCheck.abc(a, torch.Tensor, stop=False) and TypeCheck.abc(b, torch.Tensor, stop=False)

                if items and all(is_pair(p) for p in items):
                    return sum(Losses.mse(p[0], p[1]) for p in map(list, items))

            # If nothing matched, raise a helpful error
            raise TypeError(
                "mse-decorated function must return one of: "
                "Tensor; Iterable[Tensor]; Iterable[Iterable[Tensor]] where inner iterables are (pred, target) pairs."
            )
        return wrapper
    
    return decorator

def mae() -> Callable[[Callable[..., OperatorLoss]], Callable[..., torch.Tensor]]:
    """
    Decorator factory that turns a function returning OperatorLoss into an MAE loss.
    Supports:
      - Tensor -> uses zeros_like as target
      - Iterable[Tensor] -> sum of mspe(t, 0) for each
      - Iterable[Iterable[Tensor]] -> assumes pairs (pred, target)
    """
    def decorator(fn: Callable[..., OperatorLoss]) -> Callable[..., torch.Tensor]:
        
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> torch.Tensor:
            out = fn(*args, **kwargs)

            # Case 1: single tensor
            if TypeCheck.abc(out, torch.Tensor, stop=False):
                return Losses.mae(out, torch.zeros_like(out))

            # We may need to iterate more than once, so snapshot
            if isinstance(out, Iterable):
                items = list(out)

                # Case 2: iterable of tensors
                if items and all(TypeCheck.abc(t, torch.Tensor, stop=False) for t in items):
                    return sum(Losses.mae(t, torch.zeros_like(t)) for t in items)

                # Case 3: iterable of iterable-of-tensors (pairs)
                def is_pair(elem) -> bool:
                    if not TypeCheck.iterable(elem, stop=False):
                        return False
                    pair = list(elem)
                    if len(pair) != 2:
                        return False
                    a, b = pair
                    return TypeCheck.abc(a, torch.Tensor, stop=False) and TypeCheck.abc(b, torch.Tensor, stop=False)

                if items and all(is_pair(p) for p in items):
                    return sum(Losses.mae(p[0], p[1]) for p in map(list, items))

            # If nothing matched, raise a helpful error
            raise TypeError(
                "mae-decorated function must return one of: "
                "Tensor; Iterable[Tensor]; Iterable[Iterable[Tensor]] where inner iterables are (pred, target) pairs."
            )
        return wrapper
    
    return decorator

def log_cosh() -> Callable[[Callable[..., OperatorLoss]], Callable[..., torch.Tensor]]:
    """
    Decorator factory that turns a function returning OperatorLoss into an LOG_COSH loss.
    Supports:
      - Tensor -> uses zeros_like as target
      - Iterable[Tensor] -> sum of mspe(t, 0) for each
      - Iterable[Iterable[Tensor]] -> assumes pairs (pred, target)
    """
    def decorator(fn: Callable[..., OperatorLoss]) -> Callable[..., torch.Tensor]:
        
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> torch.Tensor:
            out = fn(*args, **kwargs)

            # Case 1: single tensor
            if TypeCheck.abc(out, torch.Tensor, stop=False):
                return Losses.log_cosh(out, torch.zeros_like(out))

            # We may need to iterate more than once, so snapshot
            if isinstance(out, Iterable):
                items = list(out)

                # Case 2: iterable of tensors
                if items and all(TypeCheck.abc(t, torch.Tensor, stop=False) for t in items):
                    return sum(Losses.log_cosh(t, torch.zeros_like(t)) for t in items)

                # Case 3: iterable of iterable-of-tensors (pairs)
                def is_pair(elem) -> bool:
                    if not TypeCheck.iterable(elem, stop=False):
                        return False
                    pair = list(elem)
                    if len(pair) != 2:
                        return False
                    a, b = pair
                    return TypeCheck.abc(a, torch.Tensor, stop=False) and TypeCheck.abc(b, torch.Tensor, stop=False)

                if items and all(is_pair(p) for p in items):
                    return sum(Losses.log_cosh(p[0], p[1]) for p in map(list, items))

            # If nothing matched, raise a helpful error
            raise TypeError(
                "log_cosh-decorated function must return one of: "
                "Tensor; Iterable[Tensor]; Iterable[Iterable[Tensor]] where inner iterables are (pred, target) pairs."
            )
        return wrapper
    
    return decorator

def mape(*, eps: float = 1e-6) -> Callable[[Callable[..., OperatorLoss]], Callable[..., torch.Tensor]]:
    """
    Decorator factory that turns a function returning OperatorLoss into an MAPE loss.
    Supports:
      - Tensor -> uses zeros_like as target
      - Iterable[Tensor] -> sum of mspe(t, 0) for each
      - Iterable[Iterable[Tensor]] -> assumes pairs (pred, target)
    """
    def decorator(fn: Callable[..., OperatorLoss]) -> Callable[..., torch.Tensor]:
        
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> torch.Tensor:
            out = fn(*args, **kwargs)

            # Case 1: single tensor
            if TypeCheck.abc(out, torch.Tensor, stop=False):
                return Losses.mape(out, torch.zeros_like(out), eps=eps)

            # We may need to iterate more than once, so snapshot
            if isinstance(out, Iterable):
                items = list(out)

                # Case 2: iterable of tensors
                if items and all(TypeCheck.abc(t, torch.Tensor, stop=False) for t in items):
                    return sum(Losses.mape(t, torch.zeros_like(t), eps=eps) for t in items)

                # Case 3: iterable of iterable-of-tensors (pairs)
                def is_pair(elem) -> bool:
                    if not TypeCheck.iterable(elem, stop=False):
                        return False
                    pair = list(elem)
                    if len(pair) != 2:
                        return False
                    a, b = pair
                    return TypeCheck.abc(a, torch.Tensor, stop=False) and TypeCheck.abc(b, torch.Tensor, stop=False)

                if items and all(is_pair(p) for p in items):
                    return sum(Losses.mape(p[0], p[1], eps=eps) for p in map(list, items))

            # If nothing matched, raise a helpful error
            raise TypeError(
                "mape-decorated function must return one of: "
                "Tensor; Iterable[Tensor]; Iterable[Iterable[Tensor]] where inner iterables are (pred, target) pairs."
            )
        return wrapper
    
    return decorator

def smape(*, eps: float = 1e-6) -> Callable[[Callable[..., OperatorLoss]], Callable[..., torch.Tensor]]:
    """
    Decorator factory that turns a function returning OperatorLoss into an SMAPE loss.
    Supports:
      - Tensor -> uses zeros_like as target
      - Iterable[Tensor] -> sum of mspe(t, 0) for each
      - Iterable[Iterable[Tensor]] -> assumes pairs (pred, target)
    """
    def decorator(fn: Callable[..., OperatorLoss]) -> Callable[..., torch.Tensor]:
        
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> torch.Tensor:
            out = fn(*args, **kwargs)

            # Case 1: single tensor
            if TypeCheck.abc(out, torch.Tensor, stop=False):
                return Losses.smape(out, torch.zeros_like(out), eps=eps)

            # We may need to iterate more than once, so snapshot
            if isinstance(out, Iterable):
                items = list(out)

                # Case 2: iterable of tensors
                if items and all(TypeCheck.abc(t, torch.Tensor, stop=False) for t in items):
                    return sum(Losses.smape(t, torch.zeros_like(t), eps=eps) for t in items)

                # Case 3: iterable of iterable-of-tensors (pairs)
                def is_pair(elem) -> bool:
                    if not TypeCheck.iterable(elem, stop=False):
                        return False
                    pair = list(elem)
                    if len(pair) != 2:
                        return False
                    a, b = pair
                    return TypeCheck.abc(a, torch.Tensor, stop=False) and TypeCheck.abc(b, torch.Tensor, stop=False)

                if items and all(is_pair(p) for p in items):
                    return sum(Losses.smape(p[0], p[1], eps=eps) for p in map(list, items))

            # If nothing matched, raise a helpful error
            raise TypeError(
                "smape-decorated function must return one of: "
                "Tensor; Iterable[Tensor]; Iterable[Iterable[Tensor]] where inner iterables are (pred, target) pairs."
            )
        return wrapper
    
    return decorator

def mspe(*, eps: float = 1e-6) -> Callable[[Callable[..., OperatorLoss]], Callable[..., torch.Tensor]]:
    """
    Decorator factory that turns a function returning OperatorLoss into an MSPE loss.
    Supports:
      - Tensor -> uses zeros_like as target
      - Iterable[Tensor] -> sum of mspe(t, 0) for each
      - Iterable[Iterable[Tensor]] -> assumes pairs (pred, target)
    """
    def decorator(fn: Callable[..., OperatorLoss]) -> Callable[..., torch.Tensor]:
        
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> torch.Tensor:
            out = fn(*args, **kwargs)

            # Case 1: single tensor
            if TypeCheck.abc(out, torch.Tensor, stop=False):
                return Losses.mspe(out, torch.zeros_like(out), eps=eps)

            # We may need to iterate more than once, so snapshot
            if isinstance(out, Iterable):
                items = list(out)

                # Case 2: iterable of tensors
                if items and all(TypeCheck.abc(t, torch.Tensor, stop=False) for t in items):
                    return sum(Losses.mspe(t, torch.zeros_like(t), eps=eps) for t in items)

                # Case 3: iterable of iterable-of-tensors (pairs)
                def is_pair(elem) -> bool:
                    if not TypeCheck.iterable(elem, stop=False):
                        return False
                    pair = list(elem)
                    if len(pair) != 2:
                        return False
                    a, b = pair
                    return TypeCheck.abc(a, torch.Tensor, stop=False) and TypeCheck.abc(b, torch.Tensor, stop=False)

                if items and all(is_pair(p) for p in items):
                    return sum(Losses.mspe(p[0], p[1], eps=eps) for p in map(list, items))

            # If nothing matched, raise a helpful error
            raise TypeError(
                "mspe-decorated function must return one of: "
                "Tensor; Iterable[Tensor]; Iterable[Iterable[Tensor]] where inner iterables are (pred, target) pairs."
            )
        return wrapper
    
    return decorator

def smspe(*, eps: float = 1e-6) -> Callable[[Callable[..., OperatorLoss]], Callable[..., torch.Tensor]]:
    """
    Decorator factory that turns a function returning OperatorLoss into an SMSPE loss.
    Supports:
      - Tensor -> uses zeros_like as target
      - Iterable[Tensor] -> sum of mspe(t, 0) for each
      - Iterable[Iterable[Tensor]] -> assumes pairs (pred, target)
    """
    def decorator(fn: Callable[..., OperatorLoss]) -> Callable[..., torch.Tensor]:
        
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> torch.Tensor:
            out = fn(*args, **kwargs)

            # Case 1: single tensor
            if TypeCheck.abc(out, torch.Tensor, stop=False):
                return Losses.smspe(out, torch.zeros_like(out), eps=eps)

            # We may need to iterate more than once, so snapshot
            if isinstance(out, Iterable):
                items = list(out)

                # Case 2: iterable of tensors
                if items and all(TypeCheck.abc(t, torch.Tensor, stop=False) for t in items):
                    return sum(Losses.smspe(t, torch.zeros_like(t), eps=eps) for t in items)

                # Case 3: iterable of iterable-of-tensors (pairs)
                def is_pair(elem) -> bool:
                    if not TypeCheck.iterable(elem, stop=False):
                        return False
                    pair = list(elem)
                    if len(pair) != 2:
                        return False
                    a, b = pair
                    return TypeCheck.abc(a, torch.Tensor, stop=False) and TypeCheck.abc(b, torch.Tensor, stop=False)

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