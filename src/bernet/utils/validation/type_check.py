#####################################################################################
import torch

from typing import Any, Optional
from collections.abc import Iterable, Mapping

#####################################################################################
#-- Helper function
def _validation(value: Any, target: Iterable[Any], message: str, stop: bool) -> bool:
    if not (True in [isinstance(value, reference) for reference in target]):
        if stop:
            raise TypeError(f"Type error: {message}")
        else:
            return False
    return True

#-- Main class
class TypeCheck():
    
    @staticmethod
    def float(value: Any, message: Optional[str] = None, stop: bool = True, include_none: bool = False) -> None:
        return _validation(
            value=value,
            target=[float, type(None)] if include_none else [float],
            message=message if message is not None else f"{value} must be of type float",
            stop=stop,
        )
    
    @staticmethod
    def int(value: Any, message: Optional[str] = None, stop: bool = True, include_none: bool = False) -> None:
        return _validation(
            value=value,
            target=[int, type(None)] if include_none else [int],
            message=message if message is not None else f"{value} must be of type int",
            stop=stop,
        )
        
    @staticmethod
    def str(value: Any, message: Optional[str] = None, stop: bool = True, include_none: bool = False) -> None:
        return _validation(
            value=value,
            target=[str, type(None)] if include_none else [str],
            message=message if message is not None else f"{value} must be of type str",
            stop=stop,
        )
    
    @staticmethod
    def bool(value: Any, message: Optional[str] = None, stop: bool = True, include_none: bool = False) -> None:
        return _validation(
            value=value,
            target=[bool, type(None)] if include_none else [bool],
            message=message if message is not None else f"{value} must be of type bool",
            stop=stop,
        )
    
    @staticmethod
    def number(value: Any, message: Optional[str] = None, stop: bool = True, include_none: bool = False) -> None:
        return _validation(
            value=value,
            target=[float, int, type(None)] if include_none else [float, int],
            message=message if message is not None else f"{value} must be of type float or int",
            stop=stop,
        )
    
    @staticmethod
    def iterable(value: Any, message: Optional[str] = None, stop: bool = True, include_none: bool = False) -> bool:
        return _validation(
            value=value,
            target=[Iterable, type(None)] if include_none else [Iterable],
            message=message if message is not None else f"{value} must be an Iterable",
            stop=stop,
        )
    
    @staticmethod
    def mapping(value: Any, message: Optional[str] = None, stop: bool = True, include_none: bool = False) -> bool:
        return _validation(
            value=value,
            target=[Mapping, type(None)] if include_none else [Mapping],
            message=message if message is not None else f"{value} must be of type Mapping",
            stop=stop,
        )
    
    @staticmethod
    def generic(value: Any, target: Iterable[Any], message: Optional[str] = None, stop: bool = True, include_none: bool = False) -> bool:
        return _validation(
            value=value,
            target=list(target) + [type(None)] if include_none else target,
            message=message if message is not None else f"{value} must be one of {target}",
            stop=stop,
        )
    
    @staticmethod
    def callable(value: Any, message: Optional[str] = None, stop: bool = True, include_none: bool = False) -> bool:
        if include_none and value is None:
            return True
        if not callable(value):
            if stop:
                raise TypeError(f"TypeError: '{value}' must be of type callable")
            else:
                return False
        return True

#####################################################################################