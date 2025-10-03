#####################################################################################
import torch

from typing import Any, Iterable, get_args

#####################################################################################
class TypeCheck():
    
    @staticmethod
    def float(x: Any, name: str = "x") -> None:
        if not isinstance(x, float):
            raise TypeError(f"TypeError: '{name}' must be of type float")
        return
    
    @staticmethod
    def int(x: Any, name: str = "x") -> None:
        if not isinstance(x, int):
            raise TypeError(f"TypeError: '{name}' must be of type int")
        return
    
    @staticmethod
    def number(x: Any, name: str = "x") -> None:
        if not (isinstance(x, float) or isinstance(x, int)):
            raise TypeError(f"TypeError: '{name}' must be of type float or int")
        return
    
    @staticmethod
    def bool(x: Any, name: str = "x") -> None:
        if not isinstance(x, bool):
            raise TypeError(f"TypeError: '{name}' must be of type bool")
        return
    
    @staticmethod
    def callable(x: Any, name: str = "x") -> None:
        if not callable(x):
            raise TypeError(f"TypeError: '{name}' must be of type callable")
        return
    
    @staticmethod
    def torch_dtype(x: Any, name: str = "x") -> None:
        if not isinstance(x, torch.dtype):
            raise TypeError(f"TypeError: '{name}' must be of type torch.dtype")
        return
    
    @staticmethod
    def float_none(x: Any, name: str = "x") -> None:
        if not (isinstance(x, float) or x is None):
            raise TypeError(f"TypeError: '{name}' must be of type float or None")
        return
    
    @staticmethod
    def int_none(x: Any, name: str = "x") -> None:
        if not (isinstance(x, int) or x is None):
            raise TypeError(f"TypeError: '{name}' must be of type int or None")
        return
    
    @staticmethod
    def iterable(x: Any, name: str = "x") -> None:
        if not isinstance(x, Iterable):
            raise TypeError(f"TypeError: '{name}' must be of type Iterable")
        return
    
#####################################################################################