#####################################################################################
from typing import Any, Iterable

#####################################################################################
class ValueCheck():

    @staticmethod
    def on_iterable(value: Any, iterable: Iterable[Any], stop: bool = True) -> None:
        if value not in iterable:
            if stop:
                raise ValueError(f"ValueError: {value} must be in {iterable}")
            else:
                return False
        return True

#####################################################################################