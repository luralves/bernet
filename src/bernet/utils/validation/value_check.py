#####################################################################################
from typing import Any, Iterable

#####################################################################################
class ValueCheck():

    @staticmethod
    def on_iterable(x: Any, values: Iterable) -> None:
        if x not in values:
            raise ValueError(f"ValueError: {x} must be one of {values}")
        return

#####################################################################################