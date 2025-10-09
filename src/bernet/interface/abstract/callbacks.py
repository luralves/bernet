#####################################################################################
from abc import ABC, abstractmethod

#####################################################################################
class ICallbacks(ABC):
    """
    Callbacks to be called during the training process.
    
    Notes
    -----
    The methods were created with the intention of being
    called at specific moments in the training process.
    """

    @abstractmethod
    def train_start(self, *args, **kargs) -> None:
        """Intended to be called when training starts"""
        ...

    @abstractmethod
    def epoch_start(self, *args, **kargs) -> None:
        """Intended to be called at the start of each epoch"""
        ...

    @abstractmethod
    def batch_start(self, *args, **kargs) -> None:
        """Intended to be called at the start of each batch"""
        ...

    @abstractmethod
    def batch_end(self, *args, **kargs) -> None:
        """Intended to be called at the end of each batch"""
        ...

    @abstractmethod
    def epoch_end(self, *args, **kargs) -> None:
        """Intended to be called at the end of each epoch"""
        ...

    @abstractmethod
    def exception(self, *args, **kargs) -> None:
        """Intended to be called when an exception occurs during training"""
        ...

    @abstractmethod
    def train_end(self, *args, **kargs) -> None:
        """Intended to be called when training ends"""
        ...

#####################################################################################