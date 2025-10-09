#####################################################################################
from abc import ABC, abstractmethod

#####################################################################################
class ILogger(ABC):
    """Stores a summary about the model and training data"""

    @abstractmethod
    def train_start(self, *args, **kargs) -> None:
        """Intended to be called when training starts"""
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
    def training_end(self, *args, **kargs) -> None:
        """Intended to be called when training ends"""
        ...
    
    @abstractmethod
    def save(self, filename: str) -> None:
        """
        Save the data into a log file.

        Parameters
        ----------
        filename : str
            The name of the file.
        """
        ...
    
#####################################################################################