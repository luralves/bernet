#####################################################################################
from abc import ABC, abstractmethod

#####################################################################################
class ICallbacks(ABC):
    """
    Callbacks that are called during the training process.
    """

    @abstractmethod
    def train_start(self) -> None:
        """
        Called when training starts.
        """
        ...

    @abstractmethod
    def epoch_start(self) -> None:
        """
        Called at the start of each epoch.
        """
        ...

    @abstractmethod
    def batch_start(self) -> None:
        """
        Called at the start of each batch.
        """
        ...

    @abstractmethod
    def batch_end(self) -> None:
        """
        Called at the end of each batch.
        """
        ...

    @abstractmethod
    def epoch_end(self) -> None:
        """
        Called at the end of each epoch.
        """
        ...

    @abstractmethod
    def exception(
            self,
            e: BaseException,
        ) -> None:
        """
        Called when an exception occurs during training.

        Parameters
        ----------
        e : BaseException
            The exception that was raised.
        """
        ...

    @abstractmethod
    def train_end(
            self,
            stopped: bool,
        ) -> None:
        """
        Called when training ends.

        Parameters
        ----------
        stopped : bool
            True if stopped by EarlyStop
        """
        ...

#####################################################################################