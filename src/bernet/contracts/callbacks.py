#####################################################################################
from abc import ABC

#####################################################################################
class CallbackBASE(ABC):
    """
    Optional callback hooks that are called during the training process.
    Override only what you need.
    """
    def on_train_start(self) -> None:
        """
        Called when training starts.
        """
        ...

    def on_epoch_start(self) -> None:
        """
        Called at the start of each epoch.
        """
        ...

    def on_batch_start(self) -> None:
        """
        Called at the start of each batch.
        """
        ...

    def on_batch_end(self) -> None:
        """
        Called at the end of each batch.
        """
        ...

    def on_epoch_end(self) -> None:
        """
        Called at the end of each epoch.
        """
        ...

    def on_exception(self, e: BaseException) -> None:
        """
        Called when an exception occurs during training.

        Parameters:
        ----------
        - e: BaseException
          > The exception that was raised.
        """
        ...

    def on_train_end(self) -> None:
        """
        Called when training ends.
        """
        ...

#####################################################################################