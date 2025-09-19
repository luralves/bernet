#####################################################################################
from typing import Dict, Optional, Union

from bernet.contracts import (
    SamplerBASE,
    LossBASE,
    MetricsBASE,
    CallbackBASE,
    LoggerBASE,
    EarlyStopBASE,
)

import torch

#####################################################################################
class Trainer():
    """
    Trainer class for training neural networks.
    """


    def __init__(
        self,
        model: torch.nn.Module,
        sampler: SamplerBASE,
        loss: LossBASE,
        optimizer: torch.optim.Optimizer,
        metrics: Optional[MetricsBASE] = None,
        callback: Optional[CallbackBASE] = None,
        logger: Optional[LoggerBASE] = None,
        early_stopping: Optional[EarlyStopBASE] = None,
        device: Optional[Union[str, torch.device]] = "auto",
        ):
        """
        Parameters:
        ----------
        - model: nn.Module
          > The model to be trained.
        - sampler: SamplerABC
          > The sampler to be used for data loading.
        - losses: LossABC
          > The loss function to be used during training.
        - optimizer: torch.optim.Optimizer
          > The optimizer to be used for updating model weights.
        - metrics: Optional[MetricsABC]
          > The metrics to be used for evaluation.
        - callback: Optional[CallbackABC]
          > The callback for training events.
        - logger: Optional[LoggerABC]
           > The logger for logging training progress.
        - device: device
          > The device to run the training on. If "auto", selects GPU if available,
        """
        #-- Inputs
        self._model = model
        self._sampler = sampler
        self._loss = loss
        self._optimizer = optimizer
        self._metrics = metrics
        self._callback = callback
        self._early_stopping = early_stopping
        self._logger = logger

        #-- Computed parameters
        self._device = None

        # > Select device
        if device == "auto":
            if torch.cuda.is_available():
                self._device =  torch.device("cuda")
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self._device =  torch.device("mps")
            else:
                self._device =  torch.device("cpu")
        else:
            self._device = torch.device(device) if isinstance(device, str) else device


        #-- Move model to device
        self._model.to(self._device)
 
        return

    #-- Training loop
    def fit(
        self,
        num_epochs: int,
        ) -> None:
        """
        Perform training.

        Parameters:
        ----------
        - num_epochs: int
          > Number of epochs to train the model for.
        """

        #-- Callback
        if self._callback:
            self._callback.on_train_start()

        #-- Log start
        if self._logger is not None:
            self._logger.start()
        
        try:

            #-- Loop through epochs
            for epoch in range(num_epochs):

                #-- Callback
                if self._callback:
                    self._callback.on_epoch_start()

                #-- Reset epoch loss
                loss_epoch = 0.0
                terms_epoch: Dict[str, float] = {}

                #-- Loop through batches
                for batch in range(self._sampler.num_batches):

                    #-- Callback
                    if self._callback:
                        self._callback.on_batch_start()

                    #-- Sample batch
                    batch = self._sampler.batch()
                    batch = {k: v.to(self._device) for k, v in batch.items()}
                    
                    #-- Zero gradients
                    self._optimizer.zero_grad(set_to_none=True)

                    #-- Compute loss
                    total_loss, loss_terms = self._loss(model=self._model, batch=batch)

                    #-- Log loss
                    loss_epoch += total_loss.item()
                    terms_epoch = {k: terms_epoch.get(k, 0.0) + v.item() for k, v in (loss_terms or {}).items()}

                    #-- Backward pass
                    total_loss.backward()

                    #-- Adjust learning weights
                    self._optimizer.step()

                    #-- Callback
                    if self._callback:
                        self._callback.on_batch_end()
                
                #-- Compute metrics
                if self._metrics is not None:
                    metrics = self._metrics(
                        model=self._model,
                        data=self._sampler.metrics(),
                    )
                else:
                    metrics = None
                
                #-- Log epoch
                if self._logger is not None:
                    avg_loss = loss_epoch / self._sampler.num_batches
                    avg_terms = {k: v / self._sampler.num_batches for k, v in terms_epoch.items()}
                    self._logger.epoch(epoch=epoch, loss=avg_loss, terms=avg_terms, metrics=metrics)

                #-- Early stopping
                if self._early_stopping:

                    #-- Compute stopping criteria
                    if self._logger is not None:
                        stop = self._early_stopping(data={key: value[-1] for key, value in self._logger.data.items()})
                    else:
                        if not (avg_loss is not None and avg_terms is not None):
                            avg_loss = loss_epoch / self._sampler.num_batches
                            avg_terms = {k: v / self._sampler.num_batches for k, v in terms_epoch.items()}
                        stop = self._early_stopping(data={"loss": avg_loss, **avg_terms})
                    
                    #-- Stop if criteria met
                    if stop:
                        if self._logger is not None:
                            self._logger.stopped()
                        break
                
                #-- Callback
                if self._callback:
                    self._callback.on_epoch_end()

        except BaseException as e:
            #-- Callback
            if self._callback:
                self._callback.on_exception(e)

            #-- Log exception
            if self._logger is not None:
                self._logger.exception(e)

        finally:
            #-- Callback
            if self._callback:
                self._callback.on_train_end()

            #-- Log close
            if self._logger is not None:
                self._logger.close()
        
        return