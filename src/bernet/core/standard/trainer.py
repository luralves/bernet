#####################################################################################
import torch

from typing import List, Literal, Optional, get_args

from bernet.contracts.metrics import IMetrics
from bernet.contracts.sampler import ISampler
from bernet.contracts.loss import ILoss, BatchLoss
from bernet.contracts.callbacks import ICallbacks
from bernet.contracts.early_stop import IEarlyStop
from bernet.contracts.logger import ILogger
from bernet.contracts.trainer import ITrainer

from bernet.core.metrics import MSE, MAE, MSPE, MAPE

from bernet.utils.validation import TypeCheck, ValueCheck
from bernet.utils.processing import Initialization

#####################################################################################
Metrics = Literal[
    "mse",
    "mae",
    "mspe",
    "mape",
]

Weights = Literal[
    "xavier_uniform",
    "xavier_normal",
    "kaiming_uniform",
    "kaiming_normal",
    "orthogonal",
]

class Trainer(ITrainer):
    
    def __init__(
            self,
            model: torch.nn.Module,
            sampler: ISampler,
            loss: ILoss,
            optimizer: torch.optim.Optimizer,
            *,
            metrics: Optional[List[Metrics]] = None,
            callbacks: Optional[ICallbacks] = None,
            early_stop: Optional[IEarlyStop] = None,
            initialization: Optional[Weights] = "",
            logger: Optional[ILogger] = None,
            device: Optional[str | torch.device] = "cpu",
        ) -> None:
        """
        Parameters
        ----------
        model : nn.Module
            The model to be trained.
        sampler : ISampler
            The sampler to be used for data loading.
        loss : ILoss
            The loss function to be used during training.
        optimizer : torch.optim.Optimizer
            The optimizer to be used for updating model weights.
        metrics : Optional[List[Metrics]]
            The metrics to be used for evaluation.
        callback : Optional[Callback]
            The callback for training events.
        early_stop : Optional[IEarlyStop]
            The early stop functionality.
        logger : Optional[ILogger]
            Logger class necessary to store data training.
        device : device
            The device to run the training on. If "auto", selects GPU if available,
        """
        super().__init__(model, sampler, loss, optimizer)

        #-- Verification
        TypeCheck.iterable_none(metrics)
        TypeCheck.abc_none(callbacks, ICallbacks)
        TypeCheck.abc_none(early_stop, IEarlyStop)
        TypeCheck.iterable_none(initialization)
        TypeCheck.abc_none(logger, ILogger)
        TypeCheck.str_none(device)

        if metrics is not None:
            for metric in metrics:
                ValueCheck.on_iterable(metric, get_args(Metrics))
        
        if initialization is not None:
            ValueCheck.on_iterable(initialization, get_args(Weights))
        
        #-- Inputs
        self._callbacks = callbacks
        self._early_stop = early_stop
        self._logger = logger

        #-- Auxiliary parameters
        self._metrics: List[IMetrics] | None = [] if metrics is not None else None
        self._device: torch.device = None

        #-- Select metrics
        if metrics is not None:
            for v in metrics:
                if v == "mse": self._metrics.append(MSE())
                if v == "mae": self._metrics.append(MAE())
                if v == "mspe": self._metrics.append(MSPE())
                if v == "mape": self._metrics.append(MAPE())


        #-- Select device
        if device == "auto":
            if torch.cuda.is_available():
                self._device =  torch.device("cuda")
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self._device =  torch.device("mps")
            else:
                self._device =  torch.device("cpu")
        else:
            self._device = torch.device(device) if isinstance(device, str) else device

        #-- Initialize model weights
        if initialization:
            if initialization == "xavier_uniform": model.apply(Initialization.xavier_uniform)
            if initialization == "xavier_normal": model.apply(Initialization.xavier_normal)
            if initialization == "kaiming_uniform": model.apply(Initialization.kaiming_uniform)
            if initialization == "kaiming_normal": model.apply(Initialization.kaiming_normal)
            if initialization == "orthogonal": model.apply(Initialization.orthogonal)
        
        #-- Move model to device
        self._model.to(self._device)
        
        return

    #-- Override
    def fit(
            self,
            num_epochs: int,
            *,
            verbose: bool  = True,
        ) -> None:
        
        #-- Callback
        if self._callbacks:
            self._callbacks.train_start()

        #-- Log start
        if self._logger is not None:
            self._logger.train_start(model=self._model, optimizer=self._optimizer)
        
        try:

            #-- Loop through epochs
            for epoch in range(num_epochs):

                #-- Callback
                if self._callbacks:
                    self._callbacks.epoch_start()
                
                #-- Create batches
                num_batches = self._sampler.generate()

                #-- Reset epoch loss
                terms_epoch = BatchLoss(.0, .0, .0, .0)

                #-- Loop through batches
                for index in range(num_batches):

                    #-- Callback
                    if self._callbacks:
                        self._callbacks.batch_start()

                    #-- Sample batch
                    batch = self._sampler.batch(index)
                    
                    #-- Move batch tensors to device
                    batch.to_device(self._device)
                    
                    #-- Zero gradients
                    self._optimizer.zero_grad(set_to_none=True)

                    #-- Compute loss
                    terms = self._loss.compute(model=self._model, batch=batch)

                    #-- Compute total loss
                    loss = terms.sum()

                    #-- Backward pass
                    loss.backward()

                    #-- Adjust learning weights
                    self._optimizer.step()

                    #-- Sum to epoch loss
                    terms_epoch += terms.to_float()

                    #-- Callback
                    if self._callbacks:
                        self._callbacks.batch_end()
                    
                #-- Compute metrics
                if self._metrics is not None:
                    t_sample = self._sampler.test()
                    tests = {}
                    for metric in self._metrics:
                        for k, v in metric.evaluate(model=self._model, data=t_sample).items():
                            tests[k] = v.item()
                else:
                    tests = None
                
                #-- Compute the average loss values
                avg_terms = terms_epoch / num_batches

                #-- Log epoch
                if self._logger is not None:
                    self._logger.epoch_end(losses=avg_terms, tests=tests)

                #-- Early stopping
                if self._early_stop:

                    #-- Compute stopping criteria
                    stop = self._early_stop.evaluate(losses=avg_terms, tests=tests)
                    
                    #-- Stop if criteria met
                    if stop:
                        if self._logger is not None:
                            self._logger.training_end(stopped=True)
                        break
                
                #-- Show progress
                if verbose:
                    if tests is not None:

                        print(
                            f"Epoch {epoch + 1}/{num_epochs}; "
                            f"| ILoss: {avg_terms.sum():.3e} "
                            f"({avg_terms.residual:.3e}; {avg_terms.boundary:.3e}, "
                            f"{avg_terms.initial:.3e}, {avg_terms.observational:.3e}) "
                            f"| [{', '.join(f'{k}: {v:.3e}' for k, v in tests.items())}]"
                        )
                    else:
                        print(
                            f"Epoch {epoch + 1}/{num_epochs}; "
                            f"ILoss: {avg_terms.sum():.3e}; "
                            f"Terms: [{avg_terms.residual:.3e}, {avg_terms.boundary:.3e}, "
                            f"{avg_terms.initial:.3e}, {avg_terms.observational:.3e}] "
                        )
                
                #-- Callback
                if self._callbacks:
                    self._callbacks.epoch_end()

        except Exception as e:
            #-- Callback
            if self._callbacks:
                self._callbacks.exception(e)

            #-- Log exception
            if self._logger is not None:
                self._logger.exception(e)

            #-- Show error
            if verbose:
                print(e)

        finally:
            #-- Callback
            if self._callbacks:
                self._callbacks.train_end()

            #-- Log close
            if self._logger is not None:
                self._logger.training_end(stopped=False)
        
        return None if self._logger is None else self._logger.data

    def save(self, filename: str) -> None:
        """
        Save logger to file.

        Parameters
        ----------
        filename : str
            The name of the file to save the data.
        """

        #-- Validation
        TypeCheck.str(filename)

        #-- Save
        self._logger.save(filename)

        return