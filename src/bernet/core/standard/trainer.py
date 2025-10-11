#####################################################################################
import torch

from typing import List, Literal, Optional, get_args

from bernet.interface.abstract.metrics import IMetrics
from bernet.interface.abstract.sampler import ISampler
from bernet.interface.abstract.loss import ILoss
from bernet.interface.abstract.callbacks import ICallbacks
from bernet.interface.abstract.early_stop import IEarlyStop
from bernet.interface.abstract.logger import ILogger
from bernet.interface.abstract.trainer import ITrainer
from bernet.interface.typing.dataclass import Losses
from bernet.interface.typing.aliases import Model, Optimizer

from bernet.core.metrics import MSE, MAE, MSPE, MAPE

from bernet.utils.validation import TypeCheck, ValueCheck
from bernet.utils.processing import Initialization

#####################################################################################
#--
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

#--
class Trainer(ITrainer):
    
    def __init__(
            self,
            model: Model,
            sampler: ISampler,
            loss: ILoss,
            optimizer: Optimizer,
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
        TypeCheck.sequence(metrics, include_none=True)
        TypeCheck.generic(callbacks, [ICallbacks], include_none=True)
        TypeCheck.generic(early_stop, [IEarlyStop], include_none=True)
        TypeCheck.sequence(initialization, include_none=True)
        TypeCheck.generic(logger, [ILogger], include_none=True)
        TypeCheck.str(device, include_none=True)

        if metrics is not None:
            for metric in metrics:
                ValueCheck.on_iterable(metric, get_args(Metrics))
        
        if initialization is not None:
            ValueCheck.on_iterable(initialization, get_args(Weights))
        
        #-- Inputs
        self.callbacks = callbacks
        self.early_stop = early_stop
        self.logger = logger

        #-- Auxiliary parameters
        self.metrics: List[IMetrics] | None = [] if metrics is not None else None
        self.device: torch.device = None

        #-- Select metrics
        if metrics is not None:
            for v in metrics:
                if v == "mse":
                    self.metrics.append(MSE())

                if v == "mae":
                    self.metrics.append(MAE())

                if v == "mspe":
                    self.metrics.append(MSPE())

                if v == "mape":
                    self.metrics.append(MAPE())


        #-- Select device
        if device == "auto":
            if torch.cuda.is_available():
                self.device =  torch.device("cuda")
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self.device =  torch.device("mps")
            else:
                self.device =  torch.device("cpu")
        else:
            self.device = torch.device(device) if isinstance(device, str) else device

        #-- Initialize model weights
        if initialization:
            if initialization == "xavier_uniform":
                model.apply(Initialization.xavier_uniform)

            if initialization == "xavier_normal":
                model.apply(Initialization.xavier_normal)

            if initialization == "kaiming_uniform":
                model.apply(Initialization.kaiming_uniform)

            if initialization == "kaiming_normal":
                model.apply(Initialization.kaiming_normal)

            if initialization == "orthogonal":
                model.apply(Initialization.orthogonal)
        
        #-- Move model to device
        self.model.to(self.device)
        
        return

    #-- Override
    def fit(
            self,
            num_epochs: int,
            *,
            verbose: bool  = True,
        ) -> None:
        
        #-- Callback
        if self.callbacks:
            self.callbacks.train_start()

        #-- Log start
        if self.logger is not None:
            self.logger.train_start(model=self.model, optimizer=self.optimizer)
        
        try:

            #-- Loop through epochs
            for epoch in range(num_epochs):

                #-- Callback
                if self.callbacks:
                    self.callbacks.epoch_start()
                
                #-- Create batches
                num_batches = self.sampler.generate(self.device)

                #-- Reset epoch loss
                terms_epoch = Losses(
                    residual=torch.tensor(0.0),
                    boundary=torch.tensor(0.0),
                    initial=torch.tensor(0.0),
                    observational=torch.tensor(0.0),
                )

                #-- Loop through batches
                for index in range(num_batches):

                    #-- Callback
                    if self.callbacks:
                        self.callbacks.batch_start()

                    #-- Sample batch
                    batch = self.sampler.batch(index)

                    #-- Zero gradients
                    self.optimizer.zero_grad(set_to_none=True)

                    #-- Compute loss
                    terms = self.loss.compute(model=self.model, batch=batch)

                    #-- Compute total loss
                    loss = terms.sum()

                    #-- Backward pass
                    loss.backward()

                    #-- Adjust learning weights
                    self.optimizer.step()

                    #-- Sum to epoch loss
                    terms_epoch = terms_epoch + terms

                    #-- Callback
                    if self.callbacks:
                        self.callbacks.batch_end()
                    
                #-- Compute metrics
                if self.metrics is not None:
                    t_sample = self.sampler.test()
                    tests = {}
                    for metric in self.metrics:
                        for k, v in metric.evaluate(model=self.model, data=t_sample).items():
                            tests[k] = v.item()
                else:
                    tests = None
                
                #-- Compute the average loss values
                avg_terms = terms_epoch / num_batches

                #-- Log epoch
                if self.logger is not None:
                    self.logger.epoch_end(losses=avg_terms, tests=tests)

                #-- Early stopping
                if self.early_stop:

                    #-- Compute stopping criteria
                    stop = self.early_stop.evaluate(losses=avg_terms, tests=tests)
                    
                    #-- Stop if criteria met
                    if stop:
                        if self.logger is not None:
                            self.logger.training_end(stopped=True)
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
                if self.callbacks:
                    self.callbacks.epoch_end()

        # except Exception as e:
        #     #-- Callback
        #     if self.callbacks:
        #         self.callbacks.exception(e)

        #     #-- Log exception
        #     if self.logger is not None:
        #         self.logger.exception(e)

        #     #-- Show error
        #     if verbose:
        #         print(e)

        finally:
            #-- Callback
            if self.callbacks:
                self.callbacks.train_end()

            #-- Log close
            if self.logger is not None:
                self.logger.training_end(stopped=False)
        
        return None if self.logger is None else self.logger.data

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
        self.logger.save(filename)

        return

#####################################################################################