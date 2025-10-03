#####################################################################################
import torch

from typing import Optional

from bernet.contracts.sampler import SamplerABC
from bernet.contracts.loss import LossABC, BatchLoss
from bernet.contracts.metrics import MetricsABC
from bernet.contracts.callbacks import CallbacksABC
from bernet.contracts.logger import LoggerABC
from bernet.contracts.early_stop import EarlyStopABC
from bernet.contracts.trainer import TrainerABC

#####################################################################################
class DefaultTrainer(TrainerABC):
    
    def __init__(
            self,
            model: torch.nn.Module,
            sampler: SamplerABC,
            loss: LossABC,
            optimizer: torch.optim.Optimizer,
            *,
            metrics: Optional[MetricsABC] = None,
            callbacks: Optional[CallbacksABC] = None,
            logger: Optional[LoggerABC] = None,
            early_stop: Optional[EarlyStopABC] = None,
            device: Optional[str | torch.device] = "cpu",
        ) -> None:
        super().__init__(model, sampler, loss, optimizer, metrics, callbacks, logger, early_stop, device)
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
                num_batches = self.sampler.generate()

                #-- Reset epoch loss
                terms_epoch = BatchLoss(.0, .0, .0, .0)

                #-- Loop through batches
                for index in range(num_batches):

                    #-- Callback
                    if self.callbacks:
                        self.callbacks.batch_start()

                    #-- Sample batch
                    batch = self.sampler.batch(index)
                    
                    #-- Move batch tensors to device
                    batch.to_device(self.device)
                    
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
                    terms_epoch += terms.to_float()

                    #-- Callback
                    if self.callbacks:
                        self.callbacks.batch_end()
                    
                #-- Compute metrics
                if self.metrics is not None:
                    m_sample = self.sampler.metrics()
                    metrics = {}
                    for metric in self.metrics:
                        for k, v in metric.evaluate(model=self.model, data=m_sample).items():
                            metrics[k] = v.item()
                else:
                    metrics = None
                
                #-- Compute the average loss values
                avg_terms = terms_epoch / num_batches

                #-- Log epoch
                if self.logger is not None:
                    self.logger.epoch_end(losses=avg_terms, metrics=metrics)

                #-- Early stopping
                if self.early_stop:

                    #-- Compute stopping criteria
                    stop = self.early_stop.evaluate(losses=avg_terms, metrics=metrics)
                    
                    #-- Stop if criteria met
                    if stop:
                        if self.logger is not None:
                            self.logger.training_end(stopped=True)
                        break
                
                #-- Show progress
                if verbose:
                    if self.metrics is not None:

                        print(
                            f"Epoch {epoch + 1}/{num_epochs}; "
                            f"| Loss: {avg_terms.sum():.3e} "
                            f"({avg_terms.residual:.3e}; {avg_terms.boundary:.3e}, "
                            f"{avg_terms.initial:.3e}, {avg_terms.observational:.3e}) "
                            f"| [{', '.join(f'{k}: {v:.3e}' for k, v in metrics.items())}]"
                        )
                    else:
                        print(
                            f"Epoch {epoch + 1}/{num_epochs}; "
                            f"Loss: {avg_terms.sum():.3e}; "
                            f"Terms: [{avg_terms.residual:.3e}, {avg_terms.boundary:.3e}, "
                            f"{avg_terms.initial:.3e}, {avg_terms.observational:.3e}] "
                        )
                
                #-- Callback
                if self.callbacks:
                    self.callbacks.epoch_end()

        except Exception as e:
            #-- Callback
            if self.callbacks:
                self.callbacks.exception(e)

            #-- Log exception
            if self.logger is not None:
                self.logger.exception(e)

            #-- Show error
            if verbose:
                print(e)

        finally:
            #-- Callback
            if self.callbacks:
                self.callbacks.train_end()

            #-- Log close
            if self.logger is not None:
                self.logger.training_end(stopped=False)
        
        return