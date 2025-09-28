#####################################################################################
import torch

from typing import Optional

from bernet.contracts.sampler import ISampler
from bernet.contracts.loss import ILoss, Losses
from bernet.contracts.metrics import IMetrics
from bernet.contracts.callbacks import ICallbacks
from bernet.contracts.logger import ILogger
from bernet.contracts.early_stop import IEarlyStop
from bernet.contracts.trainer import ITrainer

#####################################################################################
class DFLTTrainer(ITrainer):
    
    def __init__(
            self,
            model: torch.nn.Module,
            sampler: ISampler,
            loss: ILoss,
            optimizer: torch.optim.Optimizer,
            metrics: Optional[IMetrics] = None,
            callbacks: Optional[ICallbacks] = None,
            logger: Optional[ILogger] = None,
            early_stop: Optional[IEarlyStop] = None,
            device: Optional[str | torch.device] = "cpu",
        ) -> None:
        super().__init__(model, sampler, loss, optimizer, metrics, callbacks, logger, early_stop, device)
        return

    #-- Override
    def fit(
        self,
        num_epochs: int,
        verbose: bool  = True,
        ) -> None:
        
        #-- Callback
        if self.callbacks:
            self.callbacks.on_train_start()

        #-- Log start
        if self.logger is not None:
            self.logger.on_train_start(model=self.model, optimizer=self.optimizer)
        
        try:

            #-- Loop through epochs
            for epoch in range(num_epochs):

                #-- Callback
                if self.callbacks:
                    self.callbacks.on_epoch_start()
                
                #-- Create batches
                num_batches = self.sampler.generate_training_data()

                #-- Reset epoch loss
                terms_epoch = Losses(.0, .0, .0, .0)

                #-- Loop through batches
                for batch in range(num_batches):

                    #-- Callback
                    if self.callbacks:
                        self.callbacks.on_batch_start()

                    #-- Sample batch
                    batch = self.sampler.get_training_batch()
                    
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
                        self.callbacks.on_batch_end()
                    
                #-- Compute metrics
                if self.metrics is not None:
                    metrics = self.metrics(
                        model=self.model,
                        data=self.sampler.get_metrics_data()
                    )
                else:
                    metrics = None
                
                #-- Compute the average loss values
                avg_terms = terms / num_batches

                #-- Log epoch
                if self.logger is not None:
                    self.logger.on_epoch_end(terms=avg_terms, metrics=metrics)

                #-- Early stopping
                if self.early_stop:

                    #-- Compute stopping criteria
                    stop = self.early_stop.evaluate(terms=avg_terms, metrics=metrics)
                    
                    #-- Stop if criteria met
                    if stop:
                        if self.logger is not None:
                            self.logger.on_training_end(stopped=True)
                        break
                
                #-- Show progress
                if verbose:
                    terms_str = [f"{num:.3e}" for num in avg_terms.to_list()]
                    print(f"Epoch {epoch + 1}/{num_epochs}; Loss: {avg_terms.sum():.3e}; Terms: " + terms_str)
                
                #-- Callback
                if self.callbacks:
                    self.callbacks.on_epoch_end()

        except BaseException as e:
            #-- Callback
            if self.callbacks:
                self.callbacks.on_exception(e)

            #-- Log exception
            if self.logger is not None:
                self.logger.on_exception(e)

            #-- Show error
            if verbose:
                print(f"Error: {e}")

        finally:
            #-- Callback
            if self.callbacks:
                self.callbacks.on_train_end()

            #-- Log close
            if self.logger is not None:
                self.logger.on_training_end(stopped=False)
        
        return