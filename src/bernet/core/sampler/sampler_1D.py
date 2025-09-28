#####################################################################################
import torch
import numpy as np

from typing import Mapping, List, Callable

from bernet.contracts import ISampler, Batch
from bernet.utils.statistics import lhs_1

#####################################################################################
class Sampler1D(ISampler):
    """
    Hypercube sampler implementation.
    """
    
    def __init__(
            self,
            spacing: int,
            batch_size: int,
            ratio: float,
            func: Callable[[np.ndarray], np.ndarray],
        ) -> None:
        """
        Parameters
        ----------
        num_samples : int
            Total number of samples.
        batch_size : int
            Number of samples per batch.
        ratio : float
            Number of samples used by metrics divided by num_samples.
        func : Callable[[np.ndarray], np.ndarray]
            Solution based on input
        """
        super().__init__()

        #-- Verification
        if dim >= 3:
            raise "Not implemented for dim >= 3."

        #-- Inputs
        self._spacing = spacing
        self._spacing_bc = spacing if spacing_bc is None else spacing_bc
        self._dim = dim
        self._batch_size = batch_size
        self._ratio = ratio
        self._solution = solution

        #-- Parameters
        self._train_data: List[Batch] = None
        self._metrics_data: Mapping[str, torch.Tensor] = None

        return
    
    def generate(self) -> int:
        super().generate()

        #-------------------------------------------#
        # 1. Generate random points in the domain
        #-------------------------------------------#
        #-- Number of samples
        num_samples = int(1 / spacing)

        #-- Generate points using latin hypercube sampling
        x_col = (1 - spacing) * lhs_1(n=num_samples - 2) + 0.5 * spacing
        x_bc = np.array([0., 1.])
                
        #-------------------------------------------#
        # 2. Split training and testing data
        #-------------------------------------------#
        #-- Compute the permutation of all sample indexes
        indexes = np.random.permutation(num_samples)

        #-- Compute the number of training samples
        n = int((1 - ratio) * num_samples)

        #-- Create training and testing samples
        x_train = x_col[indexes[:n]]
        x_metrics = x_col[indexes[n:]]

        #-- Evaluate function at boundary and testing points
        y_bc = solution(x_bc)
        y_metrics = solution(x_metrics)

        #-------------------------------------------#
        # 3. Create training and testing data
        #-------------------------------------------#
        #-- Create testing data
        metrics_data = {
            "x": torch.from_numpy(x_metrics).view(-1, 1),
            "y": torch.from_numpy(y_metrics).view(-1, 1),
        }

        #-- Create training data
        #-- Number of batches
        num_batches = int(x_train.size / batch_size)

        #-- Create list with batches
        train_data: List[Batch] = []

        for i in range(num_batches):
            batch = Batch(
                residual={
                    "x": torch.from_numpy(x_train[i * batch_size:(i + 1) * batch_size]).view(-1, 1),
                },
                boundary={
                    "x": torch.from_numpy(x_bc).view(-1, 1),
                    "y": torch.from_numpy(y_bc).view(-1, 1),
                },
            )
            train_data.append(batch)

        return len(self._train_data)
    
    def batch(self, index: int) -> Batch:
        super().batch(index)
        return self._train_data[index]
    
    def metrics(self) -> Mapping[str, torch.Tensor]:
        super().metrics()
        return self._metrics_data

#####################################################################################