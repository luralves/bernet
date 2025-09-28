#####################################################################################
from typing import Literal, List

from bernet.contracts.early_stop import IEarlyStop
from bernet.contracts.loss import Losses
from bernet.utils.statistics import ema, sma

#####################################################################################
def _criterion(
        vector: List[float],
        mean: str,
        window: int,
        max_error: float,
        tol: float,
    ) -> bool:
    """
    Apply the criterion to a vector.

    Parameters
    ----------
    vector : List[float]
        Reference vector with len = 2 * window
    mean : str
        Mean type - simple or exponential moving average.
    window : int
        Number of epochs considered to compute the mean.
    max_error : float
        Maximum relative error.
    tol : float
        Zero approximation.
    """

    #-- Separate vector in two
    v1 = []
    v2 = []

    for i in range(window):
        v1.append(vector[i])
        v2.append(vector[i + window])
    
    #-- Compute mean
    mean_1 = sma(v1) if mean == "sma" else ema(v1)
    mean_2 = sma(v2) if mean == "sma" else ema(v2)

    #-- Compare
    check = abs(mean_2 - mean_1) / (abs(mean_2) + tol) <= max_error

    return check

class DFLTEarlyStop(IEarlyStop):

    def __init__(
            self,
            mean: Literal["sma", "ema"] = "sma",
            window: int = 10,
            max_error: float = 1e-2,
            patience: int = 50,
            tol: float = 1e-6,
        ):
        """
        Parameters
        ----------
        mean : str
            Mean type - simple or exponential moving average.
        window : int
            Number of epochs considered to compute the mean.
        max_error : float
            Maximum relative error.
        patience : int
            Number of initial epoch with no computation.
        tol : float
            Zero approximation.
        """
        super().__init__()

        #-- Inputs
        self.mean = mean
        self.window = window
        self.max_error = max_error
        self.patience = patience
        self.tol = tol

        #-- Internal parameters
        self.residual = [.0 for _ in range(2 * self.window)]
        self.boundary = [.0 for _ in range(2 * self.window)]
        self.initial = [.0 for _ in range(2 * self.window)]
        self.observational = [.0 for _ in range(2 * self.window)]

        self.epoch = 0
        self.index = 0

        return
    
    def evaluate(
            self,
            losses: Losses,
            _,
        ):
        super().evaluate(losses, None)

        #-- Initialize checks
        check_1 = False
        check_2 = False
        check_3 = False
        check_4 = False

        #-- Verify patient
        if self.epoch > self.patience:

            #-- Insert value in v
            self.residual[self.index] = losses.residual
            self.boundary[self.index] = losses.boundary
            self.initial[self.index] = losses.initial
            self.observational[self.index] = losses.observational

            #-- Update index value
            self.index += 1

            #-- Verify index limit
            if self.index == 2 * self.window:

                #-- Verify stop criterion
                check_1 = _criterion(vector=self.residual, mean=self.mean, window=self.window, max_error=self.max_error, tol=self.tol) if not (None in self.residual) else True
                check_2 = _criterion(vector=self.boundary, mean=self.mean, window=self.window, max_error=self.max_error, tol=self.tol) if not (None in self.boundary) else True
                check_3 = _criterion(vector=self.initial, mean=self.mean, window=self.window, max_error=self.max_error, tol=self.tol) if not (None in self.initial) else True
                check_4 = _criterion(vector=self.observational, mean=self.mean, window=self.window, max_error=self.max_error, tol=self.tol) if not (None in self.observational) else True
                
                #-- Move one window
                for i in range(self.window):
                    self.residual[i] = self.residual[i + self.window]
                    self.boundary[i] = self.boundary[i + self.window]
                    self.initial[i] = self.initial[i + self.window]
                    self.observational[i] = self.observational[i + self.window]
                
                self.index = self.window

        #-- Add epoch
        self.epoch += 1

        return check_1 and check_2 and check_3 and check_4