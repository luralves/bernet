#####################################################################################
import numpy as np

from typing import Callable, Optional, List

#####################################################################################
#--
def sma(x: List[float]) -> float:
    """
    Compute the simple moving average (SMA) of the input values.
        
    Parameters:
    - x (List[float]): Input list of floats.
        
    Returns:
    - float: Simple average of the values.
    """
    return sum(x) / len(x)

#--
def ema(
        x: List[float],
        alpha: float = 0.2,
    ) -> float:
    """
    Compute the exponential moving average (EMA) of the input values.
        
    Parameters:
    - x (List[float]): Input list of floats.
    - alpha (float): Smoothing factor in (0, 1].
        
    Returns:
    - float: Exponential moving average of the values.
    """
    ema = x[0]
    for v in x[1:]:
        ema = alpha * v + (1 - alpha) * ema
    return ema

#--
def lhs_1(
        n: int,
        func: Optional[Callable[[np.ndarray | float], np.ndarray | float]] = None,
        tol: float = 1e-6,
    ) -> np.ndarray:
    """
    Latin Hypercube Sampling for unidimensional variables.

    Parameters:
    ----------
    - n: int
    > Number of points.
    - func: Callable[[np.ndarray | float], np.ndarray | float]
    > Quantile function (inverse of cumulative distribution function). If func is None, func(x) = x.
    - tol: float
    > Zero approximation.
    """
    
    #-- Define a random generator
    rng = np.random.default_rng()

    #-- Compute a normalize distribution
    xn = (rng.permutation(n) + 1 - rng.random(n)) / n

    #-- Convert to real distribution if necessary
    if func is None:
        x = xn
    
    else:
        #-- Define lower and upper limits of x
        x_min = func(tol)
        x_max = func(1.0 - tol)

        #-- Compute the real distribution
        x = x_min + func(xn) * (x_max - x_min)

    return x

#--
def lhs_d(
        n: int,
        d: int,
        funcs: List[Optional[Callable[[np.ndarray | float], np.ndarray | float]]] = None,
        tol: float = 1e-6,
    ) -> np.ndarray:
    """
    Latin Hypercube Sampling for unidimensional variables.

    Parameters:
    ----------
    - n: int
    > Number of points.
    - d: int
    > Output dimension.
    - funcs: Callable[[np.ndarray | float], np.ndarray | float]
    > List of 1uantile functions (inverse of cumulative distribution function). If func is None, func(x) = x.
    - tol: float
    > Zero approximation.
    """
    
    #-- Define the output
    x = np.empty(shape=[n, d])

    #-- Compute the columns of x
    for j in range(d):

        xj = lhs_1(n=n, func=None if funcs is None else funcs[j], tol=tol)
        x[:, j] = xj

    return x

#####################################################################################