#####################################################################################
import numpy as np

from typing import Callable, Optional, List

from bernet.interface.typing.aliases import NDArray

#####################################################################################
#--
def lhs_1(
        n: int,
        func: Optional[Callable[[NDArray | float], NDArray | float]] = None,
        tol: float = 1e-6,
    ) -> NDArray:
    """
    Latin Hypercube Sampling for unidimensional variables.

    Parameters
    ----------
    n : int
        Number of points.
    func : Callable[[np.ndarray | float], np.ndarray | float]
        Quantile function (inverse of cumulative distribution function). If func is None, func(x) = x.
    tol: float
        Zero approximation.
    
    Returns
    -------
    NDArray
        Sample.
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
        funcs: List[Optional[Callable[[NDArray | float], NDArray | float]]] = None,
        tol: float = 1e-6,
    ) -> NDArray:
    """
    Latin Hypercube Sampling for multidimensional variables.

    Parameters:
    ----------
    n : int
        Number of points.
    d : int
        Output dimension.
    funcs : Callable[[np.ndarray | float], np.ndarray | float]
        List of 1uantile functions (inverse of cumulative distribution function). If func is None, func(x) = x.
    tol: float
        Zero approximation.

    Returns
    -------
    NDArray
        Sample.
    """
    
    #-- Define the output
    x = np.empty(shape=[n, d])

    #-- Compute the columns of x
    for j in range(d):

        xj = lhs_1(n=n, func=None if funcs is None else funcs[j], tol=tol)
        x[:, j] = xj

    return x

#####################################################################################