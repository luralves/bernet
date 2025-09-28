#####################################################################################
from typing import List

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
#####################################################################################