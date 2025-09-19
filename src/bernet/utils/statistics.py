#####################################################################################
from typing import List

#####################################################################################
class Statistics:
    """
    Implement statistical utilities.
    """
    
    @staticmethod
    def simple_moving_average(values: List[float]) -> float:
        """
        Compute the simple moving average (SMA) of the input values.
        
        Parameters:
            values (List[float]): Input list of floats.
        
        Returns:
            float: Simple average of the values.
        """
        return sum(values) / len(values)

    @staticmethod
    def exponential_moving_average(values: List[float], alpha: float = 0.2) -> float:
        """
        Compute the exponential moving average (EMA) of the input values.
        
        Parameters:
            values (List[float]): Input list of floats.
            alpha (float): Smoothing factor in (0, 1].
        
        Returns:
            float: Exponential moving average of the values.
        """
        ema = values[0]
        for v in values[1:]:
            ema = alpha * v + (1 - alpha) * ema
        return ema

#####################################################################################