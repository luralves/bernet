#####################################################################################
import torch

#####################################################################################
class Losses:
    """
    Losses for continuous outputs.
    """

    @staticmethod
    def mse(y_hat: torch.Tensor, y_ref: torch.Tensor) -> torch.Tensor:
        """
        Mean Squared Error.
        
        Parameters
        ----------
        y_hat : torch.Tensor
            Predicted value.
        y_ref : torch.Tensor
            Reference value.
        
        Returns
        -------
        torch.Tensor
            Loss value
        """
        return torch.mean((y_hat - y_ref) ** 2)

    @staticmethod
    def mae(y_hat: torch.Tensor, y_ref: torch.Tensor) -> torch.Tensor:
        """
        Mean Absolute Error.
        
        Parameters
        ----------
        y_hat : torch.Tensor
            Predicted value.
        y_ref : torch.Tensor
            Reference value.
        
        Returns
        -------
        torch.Tensor
            Loss value
        """
        return torch.mean((y_hat - y_ref).abs())
    
    @staticmethod
    def log_cosh(y_hat: torch.Tensor, y_ref: torch.Tensor) -> torch.Tensor:
        """
        Log(cosh(x)) error.
        
        Parameters
        ----------
        y_hat : torch.Tensor
            Predicted value.
        y_ref : torch.Tensor
            Reference value.
        
        Returns
        -------
        torch.Tensor
            Loss value
        """
        return torch.mean(torch.log(torch.cosh(y_hat - y_ref)))
    
    @staticmethod
    def mape(y_hat: torch.Tensor, y_ref: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
        """
        Mean Absolute Percentage Error.
        
        Parameters
        ----------
        y_hat : torch.Tensor
            Predicted value.
        y_ref : torch.Tensor
            Reference value.
        
        Returns
        -------
        torch.Tensor
            Loss value
        """
        return torch.mean((y_hat - y_ref).abs() / y_ref.abs().clamp_min(min=eps))

    @staticmethod
    def smape(y_hat: torch.Tensor, y_ref: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
        """
        Simmetric Mean Absolute Percentage Error.
        
        Parameters
        ----------
        y_hat : torch.Tensor
            Predicted value.
        y_ref : torch.Tensor
            Reference value.
        
        Returns
        -------
        torch.Tensor
            Loss value
        """
        return 2.0 * torch.mean((y_hat - y_ref).abs() / (y_hat.abs() + y_ref.abs()).clamp_min(min=eps))

    @staticmethod
    def mspe(y_hat: torch.Tensor, y_ref: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
        """
        Mean Squared Percentage Error.
        
        Parameters
        ----------
        y_hat : torch.Tensor
            Predicted value.
        y_ref : torch.Tensor
            Reference value.
        
        Returns
        -------
        torch.Tensor
            Loss value
        """
        return torch.mean((y_hat - y_ref) ** 2 / (y_ref ** 2).clamp_min(min=eps))

    @staticmethod
    def smspe(y_hat: torch.Tensor, y_ref: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
        """
        Simmetric Mean Squared Percentage Error.
        
        Parameters
        ----------
        y_hat : torch.Tensor
            Predicted value.
        y_ref : torch.Tensor
            Reference value.
        
        Returns
        -------
        torch.Tensor
            Loss value
        """
        return 4.0 * torch.mean((y_hat - y_ref) ** 2 / ((y_hat ** 2) + (y_ref ** 2)).clamp_min(min=eps))

#####################################################################################