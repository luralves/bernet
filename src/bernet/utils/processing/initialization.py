#####################################################################################
import torch

from bernet.interface.typing.aliases import Layer

#####################################################################################
class Initialization:
    """
    Common weight initialization strategies for torch.nn modules.
    Use these as static methods for layer-wise initialization.
    """

    @staticmethod
    def xavier_uniform(layer: Layer) -> None:
        """
        Xavier/Glorot uniform initialization.

        Parameters
        ----------
        layer : Layer
            Model layer.

        Notes
        -----
        Recommended for: tanh, sigmoid, or linear activations.
        Balances variance of activations across layers.
        """
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)
        return

    @staticmethod
    def xavier_normal(layer: Layer) -> None:
        """
        Xavier/Glorot normal initialization.

        Parameters
        ----------
        layer : Layer
            Model layer.
        
        Notes
        -----
        Recommended for: tanh, sigmoid, or linear activations.
        Uses a normal distribution for initializing weights.
        """
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)
        return

    @staticmethod
    def kaiming_uniform(layer: Layer, nonlinearity: str = 'relu') -> None:
        """
        Kaiming/He uniform initialization.

        Parameters
        ----------
        layer : Layer
            Model layer.
        nonlinearity : str
            Type of non-linear activation ('relu', 'leaky_relu', etc.)

        Notes
        -----
        Recommended for: ReLU and leaky ReLU activations.
        Helps preserve variance in forward/backward passes.
        """
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)

    @staticmethod
    def kaiming_normal(layer: Layer, nonlinearity: str = 'relu') -> None:
        """
        Kaiming/He normal initialization.

        Parameters
        ----------
        layer : Layer
            Model layer.
        nonlinearity : str
            Type of non-linear activation ('relu', 'leaky_relu', etc.)

        Notes
        -----
        Recommended for: ReLU and leaky ReLU activations.
        Uses a normal distribution tailored to ReLU-family activations.
        """
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(layer.weight, nonlinearity=nonlinearity)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)
        return

    @staticmethod
    def orthogonal(layer: Layer) -> None:
        """
        Orthogonal initialization.

        Parameters
        ----------
        layer : Layer
            Model layer.
        
        Notes
        -----
        Recommended for: RNNs and very deep networks.
        Maintains gradient norms and avoids weight correlation.
        """
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)
        return

#####################################################################################