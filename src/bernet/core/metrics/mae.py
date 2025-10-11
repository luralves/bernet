#####################################################################################
from bernet.interface.abstract.metrics import IMetrics
from bernet.interface.typing.aliases import Model, Tensor, TensorData
from bernet.utils.processing.error import Error
from bernet.utils.validation.type_check import TypeCheck

#####################################################################################
class MAE(IMetrics):
    """
    Mean Absolute Error
    
    Notes
    -----
    The output of 'evaluate' depends on 'data':
    - If 'data' is of type Iterable, the output will be a tensor.
    - If 'data' is of type Mapping, the output will be a dict.
    If 'data' is of type Mapping, it will try to get the default
    parameters defined as 'x' for input, and 'y' for expected output.
    """

    #-- Override
    def evaluate(self, model: Model, data: TensorData) -> TensorData:
        super().evaluate(model, data)

        #-- 'data' type
        if TypeCheck.sequence(data, stop=False):
            
            #-- Get parameters
            x, y = data

            #-- Validation
            TypeCheck.generic(x, [Tensor])
            TypeCheck.generic(y, [Tensor])

            #-- Predict
            y_hat = model(x)

            #-- Compute loss
            mae = Error.mae(y_hat, y)

            return mae
        
        elif TypeCheck.mapping(data, stop=False):

            #-- Get parameters
            x = data.get("x", None)
            y = data.get("y", None)

            #-- Validation
            TypeCheck.generic(x, [Tensor])
            TypeCheck.generic(y, [Tensor])

            #-- Predict
            y_hat = model(x)

            #-- Compute loss
            mae = Error.mae(y_hat, y)

            return {"mae": mae}
        
        #-- Raise an error
        raise TypeError("Type Error: 'data' type error.")
        

#####################################################################################