#####################################################################################
import numpy as np
import torch
import math

from typing import Literal, Optional, List, Mapping, Callable, Union, get_args

from bernet.interface.abstract.sampler import ISampler
from bernet.interface.typing.aliases import TensorData
from bernet.interface.typing.structs import Batch
from bernet.utils.statistics import lhs_1, lhs_d
from bernet.utils.validation import TypeCheck, ValueCheck

#####################################################################################
#--
Side = Literal["top", "bottom", "left", "right"]

#--
class Square(ISampler):
    """
    Sampler for PINNs on the unit square [0,1]^2.

    Attributes
    ----------
    _n_it : int
        Number of interior points.
    _n_bc : int
        Number of boudanry points per side.
    _n_in : int
        Number of initial condition points.
    _n_ob : int
        Number of observational points.
    _n_mt : int
        Number of metrics points.
    _n_rs : int
        Number of residual points.
    _ratio : List[float]
        Ratio between the number of points normalize by its maximum value.
    _batches : List[Batch]
        List with batches.
    _metrics : TensorData
        Metrics data.

    Methods
    -------
    generate() -> int
        Generates residual and boundary points and prepares batches.
    batch(index: int) -> Batch
        Returns a Batch object with residual, boundary, and observational
        points for the given batch index.
    test() -> TensorData
        Returns the test (metrics) dataset.
    validate() -> TensorData
        Returns the validation (metrics) dataset.

    Notes
    -----
    - Residual (interior) points via LHS in 2D using ``lhs_d``.
    - Boundary points on all four edges using 1D LHS ``lhs_1``.
    - Independent test (metrics) set via LHS.
    - Batching with reshuffle or not each epoch via ``generate``.
    """
    
    def __init__(
            self,
            interior_h: float,
            boundary_h: Optional[float] = None,
            *,
            initial_h: Optional[float] = None,
            ratio_ob: Optional[float] = .1,
            ratio_mt: Optional[float] = .1,
            padding: Optional[float] = .01,
            batch_size: Optional[int] = 50,
            include_corners: bool = True,
            keep_ratio: bool = True,
            include_name: bool = True,
            resample: bool = True,
            remove: List[Side] = None,
            func: Callable[[np.ndarray], np.ndarray] = lambda x: np.zeros(shape=[x.shape[0], 1]),
            dtype: torch.dtype = torch.float32,
        ) -> None:
        """
        Parameters
        ----------
        interior_h : float
            Target mean spacing for interior residual points.
        boundary_h : Optional[float]
            Target mean spacing along boundary edges.
        initial_h : Optional[float]
            Target mean spacing along initial condition edge.
        ratio_ob : Optional[float]
            Ratio between the observational data and interior data.
        ratio_mt : Optional[float]
            Ratio between the metrics data and interior data.
        padding : Optional[float]
            Spacing between the internal points and boundary points.
        batch_size : Optional[int]
            Batch size for training.
        include_corners : bool
            If True, ensures the four corners of the unit square are included
            in the boundary set (if boundary sampling is enabled).
        keep_ratio : bool
            If True, keep the ratio of number of points in each batch.
        include_name : bool
            If True, return a Dict as output. If False, return Tensor or a list
            of Tensors when necessary.
        resample : bool
            If True, resample every time generate is called.
        remove : List[Side]
            Remove the corresponding sides from the boundary.
        func : Callable[[np.ndarray], np.ndarray]
            Return the true solution for boundary, observational, and metrics points.
        dtype : torch.dtype
            Data type for torch tensors.
        """
        super().__init__()

        #-- Type check
        TypeCheck.float(interior_h)
        TypeCheck.float(boundary_h, include_none=True)
        TypeCheck.float(initial_h, include_none=True)
        TypeCheck.float(ratio_ob, include_none=True)
        TypeCheck.float(ratio_mt, include_none=True)
        TypeCheck.int(batch_size, include_none=True)
        TypeCheck.bool(include_corners)
        TypeCheck.bool(keep_ratio)
        TypeCheck.bool(include_name)
        TypeCheck.bool(resample)
        TypeCheck.callable(func)
        TypeCheck.generic(dtype, target=[torch.dtype])
        
        if remove is not None:
            TypeCheck.sequence(remove)
            for side in remove:
                ValueCheck.on_iterable(side, get_args(Side))

        #-- Inputs
        self.interior_h = interior_h
        self.boundary_h = boundary_h
        self.initial_h = initial_h
        self.ratio_ob = ratio_ob
        self.ratio_mt = ratio_mt
        self.padding = padding if padding is not None else 0.0
        self.batch_size = batch_size
        self.include_corners = include_corners
        self.keep_ratio = keep_ratio
        self.include_name = include_name
        self.resample = resample
        self.remove = remove if remove is not None else []
        self.func = func
        self.dtype = dtype

        #-- Number of points
        self.n_it = int(math.ceil(1.0 / (self.interior_h ** 2)))                                      # Number of interior points
        self.n_bc = int(math.ceil(1.0 / (self.boundary_h))) if self.boundary_h is not None else 0     # Number of boudanry points per side
        self.n_in = int(math.ceil(1.0 / (self.initial_h))) if self.initial_h is not None else 0       # Number of initial condition points
        self.n_ob = int(math.floor(self.ratio_ob * self.n_it)) if self.ratio_ob is not None else 0    # Number of observational points
        self.n_mt = int(math.floor(self.ratio_mt * self.n_it)) if self.ratio_mt is not None else 0    # Number of metrics points
        self.n_rs = self.n_it - self.n_ob - self.n_mt                                                 # Number of residual points

        #-- Ratio between different samples
        if self.keep_ratio:
            ns = np.array([self.n_rs, self.n_bc, self.n_in, self.n_ob])
            ns_arg_max = np.argmax(ns)
            self.ratio = ns / ns[ns_arg_max]
        else:
            self.ratio = np.ones(4)
        
        #-- Output
        self.batches: List[Batch] = None
        self.metrics: TensorData = None

        return
    #----------------------------------------------------------------#
    #-- Override
    def generate(self, device: torch.device) -> int:
        if self.batches is None or self.resample:
            super().generate()
            x_rs, x_bc, x_in, x_ob, x_mt = self._generate_points()
            num_batches, indexes_rs, indexes_bc, indexes_in, indexes_ob = self._generate_indexes(x_rs, x_bc, x_in, x_ob)
            self._generate_train_test(num_batches, x_rs, x_bc, x_in, x_ob, x_mt, indexes_rs, indexes_bc, indexes_in, indexes_ob, device)
        return len(self.batches)
    
    #-- Override
    def batch(self, index: int) -> Batch:
        super().batch(index)
        return self.batches[index]
    
    #-- Override
    def test(self) -> TensorData:
        super().test()
        return self.metrics
    
    #-- Override
    def validate(self) -> None:
        super().validate()
        return None
    
    #----------------------------------------------------------------#
    #-- Auxiliary
    def _generate_points(self) -> List[np.ndarray]:
        """Generate points"""

        #-- Resample interior points
        x_it = (1 - self.padding) * lhs_d(n=self.n_it, d=2) + 0.5 * self.padding   # Interior points

        #-- Separate observational, metric, and residual data
        ids_it = np.random.permutation(self.n_it)                                              # Interior indexes

        x_mt = x_it[ids_it[:self.n_mt], :] if self.n_mt != 0 else None                        # Metrics points
        x_ob = x_it[ids_it[self.n_mt:self.n_mt + self.n_ob], :] if self.n_ob != 0 else None # Observational points
        x_rs = x_it[ids_it[self.n_mt + self.n_ob:], :]                                        # Residual points

        #-- Resample boundary points
        if self.n_bc != 0:
            
            #-- Sides
            sides = []
            if "bottom" not in self.remove:
                sides.append(np.vstack([lhs_1(n=self.n_in if self.n_in != 0 else self.n_bc), np.zeros(self.n_in if self.n_in != 0 else self.n_bc)]).T)
            
            if "top" not in self.remove:
                sides.append(np.vstack([lhs_1(n=self.n_bc), np.ones(self.n_bc)]).T)

            if "left" not in self.remove:
                sides.append(np.vstack([np.zeros(self.n_bc), lhs_1(n=self.n_bc)]).T)
            
            if "right" not in self.remove:
                sides.append(np.vstack([np.ones(self.n_bc), lhs_1(n=self.n_bc)]).T)

            #-- Create boundary points
            if self.n_in != 0:
                x_in = sides.pop(0)
                x_bc = np.concat(sides, axis=0)   # Boundary points
            else:
                x_in = None
                x_bc = np.concat(sides, axis=0)   # Boundary points

            #-- Include corners
            if self.include_corners:
                if self.n_in != 0:
                    x_bc = np.concat([x_bc, np.array([[0.0,1.0],[1.0,1.0]])], axis=0)
                else:
                    x_bc = np.concat([x_bc, np.array([[0.0,0.0],[1.0,0.0],[0.0,1.0],[1.0,1.0]])], axis=0)

            #-- Unique points
            x_bc = np.unique(x_bc, axis=0)

        else:
            
            #-- Define x_bc equal None
            x_bc = None

            #-- Create initial condition points if necessary
            if self.n_in != 0:
                x_in = np.vstack([lhs_1(n=self.n_in), np.zeros(self.n_in)]).T
            else:
                x_in = None

        return [x_rs, x_bc, x_in, x_ob, x_mt]
    
    def _generate_indexes(
            self,
            x_rs: np.ndarray,
            x_bc: Optional[np.ndarray],
            x_in: Optional[np.ndarray],
            x_ob: Optional[np.ndarray],
        ) -> List[Union[int, np.ndarray] | None]:

        #-- Initial check
        if self.batch_size is None:
            return [None, None, None, None, None]
        
        #-- Auxiliary function
        def _random_indexes(size: int, n_max: int) -> np.ndarray:
            """Generate a random list of indexes"""
            if size < n_max:
                indexes = np.concat([
                    np.random.permutation(size),
                    np.random.choice(np.arange(size), n_max - size),
                ], axis=0) if size < n_max else np.random.permutation(size)
            else:
                indexes = np.random.permutation(size)
            return indexes

        #-- Number of batches
        num_batches = int(max([
            math.ceil(x_rs.shape[0] / self.batch_size),                                    # Residuals
            -1 if x_bc is None else math.ceil(x_bc.shape[0] / self.batch_size),      # boundary
            -1 if x_in is None else math.ceil(x_in.shape[0] / self.batch_size),      # Initial
            -1 if x_ob is None else math.ceil(x_ob.shape[0] / self.batch_size),      # Observational
        ]))

        #-- Number of points
        n_max = num_batches * self.batch_size

        #-- Create indexes
        indexes_rs = _random_indexes(size=x_rs.shape[0], n_max=n_max)
        indexes_bc = _random_indexes(size=x_bc.shape[0], n_max=n_max) if x_bc is not None else None
        indexes_in = _random_indexes(size=x_in.shape[0], n_max=n_max) if x_in is not None else None
        indexes_ob = _random_indexes(size=x_ob.shape[0], n_max=n_max) if x_ob is not None else None

        return [num_batches, indexes_rs, indexes_bc, indexes_in, indexes_ob]

    def _generate_train_test(
            self,
            num_batches: int,
            x_rs: np.ndarray,
            x_bc: Optional[np.ndarray],
            x_in: Optional[np.ndarray],
            x_ob: Optional[np.ndarray],
            x_mt: Optional[np.ndarray],
            indexes_rs: np.ndarray,
            indexes_bc: Optional[np.ndarray],
            indexes_in: Optional[np.ndarray],
            indexes_ob: Optional[np.ndarray],
            device: torch.device,
        ) -> None:

        #-- Auxiliary function
        def _apply_ratio(data: Mapping[str, np.ndarray] | None, ratio: float) -> Mapping[str, np.ndarray]:
            """Sample from content"""

            #-- Check data
            if data is None:
                return None

            #-- Compute new size
            n = int(max(1, math.floor(ratio * data[next(iter(data))].shape[0])))

            #-- Modify data
            for k, v in data.items():
                data[k] = v[:n, :]

            return data
        
        def _numpy_to_tensor(data: Mapping[str, np.ndarray] | None) -> Mapping[str, torch.Tensor]:
            """Convert numpy array to tensor"""

            #-- Check data
            if data is None:
                return None
            
            #-- Convert to tensor
            for k, v in data.items():
                data[k] = torch.from_numpy(v).to(dtype=self.dtype, device=device)
            
            return data

        #-- Metrics
        metrics = {
            "x": torch.from_numpy(x_mt).to(dtype=self.dtype, device=device),
            "y": torch.from_numpy(self.func(x_mt)).to(dtype=self.dtype, device=device),
        }
        
        self.metrics = metrics if self.include_name else [metrics["x"], metrics["y"]]

        #-- Batches
        if self.batch_size is None:
            
            #-- Create numpy structure
            residual = {"x": x_rs}
            boundary = {"x": x_bc, "y": self.func(x_bc)} if x_bc else None
            initial = {"x": x_in, "y": self.func(x_in)} if x_in else None
            observational = {"x": x_ob, "y": self.func(x_ob)} if x_in else None

            #-- Convert to torch tensor
            residual = _numpy_to_tensor(residual)
            boundary = _numpy_to_tensor(boundary)
            initial = _numpy_to_tensor(initial)
            observational = _numpy_to_tensor(observational)

            self.batches = [
                Batch(
                    residual=residual if self.include_name or residual is None else residual["x"],
                    boundary=boundary if self.include_name or boundary is None else (boundary["x"], boundary["y"]),
                    initial=initial if self.include_name or initial is None else (initial["x"], initial["y"]),
                    observational=observational if self.include_name or observational is None else (observational["x"], observational["y"]),
                ),
            ]
        
        else:

            self.batches = []
            for i in range(num_batches):
                
                #-- Create numpy mapping
                residual = {"x": x_rs[indexes_rs[i * self.batch_size:(i + 1) * self.batch_size], :],}
                boundary = {"x": x_bc[indexes_bc[i * self.batch_size:(i + 1) * self.batch_size], :], "y": self.func(x_bc[indexes_bc[i * self.batch_size:(i + 1) * self.batch_size], :]),} if x_bc is not None else None
                initial = {"x": x_in[indexes_in[i * self.batch_size:(i + 1) * self.batch_size], :], "y": self.func(x_in[indexes_in[i * self.batch_size:(i + 1) * self.batch_size], :]),} if x_in is not None else None
                observational = {"x": x_ob[indexes_ob[i * self.batch_size:(i + 1) * self.batch_size], :], "y": self.func(x_ob[indexes_ob[i * self.batch_size:(i + 1) * self.batch_size], :]),} if x_ob is not None else None
                
                #-- Scale if necessary
                if self.keep_ratio:
                    residual = _apply_ratio(residual, self.ratio[0])
                    boundary = _apply_ratio(boundary, self.ratio[1])
                    initial = _apply_ratio(initial, self.ratio[2])
                    observational = _apply_ratio(observational, self.ratio[3])
                
                #-- Convert to tensor
                residual = _numpy_to_tensor(residual)
                boundary = _numpy_to_tensor(boundary)
                initial = _numpy_to_tensor(initial)
                observational = _numpy_to_tensor(observational)

                self.batches.append(
                    Batch(
                        residual=residual if self.include_name or residual is None else residual["x"],
                        boundary=boundary if self.include_name or boundary is None else (boundary["x"], boundary["y"]),
                        initial=initial if self.include_name or initial is None else (initial["x"], initial["y"]),
                        observational=observational if self.include_name or observational is None else (observational["x"], observational["y"]),
                    ),
                )

        return

#####################################################################################