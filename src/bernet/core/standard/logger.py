#####################################################################################
import torch
import os
import json
import math

from typing import Mapping, Any, List

from bernet.contracts.logger import ILogger
from bernet.contracts.loss import BatchLoss

#####################################################################################
#-- Upload data
def load_training_data(
        filename: str,
    ) -> Mapping[str, List[float]]:
    """
    Reads the train_data table saved by DFLTLogger.save(...) and returns
    a dictionary mapping column name -> list of float values.

    Parameters
    ----------
    filename : str
        Path to the .log file saved by DFLTLogger.
    col_width : int
        Column width used in save(). Must match the one used when writing.

    Returns
    -------
    Dict[str, List[float]]
        Dictionary with column names as keys and lists of floats as values.
    """
    BEGIN_MARKER = "## BEGIN TRAIN DATA"
    END_MARKER = "## END TRAIN DATA"
    col_width = 16

    with open(filename, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    #-- Find table boundaries
    try:
        start_idx = lines.index(BEGIN_MARKER)
        end_idx   = lines.index(END_MARKER)
    except ValueError:
        raise RuntimeError("Could not find train_data markers in log file.")

    #-- Header line
    header_line = lines[start_idx + 1]
    ncols = len(header_line) // col_width
    colnames = [header_line[i * col_width:(i+1) * col_width].strip()
                for i in range(ncols)]

    #-- Initialize dict
    data: Mapping[str, List[float]] = {c: [] for c in colnames}

    #-- Parse rows
    for row in lines[start_idx + 2 : end_idx]:
        for i, cname in enumerate(colnames):
            cell = row[i * col_width:(i+1) * col_width].strip()
            try:
                val = float(cell)
            except Exception:
                val = math.nan
            data[cname].append(val)

    return data

#-- Main class
class Logger(ILogger):
    """
    Default Logger class.
    """

    def __init__(self):
        super().__init__()
        self.data = {}
        return
    
    def train_start(
            self,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
        ) -> None:
        super().train_start(model, optimizer)

        # Model info
        self.data['model_class'] = model.__class__.__name__
        self.data['model_module'] = model.__module__
        self.data['model_str'] = str(model)
        self.data['num_parameters'] = sum(p.numel() for p in model.parameters())
        self.data['num_trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Optimizer info
        self.data['optimizer_class'] = optimizer.__class__.__name__
        self.data['optimizer_module'] = optimizer.__module__
        self.data['optimizer_params'] = optimizer.defaults

        return
    
    def epoch_end(
            self,
            losses: BatchLoss,
            tests: Mapping[str, float],
        ) -> None:
        super().epoch_end(losses, tests)

        #-- Create trianing data
        if not ("train_data" in self.data):
            self.data["train_data"] = {"iteration": [], "loss": [], "residual": [], "boundary": [], "initial": [], "observational": []}
            
            if tests:
                for k, v in tests.items():
                    self.data["train_data"][k] = []
        
        #-- Add data
        self.data["train_data"]["iteration"].append(len(self.data["train_data"]["iteration"]) + 1)
        self.data["train_data"]["loss"].append(losses.sum())
        self.data["train_data"]["residual"].append(losses.residual)
        self.data["train_data"]["boundary"].append(losses.boundary)
        self.data["train_data"]["initial"].append(losses.initial)
        self.data["train_data"]["observational"].append(losses.observational)

        if tests:
            for k, v in tests.items():
                self.data["train_data"][k + " [test]"].append(v)

        return
    
    def exception(self, e) -> None:
        super().exception(e)
        #-- Add exception
        self.data["exception"] = str(e)
        return
    
    def training_end(self, stopped: bool) -> None:
        super().training_end(stopped)
        #-- Add training end
        self.data["stopped"] = stopped
        return
    
    def save(self, filename: str) -> None:
        super().save(filename)

        #-- Configurable formatting knobs
        #-  Total width per column (incl. sign, decimal, exponent)
        COL_WIDTH = 16
        #-  Digits after decimal, e.g. 1.234567e+03
        PRECISION = 6
        BEGIN_MARKER = "## BEGIN TRAIN DATA"
        END_MARKER = "## END TRAIN DATA"

        #-- Ensure .data extension
        if not filename.lower().endswith(".data"):
            filename = f"{filename}.data"

        #-- Make sure parent folder exists
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

        #-- Split metadata vs. table
        train_data = self.data.get("train_data", None)
        meta = {k: v for k, v in self.data.items() if k != "train_data"}

        def _to_serializable(obj: Any) -> Any:
            """Make meta JSON-safe."""
            try:
                json.dumps(obj)
                return obj
            except Exception:
                return str(obj)

        #-- Build file content
        lines: List[str] = []
        lines.append("# === Training Log ===")
        lines.append("")

        #-  Metadata section (pretty JSON)
        lines.append("## METADATA (JSON)")
        lines.append(json.dumps({k: _to_serializable(v) for k, v in meta.items()},
                                indent=2, sort_keys=True))
        lines.append("")

        #-  Table section (only if train_data exists and is non-empty)
        if train_data and isinstance(train_data, dict) and any(isinstance(v, list) and len(v) for v in train_data.values()):
            #-  Determine columns (preserve insertion order of keys)
            columns = list(train_data.keys())

            #-  Determine number of rows (max length across columns)
            nrows = max(len(train_data.get(c, [])) for c in columns)

            #-  Header builder (fixed width, right-align names to match numbers)
            def _hdr_cell(name: str) -> str:
                name = str(name)
                # truncate if longer than width
                if len(name) > COL_WIDTH:
                    name = name[:COL_WIDTH]
                return f"{name:>{COL_WIDTH}s}"

            #-  Data cell builder (scientific notation)
            def _num_cell(val: Any) -> str:
                if val is None:
                    return f"{'nan':>{COL_WIDTH}s}"
                #-  Attempt float conversion
                try:
                    x = float(val)
                    return f"{x:{COL_WIDTH}.{PRECISION}e}"
                except Exception:
                    #-  Fallback for non-numeric: store as string (truncated)
                    s = str(val)
                    if len(s) > COL_WIDTH:
                        s = s[:COL_WIDTH]
                    return f"{s:>{COL_WIDTH}s}"

            #-  Write markers + header
            lines.append(BEGIN_MARKER)
            header = "".join(_hdr_cell(c) for c in columns)
            lines.append(header)

            #-  Rows
            for i in range(nrows):
                row_cells = []
                for c in columns:
                    col = train_data.get(c, [])
                    v = col[i] if i < len(col) else math.nan
                    row_cells.append(_num_cell(v))
                lines.append("".join(row_cells))

            lines.append(END_MARKER)
        else:
            lines.append("# (No train_data recorded.)")

        lines.append("")  # final newline

        #-  Write to disk
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return

#####################################################################################