#####################################################################################
from typing import Dict, List
import math

#####################################################################################
def dflt_load_logger_data(
        filename: str,
    ) -> Dict[str, List[float]]:
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
    BEGIN_MARKER = "<<<BEGIN_TRAIN_DATA>>>"
    END_MARKER = "<<<END_TRAIN_DATA>>>"
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
    data: Dict[str, List[float]] = {c: [] for c in colnames}

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

#####################################################################################