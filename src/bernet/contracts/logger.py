#####################################################################################
from typing import Mapping, Optional
from abc import ABC
from csv import DictWriter, DictReader

#####################################################################################
class LoggerBASE(ABC):
    """
    Abstract base class for loggers.

    > Necessary parameters:
    - data: Mapping[str, list[float]]
    """
    data: Mapping[str, list[float]] = {}


    def start(self) -> None:
        """
        Called at the start of training.
        """
        ...

    def epoch(
        self,
        epoch: int,
        total_loss: float,
        loss_terms: Mapping[str, float],
        metrics: Optional[Mapping[str, float]],
        ) -> None:
        """
        Called at the end of each epoch.
        """
        
        # Save total_loss
        if "total_loss" not in self.data:
            self.data["total_loss"] = []
        self.data["total_loss"].append(total_loss)

        # Save loss_terms
        for key, value in loss_terms.items():
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)

        # Save metrics if provided
        if metrics:
            for key, value in metrics.items():
                if key not in self.data:
                    self.data[key] = []
                self.data[key].append(value)
        
        return

    def stopped(self) -> None:
        """
        Called when training is stopped early.
        """
        ...

    def exception(
        self,
        e: BaseException,
        ) -> None:
        """
        Called when an exception occurs.
        """
        ...

    def close(self) -> None:
        """
        Called at the end of training.
        """
        ...
    
    def save(self, filename: str) -> None:
        """
        Save the logged data to a file.

        Parameters:
        ----------
        - filename: str
          > The name of the file to save the data to.
        """
        
        # Get all unique keys (columns)
        columns = list(self.data.keys())

        # Find the maximum length among all columns
        max_len = max(len(v) for v in self.data.values())

        # Prepare rows
        rows = []
        for i in range(max_len):
            row = {}
            for col in columns:
                values = self.data[col]
                row[col] = values[i] if i < len(values) else ""
            rows.append(row)

        # Write to CSV file
        with open(filename, "w", newline="") as f:
            writer = DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

        return

    def load(self, filename: str) -> None:
        """
        Load the logged data from a file saved by the save() method.

        Parameters:
        ----------
        - filename: str
          > The name of the file to load the data from.
        """

        with open(filename, "r", newline="") as f:

            # Read from CSV file
            reader = DictReader(f)

            # Initialize data as a dict of lists
            data = {col: [] for col in reader.fieldnames}

            for row in reader:

                for col in reader.fieldnames:
                    value = row[col]

                    # Try to convert to float if possible, else keep as string
                    if value == "":
                        data[col].append(None)
                    else:
                        try:
                            data[col].append(float(value))
                        except ValueError:
                            data[col].append(value)
            
            self.data = data

        return
    
#####################################################################################