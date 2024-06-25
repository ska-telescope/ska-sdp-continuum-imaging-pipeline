import os
from pathlib import Path
from typing import Union


class MeasurementSet:
    """
    Represents a CASA MeasurementSet v2.
    """

    def __init__(self, path: Union[str, os.PathLike]) -> None:
        self._path = Path(path).resolve()
        if not self.path.is_dir():
            msg = (
                "Cannot initialise MeasurementSet: path is not a directory: "
                f"{self.path}"
            )
            raise FileNotFoundError(msg)

    @property
    def path(self) -> Path:
        """
        Absolute path on disk.
        """
        return self._path
