import os
from pathlib import Path
from typing import Union

from casacore.tables import table
from numpy.typing import NDArray


class UnsupportedMeasurementSetLayout(Exception):
    """
    Exception raised when a given MeasurementSet layout deviates from what is
    expected/supported.
    """


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

        with self._open_table_readonly("SPECTRAL_WINDOW") as spw:
            if not spw.nrows() == 1:
                raise UnsupportedMeasurementSetLayout(
                    "Multiple spectral windows are not supported"
                )

    @property
    def path(self) -> Path:
        """
        Absolute path on disk.
        """
        return self._path

    def _open_table_readonly(self, table_name: str) -> table:
        return table(f"{self.path}::{table_name}", readonly=True, ack=False)

    def channel_frequencies(self) -> NDArray:
        """
        Channel frequencies in Hz as a one-dimensional numpy array.
        """
        # NOTE: assuming single spectral window
        with self._open_table_readonly("SPECTRAL_WINDOW") as tbl:
            return tbl.getcol("CHAN_FREQ")[0]
