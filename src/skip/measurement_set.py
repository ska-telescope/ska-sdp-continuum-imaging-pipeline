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

        self._assert_layout_supported()

    def _assert_layout_supported(self) -> None:
        """
        Enforce layout restrictions.
        """
        with self._open_table_readonly("SPECTRAL_WINDOW") as tbl:
            if not tbl.nrows() == 1:
                raise UnsupportedMeasurementSetLayout(
                    "Multiple spectral windows are not supported"
                )

        with self._open_table_readonly("POLARIZATION") as tbl:
            if not tbl.nrows() == 1:
                raise UnsupportedMeasurementSetLayout(
                    "Mixed polarization rows are not supported"
                )
            corr_types = tbl.getcol("CORR_TYPE")[0]
            if not tuple(corr_types) == (9, 10, 11, 12):
                raise UnsupportedMeasurementSetLayout(
                    "Polarization channels must be XX, XY, YX, YY"
                )

    @property
    def path(self) -> Path:
        """
        Absolute path on disk.
        """
        return self._path

    def _open_table_readonly(self, table_name: str = "") -> table:
        """
        Opens casacore table with the given name; opens the MAIN table if no
        name is provided.
        """
        if not table_name or table_name == "MAIN":
            table_spec = str(self.path)
        else:
            table_spec = f"{self.path}::{table_name}"
        return table(table_spec, readonly=True, ack=False)

    def _getcol(self, table_name: str, column_name: str) -> NDArray:
        with self._open_table_readonly(table_name) as tbl:
            return tbl.getcol(column_name)

    def channel_frequencies(self) -> NDArray:
        """
        Channel frequencies in Hz as a one-dimensional numpy array.
        """
        # NOTE: assuming single spectral window
        data = self._getcol("SPECTRAL_WINDOW", "CHAN_FREQ")
        return data[0]

    def uvw(self) -> NDArray:
        """
        UVW coordinates as a numpy array with shape (nrows, 3).
        """
        return self._getcol("MAIN", "UVW")

    def visibilities(self) -> NDArray:
        """
        Visibilities as a numpy array with shape (nrows, nchan, npol).
        """
        return self._getcol("MAIN", "DATA")

    def stokes_i_visibilities(self) -> NDArray:
        """
        Stokes I visibilities as a numpy array with shape (nrows, nchan)
        """
        vis = self.visibilities()
        # NOTE: assuming XX, XY, YX, YY correlations
        return 0.5 * (vis[..., 0] + vis[..., 3])
