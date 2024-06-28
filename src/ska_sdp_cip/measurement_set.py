import os
import warnings
from pathlib import Path
from typing import Optional, Union

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

        with self._open_table_readonly("FIELD") as tbl:
            if not tbl.nrows() == 1:
                raise UnsupportedMeasurementSetLayout(
                    "Multiple fields are not supported"
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

    def flags(self) -> NDArray:
        """
        Flags as a boolean numpy array with shape (nrows, nchan, npol).
        """
        return self._getcol("MAIN", "FLAG")

    def weights(self) -> Optional[NDArray]:
        """
        Correlator weights as a numpy array with shape (nrows, nchan, npol).
        This is the contents of the WEIGHT_SPECTRUM column. If the column does
        not exist, return None.
        """
        try:
            return self._getcol("MAIN", "WEIGHT_SPECTRUM")
        except RuntimeError:
            return None

    def stokes_i_visibilities(self) -> NDArray:
        """
        Stokes I visibilities as a numpy array with shape (nrows, nchan)
        """
        vis = self.visibilities()
        # NOTE: assuming XX, XY, YX, YY correlations
        return 0.5 * (vis[..., 0] + vis[..., 3])

    def stokes_i_flags(self) -> NDArray:
        """
        Appropriate flags for Stokes I imaging. A Stokes I visibility is
        flagged if any of the correlations that contribute to its calculation
        is flagged.
        """
        flags = self.flags()
        return flags[..., (0, 3)].max(axis=-1)

    def stokes_i_weights(self) -> Optional[NDArray]:
        """
        Appropriate weights for Stokes I visibilities as a numpy array with
        shape (nrows, nchan, npol). If the WEIGHT_SPECTRUM column does not
        exist, returns None.
        """
        weights = self.weights()
        if weights is None:
            return None

        with warnings.catch_warnings():
            # Ignore division by zero warning, it all works out below even
            # for zero weights.
            warnings.simplefilter("ignore")

            # Stokes I visibilities = 1/2 * (vis_xx + vis_yy).
            # Weights are the inverse of variances, and variances add linearly,
            # hence this formula.
            wxx = weights[..., 0]
            wyy = weights[..., 3]
            return 4.0 / (1.0 / wxx + 1.0 / wyy)
