from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Iterator, Union

from casacore.tables import table
from numpy.typing import NDArray


class UnsupportedMeasurementSetLayout(Exception):
    """
    Exception raised when a given MeasurementSet layout deviates from what is
    expected/supported.
    """


def open_table_readonly(
    mspath: Union[str, os.PathLike], table_name: str = ""
) -> table:
    """
    Opens casacore table in measurement set `mspath` with the given name, in
    read-only mode. "MAIN" is accepted as a table name. Opens the MAIN table
    if no name is provided.
    """
    if not table_name or table_name == "MAIN":
        table_spec = str(mspath)
    else:
        table_spec = f"{mspath}::{table_name}"
    return table(table_spec, readonly=True, ack=False)


def get_column_data(
    mspath: Union[str, os.PathLike], table_name: str, column_name: str
) -> NDArray:
    """
    Get measurement set column as a numpy array.
    """
    with open_table_readonly(mspath, table_name) as tbl:
        return tbl.getcol(column_name)


@functools.lru_cache()
def get_num_rows(mspath: Union[str, os.PathLike], table_name: str) -> int:
    """
    Get number of rows in measurement set table.
    """
    with open_table_readonly(mspath, table_name) as tbl:
        return tbl.nrows()


class MeasurementSetMetadata:
    """
    Class for reading and checking the dimensions and layout of a
    measurement set.
    """

    def __init__(
        self, path: Union[str, os.PathLike], *, validate_layout: bool = True
    ) -> None:
        """
        Create a new MeasurementSetMetadata instance. `validate_layout` is for
        internal use only.
        """
        self._path = Path(path).resolve()
        if not self.path.is_dir():
            msg = (
                "Cannot initialise MeasurementSet: path is not a directory: "
                f"{self.path}"
            )
            raise FileNotFoundError(msg)

        if validate_layout:
            self._validate_layout()

    def _validate_layout(self) -> None:
        """
        Enforce layout restrictions.
        """
        if not get_num_rows(self.path, "SPECTRAL_WINDOW") == 1:
            raise UnsupportedMeasurementSetLayout(
                "Multiple spectral windows are not supported"
            )

        if not get_num_rows(self.path, "FIELD") == 1:
            raise UnsupportedMeasurementSetLayout(
                "Multiple fields are not supported"
            )

        if not get_num_rows(self.path, "POLARIZATION") == 1:
            raise UnsupportedMeasurementSetLayout(
                "Mixed polarization rows are not supported"
            )

        corr_types = get_column_data(self.path, "POLARIZATION", "CORR_TYPE")[0]
        accepted_corr_types = {
            (9, 10, 11, 12),  # linear polarisation frame
            (5, 6, 7, 8),  # circular polarisation frame
        }
        if not tuple(corr_types) in accepted_corr_types:
            raise UnsupportedMeasurementSetLayout(
                "Polarization channels must be either XX, XY, YX, YY or "
                "RR, RL, LR, LL"
            )

    @property
    def path(self) -> Path:
        """
        Absolute path on disk.
        """
        return self._path

    @property
    def num_data_rows(self) -> int:
        """
        Total number of rows in MAIN table.
        """
        return get_num_rows(self.path, "MAIN")

    @functools.cached_property
    def num_channels(self) -> int:
        """
        Total number of frequency channels.
        """
        # NOTE: works only because we're assuming a single spectral window
        return get_column_data(self.path, "SPECTRAL_WINDOW", "CHAN_FREQ").size


class MeasurementSetReader:
    """
    Provides reading of a MeasurementSet v2, or a slice thereof. It is
    effectively stores a path to disk and reading bounds along the rows
    and the frequency channels.

    Example
    -------
    >>> ms = MeasurementSet("path/to/dataset.ms")
    >>> ms.set_row_bounds(0, 1_000_000)
    >>> ms.set_channel_bounds(16, 32)
    """

    def __init__(
        self,
        path: Union[str, os.PathLike],
        *,
        validate_layout: bool = True,
    ) -> None:
        """
        Initialize a MeasurementSetReader instance, with reading bounds that
        include the whole data.

        Parameters
        ----------
        path : Union[str, os.PathLike]
            The path to the MeasurementSet.
        validate_layout : bool, optional
            For internal use only, leave to default value of True.
        """
        self._metadata = MeasurementSetMetadata(
            path, validate_layout=validate_layout
        )
        self._row_start = 0
        self._row_end = self._metadata.num_data_rows
        self._channel_start = 0
        self._channel_end = self._metadata.num_channels

    @property
    def path(self) -> Path:
        """
        Absolute path on disk.
        """
        return self._metadata.path

    @property
    def row_start(self) -> int:
        """
        Absolute start row index.
        """
        return self._row_start

    @property
    def row_end(self) -> int:
        """
        Absolute end row index (exclusive).
        """
        return self._row_end

    @property
    def num_data_rows(self) -> int:
        """
        Total number of rows in MAIN table within reading bounds.
        """
        return self.row_end - self.row_start

    @property
    def channel_start(self) -> int:
        """
        Absolute start channel index.
        """
        return self._channel_start

    @property
    def channel_end(self) -> int:
        """
        Absolute end channel index (exclusive).
        """
        return self._channel_end

    @property
    def num_channels(self) -> int:
        """
        Total number of frequency channels within reading bounds.
        """
        return self.channel_end - self.channel_start

    def set_row_bounds(self, row_start: int, row_end: int) -> None:
        """
        Set reading bounds along the row dimension. Out-of-bounds arguments
        are clipped. Start index is inclusive, end index is exclusive.
        """
        self._row_start = max(row_start, 0)
        self._row_end = min(row_end, self._metadata.num_data_rows)

    def set_channel_bounds(self, channel_start: int, channel_end: int) -> None:
        """
        Set reading bounds along the frequency dimension. Out-of-bounds
        arguments are clipped. Start index is inclusive, end index is
        exclusive.
        """
        self._channel_start = max(channel_start, 0)
        self._channel_end = min(channel_end, self._metadata.num_channels)

    def partition(
        self, row_chunks: int, freq_chunks: int
    ) -> list[MeasurementSetReader]:
        """
        Partition the measurement set into chunks based on the specified number
        of row and frequency chunks.

        Parameters
        ----------
        row_chunks : int
            The number of chunks to divide the data rows into.
            Must be less than the total number of data rows.
        freq_chunks : int
            The number of chunks to divide the frequency channels into.
            Must be less than the total number of frequency channels.

        Returns
        -------
        list of MeasurementSetReader
            A list of MeasurementSetReader objects, each covering its own
            data slice.
        """
        if not 1 <= row_chunks <= self.num_data_rows:
            raise ValueError(
                "Number of row chunks must be within [1, total data rows]"
            )

        if not 1 <= freq_chunks <= self.num_channels:
            raise ValueError(
                "Number of row chunks must be within [1, total freq channels]"
            )

        result = []
        for row_start, row_end in balanced_chunk_bounds(
            self.row_start, self.row_end, row_chunks
        ):
            for channel_start, channel_end in balanced_chunk_bounds(
                self.channel_start, self.channel_end, freq_chunks
            ):
                reader = MeasurementSetReader(self.path, validate_layout=False)
                reader.set_row_bounds(row_start, row_end)
                reader.set_channel_bounds(channel_start, channel_end)
                result.append(reader)
        return result

    def channel_frequencies(self) -> NDArray:
        """
        Channel frequencies in Hz as a one-dimensional numpy array.
        """
        with open_table_readonly(self.path, "SPECTRAL_WINDOW") as tbl:
            data = tbl.getcolslice(
                "CHAN_FREQ", blc=self.channel_start, trc=self.channel_end - 1
            )
            # Assuming single spectral window
            return data[0]

    def uvw(self) -> NDArray:
        """
        UVW coordinates as a numpy array with shape (nrows, 3).
        """
        with open_table_readonly(self.path, "MAIN") as tbl:
            return tbl.getcol(
                "UVW", startrow=self.row_start, nrow=self.num_data_rows
            )

    def flags(self) -> NDArray:
        """
        Flags as a boolean numpy array with shape (nrows, nchan, npol).
        """
        with open_table_readonly(self.path, "MAIN") as tbl:
            # NOTE: assuming 4 polarisations
            return tbl.getcolslice(
                "FLAG",
                blc=(self.channel_start, 0),
                trc=(self.channel_end - 1, 3),
                startrow=self.row_start,
                nrow=self.num_data_rows,
            )

    def visibilities(self) -> NDArray:
        """
        Visibilities as a numpy array with shape (nrows, nchan, npol).
        """
        with open_table_readonly(self.path, "MAIN") as tbl:
            # NOTE: assuming 4 polarisations
            return tbl.getcolslice(
                "DATA",
                blc=(self.channel_start, 0),
                trc=(self.channel_end - 1, 3),
                startrow=self.row_start,
                nrow=self.num_data_rows,
            )

    def weights(self) -> NDArray:
        """
        Correlator weights as a numpy array with shape (nrows, nchan, npol).
        This is the contents of the WEIGHT_SPECTRUM column. If the column does
        not exist, return the contents of the WEIGHT column repeated along the
        frequency dimension.
        """
        try:
            with open_table_readonly(self.path, "MAIN") as tbl:
                # NOTE: assuming 4 polarisations
                return tbl.getcolslice(
                    "WEIGHT_SPECTRUM",
                    blc=(self.channel_start, 0),
                    trc=(self.channel_end - 1, 3),
                    startrow=self.row_start,
                    nrow=self.num_data_rows,
                )

        # WEIGHT_SPECTRUM column does not exist: use WEIGHT column.
        # It has shape (nrows, ncol), we repeat it to (nrows, nchan, npol)
        except RuntimeError:
            with open_table_readonly(self.path, "MAIN") as tbl:
                data: NDArray = tbl.getcolslice(
                    "WEIGHT",
                    blc=0,
                    trc=3,
                    startrow=self.row_start,
                    nrow=self.num_data_rows,
                )
                nrow, npol = data.shape
                data = data.reshape(nrow, 1, npol)
                return data.repeat(self.num_channels, axis=1)


def balanced_chunk_sizes(n: int, k: int) -> Iterator[int]:
    """
    When dividing a population of size `n` into `k` chunks, returns the sizes
    of the `k` chunks that are as balanced as possible.
    """
    if not n > 0:
        raise ValueError("n must be > 0")
    if not 0 < k <= n:
        raise ValueError("k must be > 0 and <= n")

    q = n // k
    r = n % k
    for _ in range(0, r):
        yield q + 1
    for _ in range(r, k):
        yield q


def balanced_chunk_bounds(
    start: int, end: int, k: int
) -> Iterator[tuple[int, int]]:
    """
    When dividing a range of indices between `start` and `end` into `k` chunks,
    yield the start and end indices of the chunks so that their sizes are
    as balanced as possible.
    """
    n = end - start
    for size in balanced_chunk_sizes(n, k):
        end = start + size
        yield start, end
        start = end
