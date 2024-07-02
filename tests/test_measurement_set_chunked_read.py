import functools
from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray

from ska_sdp_cip import MeasurementSetReader


@dataclass
class ChunkedReadTestCase:
    """
    Parameters for a test case of MeasurementSetReader chunked data reading.
    """

    method_name: str
    """
    Method name of the MeasurementSetReader class that reads an array of data,
    e.g. "visibilities".
    """

    row_chunks: int
    """ Number of row chunks to use when reading """

    freq_chunks: int
    """ Number of frequency chunks to use when reading """

    @property
    def test_case_id(self) -> str:
        """
        Test case ID to be shown in terminal when running pytest.
        """
        return (
            f"{self.method_name}, row_chunks={self.row_chunks}, "
            f"freq_chunks={self.freq_chunks}"
        )


METHODS = ["visibilities", "flags", "weights", "uvw", "channel_frequencies"]
CHUNKING_PARAMS = [(1, 4), (2, 3), (7, 1)]


CHUNKED_READ_TEST_CASES = [
    ChunkedReadTestCase(method, row_chunks, freq_chunks)
    for method in METHODS
    for row_chunks, freq_chunks in CHUNKING_PARAMS
]

CHUNKED_READ_TEST_CASE_IDS = [
    case.test_case_id for case in CHUNKED_READ_TEST_CASES
]


@pytest.mark.parametrize(
    "case", CHUNKED_READ_TEST_CASES, ids=CHUNKED_READ_TEST_CASE_IDS
)
def test_chunked_read_correct(
    ms_reader: MeasurementSetReader, case: ChunkedReadTestCase
):
    """
    Perform a chunked read of a data column, check that the result is the same
    as when reading it in one go.
    """
    if case.method_name == "uvw":
        assert_chunked_uvw_read_correct(
            ms_reader, case.row_chunks, case.freq_chunks
        )
    elif case.method_name == "channel_frequencies":
        assert_chunked_channel_frequencies_read_correct(
            ms_reader, case.row_chunks, case.freq_chunks
        )
    else:
        assert_chunked_data_read_correct(
            ms_reader,
            case.method_name,
            case.row_chunks,
            case.freq_chunks,
        )


@functools.lru_cache()
def expected_data(mset: MeasurementSetReader, method_name: str) -> NDArray:
    """
    Read MeasurementSet data attribute in one go.
    """
    func = getattr(mset, method_name)
    return func()


def assert_chunked_data_read_correct(
    mset: MeasurementSetReader,
    method_name: str,
    row_chunks: int,
    freq_chunks: int,
):
    """
    Assert chunked read is correct for methods that return an array with the
    same shape as visibilities.
    """
    expected = expected_data(mset, method_name)
    for chunk in mset.partition(row_chunks, freq_chunks):
        getter = getattr(chunk, method_name)
        arr = getter()
        assert np.array_equal(
            arr,
            expected[
                chunk.row_start : chunk.row_end,
                chunk.channel_start : chunk.channel_end,
                ...,
            ],
        )


def assert_chunked_uvw_read_correct(
    mset: MeasurementSetReader,
    row_chunks: int,
    freq_chunks: int,
):
    """
    Self-explanatory.
    """
    expected = expected_data(mset, "uvw")
    for chunk in mset.partition(row_chunks, freq_chunks):
        getter = getattr(chunk, "uvw")
        arr = getter()
        assert np.array_equal(
            arr,
            expected[
                chunk.row_start : chunk.row_end,
                ...,
            ],
        )


def assert_chunked_channel_frequencies_read_correct(
    mset: MeasurementSetReader,
    row_chunks: int,
    freq_chunks: int,
):
    """
    Self-explanatory.
    """
    expected = expected_data(mset, "channel_frequencies")
    for chunk in mset.partition(row_chunks, freq_chunks):
        getter = getattr(chunk, "channel_frequencies")
        arr = getter()
        assert np.array_equal(
            arr,
            expected[
                chunk.channel_start : chunk.channel_end,
                ...,
            ],
        )
