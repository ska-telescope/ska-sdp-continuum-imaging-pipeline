from dataclasses import dataclass

import pytest

from ska_sdp_cip import MeasurementSetReader


@dataclass
class PartitionIndicesTestCase:
    """
    Wraps arguments and expected result for a test of
    MeasurementSetReader.partition().
    """

    row_chunks: int
    freq_chunks: int
    expected_bounds: list[tuple[int]]
    """
    Expected chunk index bounds as a list of tuples
    (row_start, row_end, chan_start, chan_end)
    """

    @property
    def name(self) -> str:
        """
        Test case ID to be shown in terminal when running pytest.
        """
        return (
            f"(row_chunks={self.row_chunks}, freq_chunks={self.freq_chunks})"
        )


PARTITION_INDICES_TEST_CASES = [
    PartitionIndicesTestCase(
        row_chunks=1,
        freq_chunks=1,
        expected_bounds=[(0, 74214, 0, 4)],
    ),
    PartitionIndicesTestCase(
        row_chunks=2,
        freq_chunks=3,
        expected_bounds=[
            (0, 37107, 0, 2),
            (0, 37107, 2, 3),
            (0, 37107, 3, 4),
            (37107, 74214, 0, 2),
            (37107, 74214, 2, 3),
            (37107, 74214, 3, 4),
        ],
    ),
    PartitionIndicesTestCase(
        row_chunks=5,
        freq_chunks=1,
        expected_bounds=[
            (0, 14843, 0, 4),
            (14843, 29686, 0, 4),
            (29686, 44529, 0, 4),
            (44529, 59372, 0, 4),
            (59372, 74214, 0, 4),
        ],
    ),
]

PARTITION_INDICES_TEST_CASE_IDS = [
    case.name for case in PARTITION_INDICES_TEST_CASES
]


@pytest.mark.parametrize(
    "case", PARTITION_INDICES_TEST_CASES, ids=PARTITION_INDICES_TEST_CASE_IDS
)
def test_measurement_set_partition_indices(
    ms_reader: MeasurementSetReader, case: PartitionIndicesTestCase
):
    """
    Test that MeasurementSet.partition() returns chunks with the correct
    indices.
    """
    chunks = ms_reader.partition(case.row_chunks, case.freq_chunks)
    chunk_indices = [
        (c.row_start, c.row_end, c.channel_start, c.channel_end)
        for c in chunks
    ]
    assert chunk_indices == case.expected_bounds


def test_measurement_set_partition_raises_on_excessive_num_chunks(
    ms_reader: MeasurementSetReader,
):
    """
    Self-explanatory.
    """
    with pytest.raises(ValueError):
        ms_reader.partition(1_000_000, 1)

    with pytest.raises(ValueError):
        ms_reader.partition(1, 1_000_000)
