import tempfile
from pathlib import Path
from typing import Iterator
from zipfile import ZipFile

import numpy as np
import pytest

from skip import MeasurementSet


@pytest.fixture(scope="module", name="measurement_set")
def fixture_measurement_set() -> Iterator[Path]:
    """
    Path to the temporary directory containing the unzipped test measurement
    set.
    """
    path = Path(__file__).parent / "data" / "aa2_mid_nano.zip"
    path = path.resolve()

    with (
        tempfile.TemporaryDirectory() as tempdir,
        ZipFile(path, "r") as zipped_ms,
    ):
        zipped_ms.extractall(tempdir)
        yield MeasurementSet(Path(tempdir) / "aa2_mid_nano.ms")


def test_measurement_set_path_is_absolute(measurement_set: MeasurementSet):
    """
    Self-explanatory.
    """
    assert measurement_set.path == measurement_set.path.absolute()


def test_filenotfound_raised_on_nonexistent_path():
    """
    Self-explanatory.
    """
    with pytest.raises(FileNotFoundError):
        MeasurementSet("definitely/does/not/exists.ms")


def test_channel_frequencies(measurement_set: MeasurementSet):
    """
    Self-explanatory.
    """
    assert np.array_equal(measurement_set.channel_frequencies(), [950.0e6])


def test_reading_uvw_and_visibilities(measurement_set: MeasurementSet):
    """
    Check that reading uvw and visibilities does not raise any error.
    """
    measurement_set.uvw()
    measurement_set.visibilities()
    measurement_set.stokes_i_visibilities()
