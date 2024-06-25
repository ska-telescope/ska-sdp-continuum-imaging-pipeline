import tempfile
from pathlib import Path
from typing import Iterator
from zipfile import ZipFile

import pytest

from skip import MeasurementSet


@pytest.fixture(scope="module")
def measurement_set() -> Iterator[MeasurementSet]:
    """
    Path to the temporary directory containing the unzipped test measurement
    set.
    """
    path = Path(__file__).parent / ".." / "data" / "aa2_mid_nano.zip"
    path = path.resolve()

    with (
        tempfile.TemporaryDirectory() as tempdir,
        ZipFile(path, "r") as zipped_ms,
    ):
        zipped_ms.extractall(tempdir)
        yield MeasurementSet(Path(tempdir) / "aa2_mid_nano.ms")
