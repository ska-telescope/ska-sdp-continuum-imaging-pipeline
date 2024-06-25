import tempfile
from pathlib import Path
from typing import Iterator
from zipfile import ZipFile

import pytest

from skip import MeasurementSet


@pytest.fixture(scope="module")
def measurement_set() -> Iterator[MeasurementSet]:
    """
    The unzipped test dataset as a MeasurementSet object.
    """
    path = Path(__file__).parent / ".." / "data" / "mkt_cdfs25_nano.zip"
    path = path.resolve()

    with (
        tempfile.TemporaryDirectory() as tempdir,
        ZipFile(path, "r") as zipped_ms,
    ):
        zipped_ms.extractall(tempdir)
        yield MeasurementSet(Path(tempdir) / "mkt_cdfs25_nano.ms")
