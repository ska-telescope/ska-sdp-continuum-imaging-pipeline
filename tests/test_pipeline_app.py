import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ska_sdp_cip import MeasurementSet
from ska_sdp_cip.apps.pipeline_app import run_program


def test_entrypoint_exists():
    """
    Self-explanatory.
    """
    subprocess.check_call(["ska-sdp-cip", "--help"])


def test_pipeline_app(measurement_set: MeasurementSet):
    """
    Run the pipeline app on the test measurement set.
    """
    num_pixels = 2048
    pixel_size_asec = 5.0

    with tempfile.TemporaryDirectory() as tempdir:
        image_path = Path(tempdir) / "image.npy"
        cli_args = [
            "-n",
            str(num_pixels),
            "-p",
            str(pixel_size_asec),
            str(measurement_set.path),
            str(image_path),
        ]
        run_program(cli_args)

        assert image_path.is_file()
        data = np.load(image_path)
        assert data.shape == (num_pixels, num_pixels)
