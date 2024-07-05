import subprocess
import tempfile
from pathlib import Path

import numpy as np
from dask.distributed import Client

from ska_sdp_cip import MeasurementSetReader
from ska_sdp_cip.apps.pipeline_app import run_program


def test_entrypoint_exists():
    """
    Self-explanatory.
    """
    subprocess.check_call(["ska-sdp-cip", "--help"])


def test_pipeline_app(ms_reader: MeasurementSetReader):
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
            str(ms_reader.path),
            str(image_path),
        ]
        run_program(cli_args)

        assert image_path.is_file()
        data = np.load(image_path)
        assert data.shape == (num_pixels, num_pixels)


def test_pipeline_app_with_dask(
    ms_reader: MeasurementSetReader, dask_client: Client
):
    """
    Run the pipeline app on the test measurement set using dask distribution.
    """
    num_pixels = 2048
    pixel_size_asec = 5.0

    cli_args_dict = {
        "-n": num_pixels,
        "-p": pixel_size_asec,
        "--dask-scheduler": str(dask_client.scheduler.address),
        "--row-chunks": 2,
        "--freq-chunks": 2,
    }

    # Optional arguments
    cli_args = []
    for key, val in cli_args_dict.items():
        cli_args.append(key)
        cli_args.append(str(val))

    with tempfile.TemporaryDirectory() as tempdir:
        image_path = Path(tempdir) / "image.npy"

        # Positional arguments
        cli_args.extend([str(ms_reader.path), str(image_path)])
        print(cli_args)
        run_program(cli_args)

        assert image_path.is_file()
        data = np.load(image_path)
        assert data.shape == (num_pixels, num_pixels)
