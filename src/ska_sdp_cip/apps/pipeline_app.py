import argparse
import sys
from pathlib import Path

import numpy as np
from dask.distributed import Client

from ska_sdp_cip import (
    MeasurementSetReader,
    __version__,
    dask_invert_measurement_set,
    invert_measurement_set,
)


def get_parser() -> argparse.ArgumentParser:
    """
    Create the CLI parser for the app.
    """
    parser = argparse.ArgumentParser(
        description=("Launch the SKA continuum imaging pipeline"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "measurement_set", type=Path, help="Path to MeasurementSet v2"
    )
    parser.add_argument(
        "output_image",
        type=Path,
        help="Path to output image, which is saved as a numpy array",
    )

    # Imaging options
    imaging_group = parser.add_argument_group("imaging")
    imaging_group.add_argument(
        "-n",
        "--num-pixels",
        type=int,
        required=True,
        help="Number of pixels across the image",
    )
    imaging_group.add_argument(
        "-p",
        "--pixel-size",
        type=float,
        required=True,
        help="Pixel size in arcseconds at the image centre",
    )

    # Dask options
    dask_group = parser.add_argument_group("dask distribution")
    dask_group.add_argument(
        "-d",
        "--dask-scheduler",
        type=str,
        default=None,
        help="Optional address of a dask scheduler to use for distribution",
    )
    dask_group.add_argument(
        "-rc",
        "--row-chunks",
        type=int,
        default=1,
        help="Number of row chunks to use for distribution",
    )
    dask_group.add_argument(
        "-fc",
        "--freq-chunks",
        type=int,
        default=None,
        help=(
            "Number of frequency chunks to use for distribution. "
            "If None, set this to the number of dask workers."
        ),
    )
    return parser


def run_program(cli_args: list[str]) -> None:
    """
    Run the app. This is the function called by the tests.
    """
    args = get_parser().parse_args(cli_args)
    mset = MeasurementSetReader(args.measurement_set)

    if args.dask_scheduler is None:
        img = invert_measurement_set(
            mset, num_pixels=args.num_pixels, pixel_size_asec=args.pixel_size
        )
    else:
        client = Client(args.dask_scheduler)
        img = dask_invert_measurement_set(
            mset,
            client,
            num_pixels=args.num_pixels,
            pixel_size_asec=args.pixel_size,
            row_chunks=args.row_chunks,
            freq_chunks=args.freq_chunks,
        )

    np.save(args.output_image.with_suffix(".npy"), img)


def main() -> None:
    """
    Entry point for the pipeline app.
    """
    run_program(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
