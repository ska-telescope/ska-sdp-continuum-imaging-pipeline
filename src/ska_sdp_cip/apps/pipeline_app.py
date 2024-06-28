import argparse
import sys
from pathlib import Path

import numpy as np

from ska_sdp_cip import MeasurementSet, __version__, invert_measurement_set


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
    parser.add_argument(
        "-n",
        "--num-pixels",
        type=int,
        required=True,
        help="Number of pixels across the image",
    )
    parser.add_argument(
        "-p",
        "--pixel-size",
        type=float,
        required=True,
        help="Pixel size in arcseconds at the image centre",
    )
    return parser


def run_program(cli_args: list[str]) -> None:
    """
    Run the app. This is the function called by the tests.
    """
    args = get_parser().parse_args(cli_args)
    mset = MeasurementSet(args.measurement_set)
    img = invert_measurement_set(
        mset, num_pixels=args.num_pixels, pixel_size_asec=args.pixel_size
    )
    np.save(args.output_image.with_suffix(".npy"), img)


def main() -> None:
    """
    Entry point for the pipeline app.
    """
    run_program(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
