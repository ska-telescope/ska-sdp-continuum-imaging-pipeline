import argparse
import sys
from pathlib import Path

from dask.distributed import Client, get_task_stream, performance_report

from ska_sdp_cip import MeasurementSetReader, __version__
from ska_sdp_cip.task_metrics import TaskMetrics
from ska_sdp_cip.uvw_tiling import reorder_by_uvw_tile


def get_parser() -> argparse.ArgumentParser:
    """
    Create the CLI parser for the app.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Convert MSv2 visibilities to Stokes I and sort them by UVW tile"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "measurement_set", type=Path, help="Path to MeasurementSet v2"
    )
    parser.add_argument(
        "-t",
        "--tile-size",
        nargs=3,
        type=float,
        required=True,
        help=(
            "UVW tile size in units of wavelength, as a space-separated "
            "sequence of 3 real-valued numbers"
        ),
    )
    parser.add_argument(
        "-d",
        "--dask-scheduler",
        type=str,
        required=True,
        help="Address of a dask scheduler to use for distribution",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=Path,
        default=Path.cwd(),
        help=(
            "Output directory for the reordered data (and temporary files). "
            "Will be created if it does not exist. "
            "Defaults to the current working directory."
        ),
    )
    parser.add_argument(
        "-n",
        "--num-time-intervals",
        type=int,
        default=None,
        help=(
            "Split the input data into this many time chunks between workers. "
            "If None, a choice is made automatically."
        ),
    )
    parser.add_argument(
        "-m",
        "--max-vis-per-chunk",
        type=int,
        default=5_000_000,
        help=(
            "Maximum number of visibility samples to be stored per tile chunk."
        ),
    )
    return parser


def run_program(cli_args: list[str]) -> None:
    """
    Run the app.
    """
    args = get_parser().parse_args(cli_args)
    mset = MeasurementSetReader(args.measurement_set)

    client = Client(args.dask_scheduler)
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    with get_task_stream(client) as stream, performance_report(
        filename="dask-report.html"
    ):
        reorder_by_uvw_tile(
            mset,
            tuple(args.tile_size),
            outdir,
            client,
            num_time_intervals=args.num_time_intervals,
            max_vis_per_chunk=args.max_vis_per_chunk,
        )

    TaskMetrics(stream.data).save_json(
        "task-list.json", indent=4, sort_keys=True
    )


def main() -> None:
    """
    Entry point for the reordering app.
    """
    run_program(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
