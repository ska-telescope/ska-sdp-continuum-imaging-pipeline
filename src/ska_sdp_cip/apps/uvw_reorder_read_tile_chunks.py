import argparse
import sys
import time
from pathlib import Path

from dask.distributed import (
    Client,
    as_completed,
    get_task_stream,
    performance_report,
)

from ska_sdp_cip import __version__
from ska_sdp_cip.task_metrics import TaskMetrics
from ska_sdp_cip.uvw_tiling import Tile


def get_parser() -> argparse.ArgumentParser:
    """
    Create the CLI parser for the app.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Read UVW tile chunks in parallel on a dask cluster, to measure "
            "parallel reading performance."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory from which to read tile chunks in .npz format",
    )
    parser.add_argument(
        "-d",
        "--dask-scheduler",
        type=str,
        required=True,
        help="Address of the dask scheduler to use for distribution",
    )
    return parser


def read_all_tiles(paths: list[Path], client: Client) -> int:
    """
    Read all tiles in distributed manner. Return the total number of
    visibilities read.
    """
    futures = client.map(read_tile, paths)
    results = [f.result() for f in as_completed(futures)]
    return sum(results)


def read_tile(path: Path) -> int:
    """
    Read given tile, return the number of visibilities read.
    """
    return Tile.load_npz(path).num_visibilities


def run_program(cli_args: list[str]) -> None:
    """
    Run the app.
    """
    args = get_parser().parse_args(cli_args)
    input_dir: Path = args.input_dir
    client = Client(args.dask_scheduler)

    with get_task_stream(client) as stream, performance_report(
        filename="dask-report.html"
    ):
        paths = list(input_dir.resolve().glob("*.npz"))
        start = time.perf_counter()
        total_num_vis = read_all_tiles(paths, client)
        elapsed = time.perf_counter() - start

    print(f"Total tile chunks: {len(paths):,d}")
    print(f"Visibilities read: {total_num_vis:,d}")
    print(f"Total time       : {elapsed:.2f} seconds")
    print(f"Read throughput  : {8 * total_num_vis / elapsed:,.0f} bytes/s")

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
