import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional
from collections import defaultdict

import dask
from dask.distributed import (
    Client,
    Future,
    WorkerPlugin,
    as_completed,
    get_worker,
)
from distributed import Worker
from numpy.typing import NDArray

from ska_sdp_cip import MeasurementSetReader
from ska_sdp_cip.uvw_tiling.tile import (
    Tile,
    rechunk_tiles_on_disk,
    rechunk_tiles,
)
from ska_sdp_cip.uvw_tiling.tile_mapping import (
    TileCoords,
    TileMapping,
    create_uvw_tile_mapping,
)


class AddProcessPool(WorkerPlugin):
    """
    WorkerPlugin that makes a worker run tasks via processes rather than
    threads. This is useful because we're running tasks that do NOT release
    the GIL, including reading MSv2 chunks via casacore.

    See: https://www.youtube.com/watch?v=vF2VItVU5zg
    """

    def setup(self, worker: Worker):
        executor = ProcessPoolExecutor(max_workers=worker.state.nthreads)
        worker.executors["processes"] = executor


# pylint:disable=too-many-locals
def reorder_by_uvw_tile(
    ms_reader: MeasurementSetReader,
    tile_size: TileCoords,
    outdir: Path,
    client: Client,
    *,
    num_time_intervals: Optional[int] = None,
    max_vis_per_chunk: int = 5_000_000,
) -> list[Path]:
    """
    Convert visibilities to Stokes I and reorder them into UVW tile chunks
    using multiple dask workers.

    The reordering is made in two passes:
    1. Process individual time intervals, bin visibilities into tiles for each
        interval, save tiles to disk.
    2. Group tiles with the same coordinates, and rechunk them into files
        containing as close to `max_vis_per_chunk` visibilities.

    Args:
        ms_reader: The MeasurementSetReader reader instance to read the
            visibility data.
        tile_size: The size of the tiles in usual uvw coordinates (in units)
            of wavelengths, as a tuple (u_size, v_size, w_size).
        outdir: The output directory where the reordered tiles will be written.
        client: The dask client used to manage parallel tasks.
        num_time_intervals: The number of time intervals to partition the data
            into. If None, pick a multiple of the number of available workers.
        max_vis_per_chunk: The maximum number of visibilities per tile chunk in
            the final output files.

    Returns:
        list[Path]: A list of Paths to the tile chunks that were written.
    """

    if num_time_intervals is None:
        total_threads = sum(
            winfo["nthreads"]
            for winfo in client.scheduler_info()["workers"].values()
        )
        num_time_intervals = max(2 * total_threads, 2)

    # Must make paths absolute before sending them to other workers
    outdir = outdir.resolve()
    channel_freqs = ms_reader.channel_frequencies()

    paths_written_futures: list[Future] = []
    remainder_chunks_futures: list[Future] = []

    client.register_plugin(AddProcessPool())

    # Reorder time intervals in parallel
    for interval_reader in ms_reader.partition(num_time_intervals, 1):
        with dask.annotate(executor="processes"):
            tile_mapping = client.submit(
                create_time_interval_tile_mapping,
                interval_reader,
                tile_size,
                channel_freqs,
            )

            tiles = client.submit(
                reorder_time_interval, interval_reader, tile_mapping
            )

            rechunk_result = client.submit(
                rechunk_and_export,
                [tiles],
                outdir,
            )

            paths_written, remaining_tiles = unpack_future_sequence(
                rechunk_result, 2, client
            )

            paths_written_futures.append(paths_written)
            remainder_chunks_futures.append(remaining_tiles)

    fut = client.submit(
        rechunk_and_export, remainder_chunks_futures, outdir, force_export=True
    )
    paths_written, __ = fut.result()

    # for x in paths_written_futures:
    #     print(x)
    #     print(x.result())
    #     print(75 * "=")

    all_paths_written = paths_written + list(
        itertools.chain.from_iterable(
            map(Future.result, paths_written_futures)
        )
    )
    return all_paths_written


def create_time_interval_tile_mapping(
    ms_reader: MeasurementSetReader,
    tile_size: TileCoords,
    channel_freqs: NDArray,
) -> TileMapping:
    """
    Compute the UVW tile mapping for a particular time interval.
    """
    uvw = ms_reader.uvw()
    return create_uvw_tile_mapping(uvw, tile_size, channel_freqs)


def reorder_time_interval(
    ms_reader: MeasurementSetReader, tile_mapping: TileMapping
) -> list[Tile]:
    """
    Perform the actual reordering of the visibilities inside a time interval,
    given a pre-computed UVW time mapping for it.

    Returns the list of TileCoords present in that interval.
    """

    uvw = ms_reader.uvw()
    vis = ms_reader.visibilities()
    stokes_i_vis = 0.5 * (vis[..., 0] + vis[..., 3])

    return [
        Tile._from_jagged_visibilities_slice(
            stokes_i_vis, uvw, coords, row_slices
        )
        for coords, row_slices in tile_mapping.items()
    ]


def rechunk_and_export(
    tile_lists: list[list[Tile]], outdir: Path, *, force_export: bool = False
) -> tuple[list[Path], list[Tile]]:
    """
    TODO
    """
    mapping: dict[TileCoords, list[Tile]] = defaultdict(list)

    for tiles in tile_lists:
        for tile in tiles:
            mapping[tile.coords].append(tile)

    paths_written: list[Path] = []
    remainder_tiles: list[Tile] = []

    for coords, tiles in mapping.items():
        u, v, w = coords
        paths, extra_tiles = rechunk_tiles(
            tiles,
            outdir,
            basename=f"tile_iu{u:+03d}_iv{v:+03d}_iw{w:+03d}",
            force_export=force_export,
        )

        remainder_tiles.extend(extra_tiles)
        paths_written.extend(paths)

    return paths_written, remainder_tiles


def tuple_getitem(items: tuple, index: int):
    return items[index]


def unpack_future_sequence(
    fut: Future, num_items: int, client: Client
) -> list[Future]:
    return [
        client.submit(tuple_getitem, fut, index) for index in range(num_items)
    ]


def rechunk_tile_chunk_group(
    tile_coords: TileCoords,
    outdir: Path,
    *,
    max_vis_per_chunk: int = 5_000_000,
) -> list[Path]:
    """
    Rechunk the files associated to given tile coords that are present inside
    `outdir`. Returns the list of written file paths.
    """
    u, v, w = tile_coords
    pattern = f"tile_iu{u:+03d}_iv{v:+03d}_iw{w:+03d}_interval*.npz"
    input_paths = list(outdir.glob(pattern))
    output_basename = f"tile_iu{u:+03d}_iv{v:+03d}_iw{w:+03d}"

    output_paths = rechunk_tiles_on_disk(
        input_paths,
        outdir,
        output_basename,
        max_vis_per_chunk=max_vis_per_chunk,
    )

    for path in input_paths:
        path.unlink()

    return output_paths


def _tile_filename(tile_coords: TileCoords, interval_index: int) -> str:
    u, v, w = tile_coords
    return (
        f"tile_iu{u:+03d}_iv{v:+03d}_iw{w:+03d}_"
        f"interval{interval_index:02d}"
        ".npz"
    )


def _get_num_threads() -> int:
    """
    Returns the number of available threads, depending on context:
    - If called from a dask worker, return the number of threads assigned to
      the worker.
    - Otherwise, return the total number of threads available on the system.
    """
    try:
        return get_worker().state.nthreads
    except ValueError:
        return os.cpu_count()
