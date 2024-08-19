import itertools
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import dask
from dask import delayed
from dask.delayed import Delayed
from dask.distributed import Client, WorkerPlugin
from distributed import Worker
from numpy.typing import NDArray

from ska_sdp_cip import MeasurementSetReader
from ska_sdp_cip.uvw_tiling.tile import Tile, rechunk_tiles
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
        max_vis_per_chunk: The maximum number of visibilities per tile chunk in
            the final output files.

    Returns:
        list[Path]: A list of Paths to the tile chunks that were written.
    """
    client.register_plugin(AddProcessPool())
    delayed_obj = recursive_reorder(
        ms_reader,
        tile_size,
        outdir,
        max_vis_per_chunk=max_vis_per_chunk,
    )
    return client.compute(delayed_obj).result()


def recursive_reorder(
    ms_reader: MeasurementSetReader,
    tile_size: TileCoords,
    outdir: Path,
    *,
    max_rows_per_interval: int = 131_072,
    max_vis_per_chunk: int = 5_000_000,
) -> Delayed:
    """
    Returns a Delayed object wrapping the list of tile chunk paths written.
    """
    # Must make paths absolute before sending them to other workers
    outdir = outdir.resolve()
    channel_freqs = ms_reader.channel_frequencies()

    paths_written, __ = _do_recursive_reorder(
        ms_reader,
        tile_size,
        channel_freqs,
        outdir,
        max_rows_per_interval=max_rows_per_interval,
        max_vis_per_chunk=max_vis_per_chunk,
        force_export=True,
    )
    return paths_written


def _do_recursive_reorder(
    ms_reader: MeasurementSetReader,
    tile_size: TileCoords,
    channel_freqs: NDArray,
    outdir: Path,
    *,
    max_rows_per_interval: int = 131_072,
    max_vis_per_chunk: int = 5_000_000,
    force_export: bool = False,
) -> tuple[Delayed, Delayed]:
    """
    Returns a tuple of Delayed containing:
    - The list of tile chunk paths written
    - The list of tiles that were too small to be exported
    """
    if ms_reader.num_data_rows <= max_rows_per_interval:
        # NOTE: force_export must be passed, in case the top-level input
        # measurement set has a sufficiently small number of input rows to be
        # processed without recursive splitting.
        return _do_reorder(
            ms_reader,
            tile_size,
            channel_freqs,
            outdir,
            max_vis_per_chunk=max_vis_per_chunk,
            force_export=force_export,
        )

    list_of_path_lists = []
    list_of_tile_lists = []

    for interval_reader in ms_reader.partition(8, 1):
        interval_paths, remaining_tiles = _do_recursive_reorder(
            interval_reader,
            tile_size,
            channel_freqs,
            outdir,
            max_rows_per_interval=max_rows_per_interval,
            max_vis_per_chunk=max_vis_per_chunk,
            force_export=False,
        )
        list_of_path_lists.append(interval_paths)
        list_of_tile_lists.append(remaining_tiles)

    paths_written = delayed(concatenate_lists)(list_of_path_lists)
    remaining_tiles = delayed(concatenate_lists)(list_of_tile_lists)

    extra_paths_written, remaining_tiles = delayed(rechunk_and_export, nout=2)(
        remaining_tiles,
        outdir,
        max_vis_per_chunk=max_vis_per_chunk,
        force_export=force_export,
    )
    paths_written = delayed(concatenate_lists)(
        [paths_written, extra_paths_written]
    )
    return paths_written, remaining_tiles


def _do_reorder(
    ms_reader: MeasurementSetReader,
    tile_size: TileCoords,
    channel_freqs: NDArray,
    outdir: Path,
    *,
    max_vis_per_chunk: int = 5_000_000,
    force_export: bool = False,
) -> tuple[Delayed, Delayed]:
    # Tile mapping computation is compute bound, so we parallelize over
    # processes
    with dask.annotate(executor="processes"):
        tile_mapping = delayed(create_time_interval_tile_mapping)(
            ms_reader,
            tile_size,
            channel_freqs,
        )
    tiles = delayed(reorder_time_interval)(ms_reader, tile_mapping)
    paths_written, remaining_tiles = delayed(rechunk_and_export, nout=2)(
        tiles,
        outdir,
        max_vis_per_chunk=max_vis_per_chunk,
        force_export=force_export,
    )
    return paths_written, remaining_tiles


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
        # pylint:disable=protected-access
        Tile._from_jagged_visibilities_slice(
            stokes_i_vis, uvw, coords, row_slices
        )
        for coords, row_slices in tile_mapping.items()
    ]


def rechunk_and_export(
    tiles: list[Tile],
    outdir: Path,
    *,
    max_vis_per_chunk: int = 5_000_000,
    force_export: bool = False,
) -> tuple[list[Path], list[Tile]]:
    """
    Rechunk list of Tiles with the same coordinates so that none contain more
    than `max_vis_per_chunk` visibility samples. Full tile chunks are written
    to `outdir`, the remaining chunks are returned.

    Return a tuple with two elements:
    - The list of tile chunk paths written
    - The list of tiles that were too small to be exported
    """
    mapping: dict[TileCoords, list[Tile]] = defaultdict(list)

    for tile in tiles:
        mapping[tile.coords].append(tile)

    paths_written: list[Path] = []
    remainder_tiles: list[Tile] = []

    for coords, tile_group in mapping.items():
        u, v, w = coords
        paths, extra_tiles = rechunk_tiles(
            tile_group,
            outdir,
            basename=f"tile_iu{u:+03d}_iv{v:+03d}_iw{w:+03d}",
            max_vis_per_chunk=max_vis_per_chunk,
            force_export=force_export,
        )

        remainder_tiles.extend(extra_tiles)
        paths_written.extend(paths)

    return paths_written, remainder_tiles


def concatenate_lists(lists: list[list]) -> list:
    """
    Self-explanatory.
    """
    return list(itertools.chain.from_iterable(lists))
