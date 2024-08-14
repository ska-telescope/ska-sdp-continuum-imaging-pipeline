import itertools
import os
from pathlib import Path

from dask.distributed import Client, Future, as_completed, get_worker
from numpy.typing import NDArray

from ska_sdp_cip import MeasurementSetReader
from ska_sdp_cip.uvw_tiling.tile import Tile, rechunk_tiles_on_disk
from ska_sdp_cip.uvw_tiling.tiling_plan import (
    TileCoords,
    TileMapping,
    create_uvw_tile_mapping,
)


# pylint:disable=too-many-locals
def reorder_by_uvw_tile(
    ms_reader: MeasurementSetReader,
    tile_size: TileCoords,
    outdir: Path,
    client: Client,
    *,
    num_time_intervals: int = 16,
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
            into.
        max_vis_per_chunk: The maximum number of visibilities per tile chunk in
            the final output files.

    Returns:
        list[Path]: A list of Paths to the tile chunks that were written.
    """
    # Must make paths absolute before sending them to other workers
    outdir = outdir.resolve()
    channel_freqs = ms_reader.channel_frequencies()
    tile_coords_lists_futures: list[Future] = []

    # Reorder time intervals in parallel
    for interval_index, interval_reader in enumerate(
        ms_reader.partition(num_time_intervals, 1)
    ):
        tile_mapping = client.submit(
            create_time_interval_tile_mapping,
            interval_reader,
            tile_size,
            channel_freqs,
            resources={"processing_slots": 1},
        )

        tile_coords_list = client.submit(
            reorder_time_interval,
            interval_reader,
            tile_mapping,
            outdir,
            interval_index=interval_index,
        )

        tile_coords_lists_futures.append(tile_coords_list)

    # Wait until all intervals have been reordered
    # Build the set of tile coordinates that are non-empty
    tile_coords_set = set()
    for future in as_completed(tile_coords_lists_futures):
        tile_coords_set.update(future.result())

    # Rechunk all tiles
    # NOTE: tile populations vary widely, so maybe we should avoid having
    # one task per tile, and instead group some sparsely populated tiles
    # together.
    rechunk_futures = [
        client.submit(
            rechunk_tile_chunk_group,
            tile_coords,
            outdir,
            max_vis_per_chunk=max_vis_per_chunk,
        )
        for tile_coords in tile_coords_set
    ]

    output_paths = list(
        itertools.chain.from_iterable(
            fut.result() for fut in as_completed(rechunk_futures)
        )
    )
    return output_paths


def create_time_interval_tile_mapping(
    ms_reader: MeasurementSetReader,
    tile_size: TileCoords,
    channel_freqs: NDArray,
) -> TileMapping:
    """
    Compute the UVW tile mapping for a particular time interval.
    """
    nthreads = _get_num_threads()
    uvw = ms_reader.uvw()
    return create_uvw_tile_mapping(
        uvw, tile_size, channel_freqs, processes=nthreads
    )


def reorder_time_interval(
    ms_reader: MeasurementSetReader,
    tile_mapping: TileMapping,
    outdir: Path,
    *,
    interval_index: int,
) -> list[TileCoords]:
    """
    Perform the actual reordering of the visibilities inside a time interval,
    given a pre-computed UVW time mapping for it.

    Returns the list of TileCoords present in that interval.
    """

    uvw = ms_reader.uvw()
    vis = ms_reader.visibilities()
    stokes_i_vis = 0.5 * (vis[..., 0] + vis[..., 3])

    for coords, row_slices in tile_mapping.items():
        # pylint: disable=protected-access
        tile = Tile._from_jagged_visibilities_slice(
            stokes_i_vis, uvw, coords, row_slices
        )
        path = outdir / _tile_filename(coords, interval_index)
        tile.save_npz(path)

    return list(tile_mapping.keys())


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
