import os
import shutil
import time
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from dask.distributed import Client, Future, get_worker

from ska_sdp_cip import MeasurementSetReader
from ska_sdp_cip.uvw_tiling.tile import Tile, concatenate_tiles, split_tile
from ska_sdp_cip.uvw_tiling.tiling_plan import (
    TileCoords,
    TileMapping,
    create_uvw_tile_mapping,
)


def reorder_by_uvw_tile(
    ms_reader: MeasurementSetReader,
    tile_size: TileCoords,
    outdir: Path,
    client: Client,
) -> None:
    """
    Top-level function.
    """
    start_time = time.perf_counter()

    # NOTE: should check that freqs are evenly spaced and nchan >= 2
    freqs = ms_reader.channel_frequencies()
    freq_start = freqs[0]
    freq_step = freqs[1] - freqs[0]

    num_intervals = 12  # NOTE: needs to be chosen dynamically
    futures: list[Future] = []

    for interval_index, interval_reader in enumerate(
        ms_reader.partition(num_intervals, 1)
    ):
        tile_mapping = client.submit(
            create_time_interval_tile_mapping,
            interval_reader,
            tile_size,
            freq_start=freq_start,
            freq_step=freq_step,
            num_channels=len(freqs),
            resources={"processing_slots": 1},
        )

        future = client.submit(
            reorder_time_interval,
            interval_reader,
            tile_mapping,
            outdir,
            interval_index=interval_index,
        )

        futures.append(future)

    for future in futures:
        future.result()

    elapsed = time.perf_counter() - start_time
    print(f"Finished in {elapsed:.1f} seconds")


def create_time_interval_tile_mapping(
    ms_reader: MeasurementSetReader,
    tile_size: TileCoords,
    *,
    freq_start: float,
    freq_step: float,
    num_channels: int,
) -> TileMapping:
    """
    Compute the UVW tile mapping for a particular time interval.
    """
    nthreads = _get_num_threads()
    uvw = ms_reader.uvw()
    channel_freqs = freq_start + freq_step * np.arange(num_channels)
    return create_uvw_tile_mapping(
        uvw, tile_size, channel_freqs, processes=nthreads
    )


def reorder_time_interval(
    ms_reader: MeasurementSetReader,
    tile_mapping: TileMapping,
    outdir: Path,
    *,
    interval_index: int,
) -> None:
    """
    Perform the actual reordering of the visibilities inside a time interval,
    given a pre-computed UVW time mapping for it.
    """

    uvw = ms_reader.uvw()
    vis = ms_reader.visibilities()
    stokes_i_vis = 0.5 * (vis[..., 0] + vis[..., 3])

    for coords, row_slices in tile_mapping.items():
        tile = Tile.from_jagged_visibilities_slice(
            stokes_i_vis, uvw, coords, row_slices
        )
        path = outdir / _tile_filename(coords, interval_index)
        tile.save_npz(path)


def _tile_filename(tile_coords: TileCoords, interval_index: int) -> str:
    u, v, w = tile_coords
    return (
        f"tile_iu{u:+03d}_iv{v:+03d}_iw{w:+03d}_"
        f"interval{interval_index:02d}"
        ".npz"
    )


def _iter_time_interval_tiles(
    directory: Path, tile_coords: TileCoords
) -> Iterator[Tile]:

    u, v, w = tile_coords
    pattern = f"tile_iu{u:+03d}_iv{v:+03d}_iw{w:+03d}*.npz"

    for path in directory.glob(pattern):
        yield Tile.load_npz(path)


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
