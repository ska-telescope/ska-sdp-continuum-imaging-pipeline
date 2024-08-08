from collections import defaultdict
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

TileCoords = tuple[int, int, int]
"""
Tile index of the form (iu, iv, iw)
"""


class TileChunk(NamedTuple):
    """
    Aggregation of integer tile coordinates (iu, iv, iw) and of a chunk index.
    The chunk index exists because some tiles contain many visibilities and
    need to be split into smaller data pieces.
    """

    coords: TileCoords
    ichunk: int


class RowSlice(NamedTuple):
    """
    Specifies a slice of a particular visibility row along the frequency axis.
    """

    irow: int
    chan_start: int
    chan_stop: int


TilingPlan = dict[TileChunk, list[RowSlice]]

DEFAULT_MAX_VIS_PER_CHUNK = 1_000_000


def create_uvw_tile_mapping(
    uvw: NDArray, tile_size: tuple[float, float, float], channel_freqs: NDArray
) -> dict[TileCoords, list[RowSlice]]:
    """
    Bin the UVW coordinates of visibilities by UVW tile.
    """
    SPEED_OF_LIGHT = 299792458.0
    wavelength_inv = channel_freqs.reshape(-1, 1) / SPEED_OF_LIGHT
    tile_size = np.asarray(tile_size)

    tile_mapping: dict[TileCoords, list[RowSlice]] = defaultdict(list)

    for irow, row_uvw in enumerate(uvw):
        # The +0.5 takes into account the fact that the tile with index
        # (0, 0, 0) is *centered* on the origin.
        tile_indices = np.floor(
            wavelength_inv * (row_uvw / tile_size) + 0.5
        ).astype(int)

        # Row slices correspond to sequences of constant tile coordinates
        # in the array above.
        subarrays = _find_all_constant_tile_index_subarrays(tile_indices)
        for start_ch, stop_ch, tile_index in subarrays:
            tile_mapping[tile_index].append(RowSlice(irow, start_ch, stop_ch))

    return tile_mapping


def split_uvw_tile_mapping(
    tile_mapping: dict[TileCoords, list[RowSlice]],
    max_vis_per_chunk: int = DEFAULT_MAX_VIS_PER_CHUNK,
) -> TilingPlan:
    """
    Further split the tiles that contain more visibility samples than the
    specified maximum. Returns a new mapping TileChunk -> list of RowSlices.
    """
    tiling_plan: dict[TileChunk, list[RowSlice]] = defaultdict(list)

    for tile_index, row_slices in tile_mapping.items():
        tile_chunk = TileChunk(tile_index, 0)
        chunk_population = 0

        for row_slice in row_slices:
            num_vis = row_slice.chan_stop - row_slice.chan_start

            # Create new chunk if chunk population would exceed limit
            if (chunk_population + num_vis) >= max_vis_per_chunk:
                tile_chunk = TileChunk(
                    tile_chunk.coords, tile_chunk.ichunk + 1
                )
                chunk_population = 0

            tiling_plan[tile_chunk].append(row_slice)
            chunk_population += num_vis

    return dict(tiling_plan)


def create_uvw_tiling_plan(
    uvw: NDArray,
    tile_size: tuple[float, float, float],
    channel_freqs: NDArray,
    max_vis_per_chunk: int = DEFAULT_MAX_VIS_PER_CHUNK,
) -> TilingPlan:
    """
    Bin the UVW coordinates of visibilities by UVW tile. Tiles whose total
    number of visibilities would exceed the given limit are further split
    into smaller chunks.

    Args:
        uvw: A 2D numpy array of shape (num_rows, 3) containing UVW
            coordinates.
        tile_size: A tuple of three floats representing the size of the tiles
            in the U, V, and W directions.
        channel_freqs: A 1D numpy array containing the channel frequencies.
        max_vis_per_chunk: Maximum number of visibility samples that can be
            contained into a data chunk.

    Returns:
        tiling_plan: A dictionary where keys are `TilingChunk` namedtuples,
            and the values are lists of `RowSlice` namedtuples.
            `TilingChunk` is the aggregation of a 3-tuple of tile coordinates
            and of a chunk index. `RowSlice` carries a row index, start channel
            index and stop channel index.
    """
    tile_mapping = create_uvw_tile_mapping(uvw, tile_size, channel_freqs)
    return split_uvw_tile_mapping(tile_mapping, max_vis_per_chunk)


def _find_all_constant_tile_index_subarrays(
    arr: NDArray, offset: int = 0
) -> list[tuple[int, int, TileCoords]]:
    """
    Find the parameters of all constant subarrays in given array of tile
    indices (iu, iv, iw) using a recursive binary search.
    We can use a binary search because we know the tile indices are
    monotonically increasing (channel frequencies are increasing).

    Returns a list of tuples (start_index, stop_index, tile_index).
    """
    n = len(arr)
    if np.array_equal(arr[0], arr[-1]):
        return [(offset, offset + n, tuple(arr[0]))]

    half = n // 2
    head = _find_all_constant_tile_index_subarrays(arr[:half], offset)
    tail = _find_all_constant_tile_index_subarrays(arr[half:], offset + half)

    # Check if the last constant subarray in "head" and first right constant
    # subarray in "tail" have the same value. If so, they form a single
    # contiguous subarray, and we need to concatenate them.
    last_start, __, last_value = head[-1]
    __, first_stop, first_value = tail[0]

    if not np.array_equal(last_value, first_value):
        return head + tail

    return head[:-1] + [(last_start, first_stop, last_value)] + tail[1:]
