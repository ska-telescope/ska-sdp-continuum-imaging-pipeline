from collections import defaultdict

import numpy as np
from numpy.typing import NDArray

TileIndex = tuple[int, int, int]
"""
Integer tuple (iu, iv, iw) containing the coordinates of a tile in units
of the tile size. Indices can be negative. (0, 0, 0) corresponds to the central
tile, which is itself centered on u = v = w = 0.0.
"""

RowSliceIndex = tuple[int, int, int]
"""
Integer tuple (row_index, channel_start, channel_stop).
"""

SPEED_OF_LIGHT = 299792458.0


def create_uvw_tiling_plan(
    uvw: NDArray,
    tile_size: tuple[float, float, float],
    channel_freqs: NDArray,
) -> dict[TileIndex, list[RowSliceIndex]]:
    """
    Bin the UVW coordinates of a set of visibilities by UVW tile.

    Args:
        uvw: A 2D numpy array of shape (num_rows, 3) containing UVW
            coordinates.
        tile_size: A tuple of three floats representing the size of the tiles
            in the U, V, and W directions.
        channel_freqs: A 1D numpy array containing the channel frequencies.

    Returns:
        A dictionary where keys are integer tile indices of the form
        (iu, iv, iw) and values are lists of tuples of the form
        (row_index, channel_start, channel_stop). `row_index` refers to the
        input `uvw` array.
    """
    wavelength_inv = channel_freqs.reshape(-1, 1) / SPEED_OF_LIGHT
    tile_size = np.asarray(tile_size)
    plan = defaultdict(list)

    for irow, row_uvw in enumerate(uvw):
        # The +0.5 takes into account the fact that the tile with index
        # (0, 0, 0) is *centered* on the origin.
        tile_indices = np.floor(wavelength_inv * (row_uvw / tile_size) + 0.5)
        subarrays = _find_all_constant_tile_index_subarrays(tile_indices)
        for start_ch, stop_ch, tile_index in subarrays:
            index = tuple(tile_index)
            plan[index].append((irow, start_ch, stop_ch))

    return dict(plan)


def _find_all_constant_tile_index_subarrays(
    arr: NDArray, offset: int = 0
) -> list[tuple[int, int, TileIndex]]:
    """
    Find the parameters of all constant subarrays in given array of tile
    indices (iu, iv, iw) using a recursive binary search.
    We can use a binary search because we know the tile indices are
    monotonically increasing (channel frequencies are increasing).

    Returns a list of tuples (start_index, stop_index, tile_index).
    """
    n = len(arr)
    if np.array_equal(arr[0], arr[-1]):
        return [(offset, offset + n, arr[0])]

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
