from collections import defaultdict
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

TileCoords = tuple[int, int, int]
"""
Tile index of the form (iu, iv, iw)
"""


class RowSliceId(NamedTuple):
    """
    Specifies a slice of a particular visibility row along the frequency axis.
    """

    irow: int
    chan_start: int
    chan_stop: int


TileMapping = dict[TileCoords, list[RowSliceId]]


def create_uvw_tile_mapping(
    uvw: NDArray,
    tile_size: tuple[float, float, float],
    channel_freqs: NDArray,
) -> TileMapping:
    """
    Bin the UVW coordinates of visibilities by UVW tile,
    using a single process.
    """
    speed_of_light = 299792458.0
    wavelength_inv = channel_freqs.reshape(-1, 1) / speed_of_light
    tile_size = np.asarray(tile_size)
    tile_mapping = defaultdict(list)

    for irow, row_uvw in enumerate(uvw):
        # The +0.5 takes into account the fact that the tile with index
        # (0, 0, 0) is *centered* on the origin.
        tile_indices = np.floor(
            wavelength_inv * (row_uvw / tile_size) + 0.5
        ).astype(int)

        # Row slices correspond to sequences of constant tile coordinates
        # in the array above.
        subarrays = _find_all_constant_tile_index_subarrays(tile_indices)
        for start_ch, stop_ch, tile_coords in subarrays:
            tile_mapping[tile_coords].append(
                RowSliceId(irow, start_ch, stop_ch)
            )

    return tile_mapping


def _find_all_constant_tile_index_subarrays(
    arr: NDArray, offset: int = 0
) -> list[tuple[int, int, TileCoords]]:
    """
    Find the parameters of all constant subarrays in given array of tile
    indices (iu, iv, iw) using a recursive binary search.
    We can use a binary search because we know the tile indices are
    monotonically increasing (channel frequencies are increasing).

    Returns a list of tuples (start_index, stop_index, (iu, iv, iw)).
    """
    n = len(arr)
    if np.array_equal(arr[0], arr[-1]):
        # Convert np.int64 indices to native python int
        u, v, w = arr[0]
        tile_coords = (int(u), int(v), int(w))
        return [(offset, offset + n, tile_coords)]

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
