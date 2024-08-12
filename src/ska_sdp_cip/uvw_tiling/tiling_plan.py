import math
import multiprocessing
import os
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


def create_uvw_tile_mapping_sequential(
    uvw: NDArray,
    tile_size: tuple[float, float, float],
    channel_freqs: NDArray,
    *,
    row_offset: int = 0,
) -> TileMapping:
    """
    Bin the UVW coordinates of visibilities by UVW tile,
    using a single process. The `row_offset` argument is used when
    parallelizing the work across multiple UVW chunks.
    """
    speed_of_light = 299792458.0
    wavelength_inv = channel_freqs.reshape(-1, 1) / speed_of_light
    tile_size = np.asarray(tile_size)
    tile_mapping = defaultdict(list)

    for irow, row_uvw in enumerate(uvw, start=row_offset):
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


class TileMappingCreator:
    """
    Helper class for parallelizing the computation of tile mappings.
    """

    def __init__(
        self, tile_size: tuple[float, float, float], channel_freqs: NDArray
    ) -> None:
        self.tile_size = tile_size
        self.channel_freqs = channel_freqs

    def __call__(
        self, chunk_and_offset: tuple[NDArray, int]
    ) -> dict[TileCoords, list[RowSliceId]]:
        uvw, row_offset = chunk_and_offset
        return create_uvw_tile_mapping_sequential(
            uvw, self.tile_size, self.channel_freqs, row_offset=row_offset
        )


def create_uvw_tile_mapping(
    uvw: NDArray,
    tile_size: tuple[float, float, float],
    channel_freqs: NDArray,
    *,
    processes: int = os.cpu_count(),
) -> TileMapping:
    """
    Bin the UVW coordinates of visibilities by UVW tile.

    Args:
        uvw: A 2D numpy array of shape (num_rows, 3) containing UVW
            coordinates.
        tile_size: A tuple of three floats representing the size of the tiles
            in the U, V, and W directions.
        channel_freqs: A 1D numpy array containing the channel frequencies.
        processes: Number of parallel processes to use.

    Returns:
        mapping: A dictionary where keys are tile coordinates as integer
            3-tuples, and the values are lists of `RowSlice` namedtuples.
            `RowSlice` carries a row index, start channel index and stop
            channel index.

    Notes:
        The tile with interger coordinates (i, j, k) is centered on uvw
        coordinates (i * Du, j * Dv, k * Dw) where (Du, Dv, Dw) is the tile
        size.
    """
    num_rows = len(uvw)
    rows_per_chunk = math.ceil(num_rows / processes)

    uvw_chunks_and_row_offsets = []
    for i in range(0, processes):
        row_start = i * rows_per_chunk
        row_end = (i + 1) * rows_per_chunk
        chunk = uvw[row_start:row_end]
        uvw_chunks_and_row_offsets.append((chunk, row_start))

    func = TileMappingCreator(tile_size, channel_freqs)

    # NOTE: we don't use a "with" statement to wrap the pool creation,
    # because if we do, we don't get code coverage measurements for code
    # executed by pool processes.
    # pylint:disable=consider-using-with
    pool = multiprocessing.Pool(processes=processes)
    result = merge_tile_mappings(pool.map(func, uvw_chunks_and_row_offsets))
    pool.close()
    pool.join()
    return result


def merge_tile_mappings(tile_mappings: list[TileMapping]) -> TileMapping:
    """
    Merge tile mappings into one.
    """

    result = defaultdict(list)
    for mapping in tile_mappings:
        for tile_coords, row_slices in mapping.items():
            result[tile_coords].extend(row_slices)

    return dict(result)


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
