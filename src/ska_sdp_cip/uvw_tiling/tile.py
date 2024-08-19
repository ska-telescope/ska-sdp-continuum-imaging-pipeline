from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from ska_sdp_cip.uvw_tiling.tile_mapping import RowSliceId, TileCoords


@dataclass(repr=False)
class Tile:
    """
    Stores the visibility data and metadata of a tile.
    """

    coords: TileCoords
    uvw: NDArray
    visibilities: NDArray
    channel_start_indices: NDArray
    channel_stop_indices: NDArray

    @property
    def num_rows(self) -> int:
        """
        Total number of rows stored.
        """
        return len(self.uvw)

    @property
    def num_visibilities(self) -> int:
        """
        Total number of visibilities stored.
        """
        return len(self.visibilities)

    def save_npz(self, path: Union[str, os.PathLike]) -> None:
        """
        Save to numpy's npz format.
        """
        items = {
            "coords": np.asarray(self.coords).astype(int),
            "uvw": self.uvw,
            "visibilities": self.visibilities,
            "channel_start_indices": self.channel_start_indices,
            "channel_stop_indices": self.channel_stop_indices,
        }
        np.savez(path, **items)

    @classmethod
    def load_npz(cls, path: Union[str, os.PathLike]) -> Tile:
        """
        Load from file in numpy's npz format.
        """
        npz = np.load(path)
        return cls(
            coords=tuple(map(int, npz["coords"])),
            uvw=npz["uvw"],
            visibilities=npz["visibilities"],
            channel_start_indices=npz["channel_start_indices"],
            channel_stop_indices=npz["channel_stop_indices"],
        )

    @classmethod
    def _zeros(
        cls, coords: TileCoords, num_row_slices: int, num_vis: int
    ) -> Tile:
        """
        Create a Tile object with the given number of rows and visibilities,
        with all data arrays initialised to zeros.
        """
        return Tile(
            coords=coords,
            uvw=np.zeros((num_row_slices, 3), dtype=float),
            visibilities=np.zeros(num_vis, dtype=np.complex64),
            channel_start_indices=np.zeros(num_row_slices, dtype=int),
            channel_stop_indices=np.zeros(num_row_slices, dtype=int),
        )

    @classmethod
    def _from_jagged_visibilities_slice(
        cls,
        vis: NDArray,
        uvw: NDArray,
        coords: TileCoords,
        row_slices: list[RowSliceId],
    ) -> Tile:
        """
        Create a Tile object from a block of visibilities in (row, freq) order,
        extracting the desired row slices from it.
        """
        # Pre-allocate tile
        tile_nvis = sum(r.chan_stop - r.chan_start for r in row_slices)
        tile = Tile._zeros(coords, len(row_slices), tile_nvis)

        vis_counter = 0
        row_counter = 0

        # Extract row slices
        for irow, start_ch, stop_ch in row_slices:
            row_nvis = stop_ch - start_ch
            out_slice = slice(vis_counter, vis_counter + row_nvis)
            chan_slice = slice(start_ch, stop_ch)
            tile.visibilities[out_slice] = vis[irow, chan_slice]
            tile.uvw[row_counter] = uvw[irow]
            tile.channel_start_indices[row_counter] = start_ch
            tile.channel_stop_indices[row_counter] = stop_ch

            vis_counter += row_nvis
            row_counter += 1

        return tile

    def __str__(self) -> str:
        return (
            f"Tile(coords={self.coords}, nrows={self.num_rows}, "
            f"nvis={self.num_visibilities})"
        )

    def __repr__(self) -> str:
        return str(self)


def concatenate_tiles(tiles: Sequence[Tile]) -> Tile:
    """
    Concatenate tiles into one.
    """
    if not tiles:
        raise ValueError("Cannot concatenate empty sequence of tiles")

    # Check coordinates are all the same
    coords = tiles[0].coords
    coords_identical = all(tile.coords == coords for tile in tiles)
    if not coords_identical:
        raise ValueError("Cannot merge tiles with different coordinates")

    array_names = [
        "uvw",
        "visibilities",
        "channel_start_indices",
        "channel_stop_indices",
    ]

    attributes = {
        name: np.concatenate([getattr(tile, name) for tile in tiles])
        for name in array_names
    }
    attributes["coords"] = coords
    return Tile(**attributes)


def split_tile(tile: Tile, max_vis_per_chunk: int) -> list[Tile]:
    """
    Split tile into several smaller tiles with no more visibility samples
    than the specified maximum.
    """
    result = []

    row_index = 0
    vis_index = 0

    chunk_rows = 0
    chunk_vis = 0

    for start, stop in zip(
        tile.channel_start_indices, tile.channel_stop_indices
    ):
        size = stop - start

        # If adding current row slice would put us above size limit, create
        # a new chunk beforehand.
        # NOTE: We need to make sure not to create empty chunks in the
        # unlikely case that `max_vis_per_chunk` is smaller than the size
        # of one row. We're not splitting individual rows.
        if chunk_vis + size > max_vis_per_chunk and chunk_rows > 0:
            row_slice = slice(row_index, row_index + chunk_rows)
            vis_slice = slice(vis_index, vis_index + chunk_vis)
            chunk = Tile(
                coords=tile.coords,
                uvw=tile.uvw[row_slice],
                visibilities=tile.visibilities[vis_slice],
                channel_start_indices=tile.channel_start_indices[row_slice],
                channel_stop_indices=tile.channel_stop_indices[row_slice],
            )
            result.append(chunk)

            row_index += chunk_rows
            vis_index += chunk_vis

            chunk_rows = 0
            chunk_vis = 0

        # current row slice goes to next chunk
        chunk_rows += 1
        chunk_vis += size

    # Deal with remainder of data, if any
    if chunk_rows:
        chunk = Tile(
            coords=tile.coords,
            uvw=tile.uvw[row_index:],
            visibilities=tile.visibilities[vis_index:],
            channel_start_indices=tile.channel_start_indices[row_index:],
            channel_stop_indices=tile.channel_stop_indices[row_index:],
        )
        result.append(chunk)

    return result


def rechunk_tiles_on_disk(
    tile_paths: Iterable[Path],
    outdir: Path,
    basename: str,
    *,
    max_vis_per_chunk: int = 5_000_000,
) -> list[Path]:
    """
    Given an iterable of tile chunk file paths with the same tile coordinates,
    write the associated data as a new set of files, each containing a number
    of visibilities as large as possible but less than `max_vis_per_chunk`.

    Output files are named `{basename}_{chunk_index}.npz`.
    Returns the list of output file Paths, in the order they were written.
    """
    queue: list[Tile] = []
    result: list[Path] = []
    num_written = 0

    def _write_tile(tile: Tile) -> None:
        nonlocal num_written
        filepath = outdir / f"{basename}_chunk{num_written:03d}.npz"
        tile.save_npz(filepath)
        result.append(filepath)
        num_written += 1

    for paths in tile_paths:
        tile = Tile.load_npz(paths)
        queue.append(tile)
        nvis_in_queue = sum(t.num_visibilities for t in queue)

        if len(queue) > 1 and nvis_in_queue > max_vis_per_chunk:
            queue = [concatenate_tiles(queue)]

        if len(queue) == 1 and nvis_in_queue > max_vis_per_chunk:
            # split tile in N chunks
            chunks = split_tile(queue[0], max_vis_per_chunk)

            # export full chunks
            for chunk in chunks[:-1]:
                _write_tile(chunk)

            # place remainder in queue
            queue = [chunks[-1]]

    if len(queue) > 1:
        queue = [concatenate_tiles(queue)]

    for tile in queue:
        _write_tile(tile)

    return result


def rechunk_tiles(
    tiles: Iterable[Tile],
    outdir: Path,
    basename: str,
    *,
    max_vis_per_chunk: int = 5_000_000,
    force_export: bool = False,
) -> tuple[list[Path], list[Tile]]:
    """
    Rechunk list of tiles with the same coordinates and write them to disk.
    Output files are named `{basename}_{UUID4}.npz`.
    If force_export is True, all data is exported, even if that means writing
    tiles smaller than `max_vis_per_chunk`.

    Returns a tuple :
        - List of Paths written
        - List of Tiles that were not exported because they were too small
    """
    paths: list[Path] = []
    queue: list[Tile] = []
    num_written = 0

    def _write_tile(tile: Tile) -> None:
        nonlocal num_written
        filepath = outdir / f"{basename}_{str(uuid.uuid4())}.npz"
        paths.append(filepath)
        tile.save_npz(filepath)
        num_written += 1

    for tile in tiles:
        queue.append(tile)
        nvis_in_queue = sum(t.num_visibilities for t in queue)

        if len(queue) > 1 and nvis_in_queue > max_vis_per_chunk:
            queue = [concatenate_tiles(queue)]

        if len(queue) == 1 and nvis_in_queue > max_vis_per_chunk:
            # split tile in N chunks
            chunks = split_tile(queue[0], max_vis_per_chunk)

            # export full chunks
            for chunk in chunks[:-1]:
                _write_tile(chunk)

            # place remainder in queue
            queue = [chunks[-1]]

    if len(queue) > 1:
        queue = [concatenate_tiles(queue)]

    if force_export:
        for tile in queue:
            _write_tile(tile)
        return paths, []

    return paths, queue
