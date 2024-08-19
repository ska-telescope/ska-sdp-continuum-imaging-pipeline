from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from dask.distributed import Client

from ska_sdp_cip import MeasurementSetReader
from ska_sdp_cip.uvw_tiling import Tile, reorder_by_uvw_tile


def test_reorder_msv2_by_uvw_tile(
    ms_reader: MeasurementSetReader, dask_client: Client
):
    """
    Self-explanatory.
    """
    tile_size = (3000.0, 3000.0, 6000.0)
    max_vis_per_chunk = 10_000

    with TemporaryDirectory() as tempdir_name:
        outdir = Path(tempdir_name)
        reorder_msv2_by_uvw_tile_and_check_result(
            ms_reader,
            tile_size,
            outdir,
            dask_client,
            max_vis_per_chunk=max_vis_per_chunk,
        )


def reorder_msv2_by_uvw_tile_and_check_result(
    ms_reader: MeasurementSetReader,
    tile_size: tuple[float, float, float],
    outdir: Path,
    client: Client,
    *,
    max_vis_per_chunk: int,
):
    """
    Performs the actual MSv2 reordering testing. We check that the UVW_lambda
    coordinates of every visibility sample is accounted for.
    """
    tile_chunk_paths = reorder_by_uvw_tile(
        ms_reader,
        tile_size,
        outdir,
        client,
        max_vis_per_chunk=max_vis_per_chunk,
    )

    assert_all_visibilities_accounted_for(ms_reader, tile_chunk_paths)


def assert_all_visibilities_accounted_for(
    ms_reader: MeasurementSetReader, tile_chunk_paths: list[Path]
):
    """
    Check that the UVW coordinate (in wavelengths) of every visibility is
    present exactly once. Assert the total number of visibilities checks out.
    We build an array of UVW_lambda coordinates of shape (nrows x nchan, 3),
    sorted by (u,v,w), both from reading the input data and the output
    data sorted by tile, and check that they are identical.
    """

    speed_of_light = 299792458.0

    expected_uvw_metres = ms_reader.uvw()
    freqs = ms_reader.channel_frequencies()
    expected_uvw_lambda = (
        expected_uvw_metres.reshape(-1, 1, 3)
        / speed_of_light
        * freqs.reshape(1, len(freqs), 1)
    )
    expected_uvw_lambda = np.sort(expected_uvw_lambda.reshape(-1, 3), axis=0)
    expected_num_vis = len(expected_uvw_metres) * len(freqs)

    actual_uvw_lambda = []
    actual_num_vis = 0

    for path in tile_chunk_paths:
        tile = Tile.load_npz(path)
        for irow, (start, stop) in enumerate(
            zip(tile.channel_start_indices, tile.channel_stop_indices)
        ):
            uvw_lambda = (
                tile.uvw[irow]
                / speed_of_light
                * freqs[start:stop].reshape(-1, 1)
            )
            actual_uvw_lambda.append(uvw_lambda)
            actual_num_vis += stop - start

    actual_uvw_lambda = np.sort(np.concatenate(actual_uvw_lambda), axis=0)

    assert np.array_equal(expected_uvw_lambda, actual_uvw_lambda)
    assert expected_num_vis == actual_num_vis
