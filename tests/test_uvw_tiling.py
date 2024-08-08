import numpy as np

from ska_sdp_cip import MeasurementSetReader
from ska_sdp_cip.uvw_tiling import create_uvw_tiling_plan


def test_create_uvw_tiling_plan(ms_reader: MeasurementSetReader):
    """
    Test the UVW tiling plan creation on the UVW layout of the test dataset.
    Check that each visibility sample goes into exactly one tile.
    """
    uvw = ms_reader.uvw()
    num_rows = len(uvw)
    tile_size = (3_000.0, 3_000.0, 6_000.0)
    max_vis_per_chunk = 1_000_000

    # Us the Parameters of the first 256 channels of MeerKAT L-Band
    freq_start = 856.0e6
    bandwidth = 214.0e6
    nchan = 256
    freq_step = bandwidth / nchan
    channel_freqs = freq_start + freq_step * np.arange(nchan)

    plan = create_uvw_tiling_plan(
        uvw, tile_size, channel_freqs, max_vis_per_chunk=max_vis_per_chunk
    )

    # Assert that each visibility sample appears in one tile exactly
    num_tiles_hit = np.zeros(shape=(num_rows, nchan), dtype=int)

    for row_slice_list in plan.values():
        for x in row_slice_list:
            num_tiles_hit[x.irow, x.chan_start : x.chan_stop] += 1

    assert (num_tiles_hit == 1).sum() == num_tiles_hit.size
