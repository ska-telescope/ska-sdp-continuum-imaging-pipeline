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

    # Us the Parameters of the first 256 channels of MeerKAT L-Band
    freq_start = 856.0e6
    bandwidth = 214.0e6
    nchan = 256
    freq_step = bandwidth / nchan
    channel_freqs = freq_start + freq_step * np.arange(nchan)

    plan = create_uvw_tiling_plan(uvw, tile_size, channel_freqs)

    # Assert that each visibility sample appears in one tile exactly
    num_tiles_hit = np.zeros(shape=(num_rows, nchan), dtype=int)

    for row_slice_list in plan.values():
        for irow, start_ch, stop_ch in row_slice_list:
            num_tiles_hit[irow, start_ch:stop_ch] += 1

    assert (num_tiles_hit == 0).sum() == 0
