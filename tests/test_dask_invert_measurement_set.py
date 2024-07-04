import numpy as np
from dask.distributed import Client

from ska_sdp_cip import (
    MeasurementSetReader,
    dask_invert_measurement_set,
    invert_measurement_set,
)


def test_dask_invert_measurement_set(
    ms_reader: MeasurementSetReader, dask_client: Client
):
    """
    Invert the test measurement set with the computation distributed on a
    dask cluster, check that the image is the same as the one obtained
    without distribution.
    """
    num_pixels = 2048
    pixel_size_asec = 5.0

    ref_image = invert_measurement_set(ms_reader, num_pixels, pixel_size_asec)
    image = dask_invert_measurement_set(
        ms_reader,
        dask_client,
        num_pixels=num_pixels,
        pixel_size_asec=pixel_size_asec,
        row_chunks=2,
        freq_chunks=2,
    )
    epsilon = 1e-5
    atol = epsilon * abs(ref_image).max()
    rtol = epsilon
    assert np.allclose(image, ref_image, atol=atol, rtol=rtol)
