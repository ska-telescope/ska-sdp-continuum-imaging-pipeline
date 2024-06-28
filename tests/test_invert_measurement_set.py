import numpy as np

from ska_sdp_cip import MeasurementSet, invert_measurement_set


def test_invert_measurement_set(measurement_set: MeasurementSet):
    """
    Invert the test measurement set, check that an image with the correct
    dimensions is returned.
    """
    num_pixels = 2048
    pixel_size_asec = 5.0
    image = invert_measurement_set(
        measurement_set, num_pixels=num_pixels, pixel_size_asec=pixel_size_asec
    )
    assert isinstance(image, np.ndarray)
    assert image.shape == (num_pixels, num_pixels)
