import numpy as np
import pytest

from skip import MeasurementSet


def test_measurement_set_path_is_absolute(measurement_set: MeasurementSet):
    """
    Self-explanatory.
    """
    assert measurement_set.path == measurement_set.path.absolute()


def test_filenotfound_raised_on_nonexistent_path():
    """
    Self-explanatory.
    """
    with pytest.raises(FileNotFoundError):
        MeasurementSet("definitely/does/not/exist.ms")


def test_channel_frequencies(measurement_set: MeasurementSet):
    """
    Self-explanatory.
    """
    assert np.array_equal(
        measurement_set.channel_frequencies(),
        [959969726.5625, 960805664.0625, 961641601.5625, 962477539.0625],
    )


def test_reading_uvw_and_visibilities(measurement_set: MeasurementSet):
    """
    Check that reading uvw and visibilities does not raise any error.
    """
    measurement_set.uvw()
    measurement_set.visibilities()
    measurement_set.stokes_i_visibilities()


def test_reading_flags(measurement_set: MeasurementSet):
    """
    Check that the flags of the test measurement set have the expected shape.
    """
    flags = measurement_set.flags()
    assert flags.shape == (74214, 4, 4)


def test_reading_weights(measurement_set: MeasurementSet):
    """
    Check that the weights of the test measurement set have the expected shape.
    """
    assert measurement_set.weights().shape == (74214, 4, 4)
