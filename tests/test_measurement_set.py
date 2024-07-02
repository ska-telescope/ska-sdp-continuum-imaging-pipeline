import numpy as np
import pytest

from ska_sdp_cip import MeasurementSetReader


def test_measurement_set_path_is_absolute(ms_reader: MeasurementSetReader):
    """
    Self-explanatory.
    """
    assert ms_reader.path == ms_reader.path.absolute()


def test_filenotfound_raised_on_nonexistent_path():
    """
    Self-explanatory.
    """
    with pytest.raises(FileNotFoundError):
        MeasurementSetReader("definitely/does/not/exist.ms")


def test_channel_frequencies(ms_reader: MeasurementSetReader):
    """
    Self-explanatory.
    """
    assert np.array_equal(
        ms_reader.channel_frequencies(),
        [959969726.5625, 960805664.0625, 961641601.5625, 962477539.0625],
    )


def test_reading_uvw_and_visibilities(ms_reader: MeasurementSetReader):
    """
    Check that reading uvw and visibilities does not raise any error.
    """
    ms_reader.uvw()
    ms_reader.visibilities()
    ms_reader.stokes_i_visibilities()


def test_reading_flags(ms_reader: MeasurementSetReader):
    """
    Check that the flags of the test measurement set have the expected shape.
    """
    flags = ms_reader.flags()
    assert flags.shape == (74214, 4, 4)


def test_reading_weights(ms_reader: MeasurementSetReader):
    """
    Check that the weights of the test measurement set have the expected shape.
    """
    assert ms_reader.weights().shape == (74214, 4, 4)
