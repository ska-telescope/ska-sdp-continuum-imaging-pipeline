import os

import numpy as np

# pylint:disable=import-error,no-name-in-module
from ducc0.wgridder import ms2dirty
from numpy.typing import NDArray

from .measurement_set import MeasurementSet


def invert_measurement_set(
    mset: MeasurementSet,
    num_pixels: int,
    pixel_size_asec: float,
    nthreads: int = os.cpu_count(),
) -> NDArray:
    """
    Invert the given MeasurementSet, returning a dirty image.
    """
    pixel_size_lm = np.sin(np.radians(pixel_size_asec / 3600.0))

    mask = np.logical_not(mset.stokes_i_flags())
    weights = mset.stokes_i_weights()
    effective_weights = (
        weights * mask if weights is not None else mask.astype("float32")
    )
    normfactor = 1.0 / effective_weights.sum()

    return normfactor * ms2dirty(
        mset.uvw(),
        mset.channel_frequencies(),
        mset.stokes_i_visibilities(),
        effective_weights,
        num_pixels,
        num_pixels,
        pixel_size_lm,
        pixel_size_lm,
        epsilon=1e-4,
        do_wstacking=True,
        nthreads=nthreads,
        mask=None,  # already accounted for in effective_weights
    )
