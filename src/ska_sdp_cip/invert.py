from __future__ import annotations

import os
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np
from dask.distributed import Client, get_worker

# pylint:disable=import-error,no-name-in-module
from ducc0.wgridder import ms2dirty
from numpy.typing import NDArray

from .measurement_set import MeasurementSetReader


@contextmanager
def set_env(name: str, value: Any):
    """
    Set environment variable within a context.
    """
    previous_value = os.environ.get(name, None)
    os.environ[name] = str(value)
    try:
        yield
    finally:
        if previous_value is None:
            os.environ.pop(name)
        else:
            os.environ[name] = previous_value


# NOTE: This class currently exists for the sake of performing data reading in
# a separate dask task. It is also assumes the input data is in a linear
# or circular polarisation frame.
# We will likely need a Visibility class that can handle all valid
# polarisation conversions.
@dataclass
class StokesIGridderInput:
    """
    Wraps Stokes I visibilities and associated data arrays to be passed to the
    gridder.
    """

    channel_frequencies: NDArray
    """
    Channel frequencies as a numpy array with shape (nchan,)
    """

    flags: NDArray
    """
    Flags as a boolean numpy array with shape (nrows, nchan, npol).
    """

    uvw: NDArray
    """
    UVW coordinates as a numpy array with shape (nrows, 3).
    """

    visibilities: NDArray
    """
    Visibilities as a numpy array with shape (nrows, nchan, npol).
    """

    weights: NDArray
    """
    Data weights as a numpy array with shape (nrows, nchan, npol).
    """

    def effective_weights(self) -> NDArray:
        """
        Returns the product `weights x (1 - flags)`
        """
        return np.logical_not(self.flags) * self.weights

    @classmethod
    def from_measurement_set_reader(
        cls, ms_reader: MeasurementSetReader
    ) -> StokesIGridderInput:
        """
        Load data from MeasurementSetReader object, converting to Stokes I
        along the way.
        """
        # NOTE: Assuming 4 polarisation products, and that indices 0 and 3
        # Correspond to either {XX, YY} or {RR, LL} (order does not matter)
        vis = ms_reader.visibilities()
        stokes_i_vis = 0.5 * (vis[..., 0] + vis[..., 3])

        # Flag output Stokes I visibility if any of the contributions in the
        # sum is flags (XX or YY / RR or LL)
        flags = ms_reader.flags()
        stokes_i_flags = flags[..., (0, 3)].max(axis=-1)

        weights = ms_reader.weights()
        with warnings.catch_warnings():
            # Ignore division by zero warning, it all works out below even
            # for zero weights.
            warnings.simplefilter("ignore")

            # Stokes I visibilities = 1/2 * (vis_aa + vis_bb),
            # where (a, b) = (X, Y) or (R, L).
            # Weights are the inverse of variances, and variances add linearly,
            # hence this formula.
            wxx = weights[..., 0]
            wyy = weights[..., 3]
            stokes_i_weights = 4.0 / (1.0 / wxx + 1.0 / wyy)

        return cls(
            ms_reader.channel_frequencies(),
            stokes_i_flags,
            ms_reader.uvw(),
            stokes_i_vis,
            stokes_i_weights,
        )


def invert_measurement_set(
    ms_reader: MeasurementSetReader,
    num_pixels: int,
    pixel_size_asec: float,
    nthreads: int = os.cpu_count(),
) -> NDArray:
    """
    Invert the given measurement set, returning a dirty image.

    Parameters
    ----------
    ms_reader : MeasurementSetReader
        An object to read the measurement set, or a slice of it.
    num_pixels : int
        The number of pixels along one dimension of the square image.
    pixel_size_asec : float
        The size of each pixel in arcseconds.
    nthreads : int, optional
        The number of threads to use for computation. Default is the number of
        CPUs available.

    Returns
    -------
    NDArray
        A 2D array representing the dirty image.
    """
    gridding_input = StokesIGridderInput.from_measurement_set_reader(ms_reader)
    image, total_weight = ducc_invert(
        gridding_input, num_pixels, pixel_size_asec, nthreads=nthreads
    )
    return (1.0 / total_weight) * image


def ducc_invert(
    gridder_input: StokesIGridderInput,
    num_pixels: int,
    pixel_size_asec: float,
    nthreads: int = os.cpu_count(),
) -> tuple[NDArray, float]:
    """
    Invert the given data, returning an unscaled dirty image and
    its total gridding weight. The latter is the factor by which
    the image values must be divided in order to obtain fluxes.
    """
    pixel_size_lm = np.sin(np.radians(pixel_size_asec / 3600.0))
    effective_weights = gridder_input.effective_weights()

    # NOTE: DUCC ignores the `nthreads` argument in favour of either
    # DUCC0_NUM_THREADS or OMP_NUM_THREADS; dask sets the latter to 1 by
    # default. This ensures we get the desired number of threads.
    with set_env("DUCC0_NUM_THREADS", nthreads):
        image = ms2dirty(
            gridder_input.uvw,
            gridder_input.channel_frequencies,
            gridder_input.visibilities,
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
    return image, effective_weights.sum()


def worker_ducc_invert(
    gridder_input: StokesIGridderInput, num_pixels: int, pixel_size_asec: float
) -> tuple[NDArray, float]:
    """
    Same as ducc_invert, but runs on a dask worker and uses exactly its number
    of allocated threads.
    """
    nthreads = get_worker().state.nthreads
    return ducc_invert(
        gridder_input, num_pixels, pixel_size_asec, nthreads=nthreads
    )


def integrate_weighted_images(
    weighted_images: Iterable[tuple[NDArray, float]]
) -> NDArray:
    """
    Integrate images from chunks of the same measurement set, normalizing by
    the total weight.
    """
    images = [img for img, _ in weighted_images]
    weights = [weight for _, weight in weighted_images]
    return sum(images) / sum(weights)


def dask_invert_measurement_set(
    ms_reader: MeasurementSetReader,
    client: Client,
    num_pixels: int,
    pixel_size_asec: float,
    *,
    row_chunks: Optional[int] = 1,
    freq_chunks: Optional[int] = None,
) -> NDArray:
    """
    Invert the given measurement set, returning a dirty image. The work is
    distributed on a Dask cluster, along the rows and frequency channels of the
    input data.

    Parameters
    ----------
    ms_reader : MeasurementSetReader
        An object to read the measurement set, or a slice of it.
    client : Client
        A Dask client for scheduling distributed tasks.
    num_pixels : int
        The number of pixels along one dimension of the square image.
    pixel_size_asec : float
        The size of each pixel in arcseconds.
    row_chunks : int or None, optional
        The number of row chunks to partition the measurement set.
        Default is 1.
    freq_chunks : int or None, optional
        The number of frequency chunks to partition the measurement set.
        Default is one chunk per worker.

    Returns
    -------
    NDArray
        A 2D array representing the dirty image.
    """
    row_chunks = max(row_chunks, 1)
    if not freq_chunks:
        # One freq chunk per worker
        num_workers = len(client.scheduler_info())
        freq_chunks = min(ms_reader.num_channels(), num_workers)

    weighted_images = []
    for chunk in ms_reader.partition(row_chunks, freq_chunks):
        gridder_input = client.submit(
            StokesIGridderInput.from_measurement_set_reader, chunk
        )
        # NOTE: the custom resource is to ensure only one multithreaded
        # function call is running on a worker at any time.
        weighted_image = client.submit(
            worker_ducc_invert,
            gridder_input,
            num_pixels,
            pixel_size_asec,
            resources={"processing_slots": 1},
        )
        weighted_images.append(weighted_image)

    return client.submit(integrate_weighted_images, weighted_images).result()
