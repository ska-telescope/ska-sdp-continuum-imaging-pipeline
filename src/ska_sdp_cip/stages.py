import os
from contextlib import contextmanager
from typing import Any, Iterable, Optional

import numpy as np
from dask import delayed
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
    yield
    if previous_value is None:
        os.environ.pop(name)
    else:
        os.environ[name] = previous_value


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
    image, total_weight = measurement_set_to_weighted_image(
        ms_reader, num_pixels, pixel_size_asec, nthreads=nthreads
    )
    return (1.0 / total_weight) * image


def measurement_set_to_weighted_image(
    ms_reader: MeasurementSetReader,
    num_pixels: int,
    pixel_size_asec: float,
    nthreads: int = os.cpu_count(),
) -> tuple[NDArray, float]:
    """
    Invert the given measurement set, returning an unscaled dirty image and
    its total gridding weight. The latter is the factor by which
    the image values must be divided in order to obtain fluxes.
    """
    pixel_size_lm = np.sin(np.radians(pixel_size_asec / 3600.0))

    mask = np.logical_not(ms_reader.stokes_i_flags())
    weights = ms_reader.stokes_i_weights()
    effective_weights = weights * mask

    # NOTE: DUCC ignores the `nthreads` argument in favour of either
    # DUCC0_NUM_THREADS or OMP_NUM_THREADS; dask sets the latter to 1 by
    # default. This ensures we get the desired number of threads.
    with set_env("DUCC0_NUM_THREADS", nthreads):
        image = ms2dirty(
            ms_reader.uvw(),
            ms_reader.channel_frequencies(),
            ms_reader.stokes_i_visibilities(),
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


def worker_measurement_set_to_weighted_image(
    ms_reader: MeasurementSetReader, num_pixels: int, pixel_size_asec: float
) -> tuple[NDArray, float]:
    """
    Same as measurement_set_to_weighted_image, but runs on a dask worker and
    uses exactly its number of allocated threads.
    """
    nthreads = get_worker().state.nthreads
    return measurement_set_to_weighted_image(
        ms_reader, num_pixels, pixel_size_asec, nthreads=nthreads
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

    weighted_images = [
        delayed(worker_measurement_set_to_weighted_image)(
            chunk, num_pixels, pixel_size_asec
        )
        for chunk in ms_reader.partition(row_chunks, freq_chunks)
    ]
    integrated_image = delayed(integrate_weighted_images)(weighted_images)
    # NOTE: the custom resource is to ensure only one multithreaded function
    # call is running on a worker at any time.
    return client.compute(
        integrated_image, resources={"processing_slots": 1}
    ).result()
