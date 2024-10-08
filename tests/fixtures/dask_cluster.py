# pylint: disable=redefined-outer-name
import dask.config
import pytest
from dask.distributed import Client, LocalCluster


@pytest.fixture(scope="session")
def dask_cluster() -> LocalCluster:
    """
    Dask cluster for the test session.
    """
    # NOTE: Need to make workers non-daemonic because we want to parallelize
    # some code on the dask workers using the multiprocessing module.
    # See:
    # https://github.com/dask/distributed/issues/2142
    # https://github.com/dask/distributed/issues/2718

    with dask.config.set({"distributed.worker.daemon": False}):
        return LocalCluster(
            "dask_cluster",
            n_workers=2,
            threads_per_worker=1,
            resources={"processing_slots": 1},
        )


@pytest.fixture(scope="session")
def dask_client(dask_cluster: LocalCluster) -> Client:
    """
    Dask client for the test session.
    """
    return Client(dask_cluster.scheduler_address)
