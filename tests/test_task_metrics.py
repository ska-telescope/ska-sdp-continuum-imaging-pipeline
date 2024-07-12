import json
import tempfile
from pathlib import Path

from ska_sdp_cip.task_metrics import TaskMetrics

# The data below is from an actual pipeline run, with the following
# modifications:
# - The keys that we don't need have been deleted
# - The start/stop timestamps have been manually modified
TASK_STREAM_DATA = [
    {
        "startstops": (
            {
                "action": "compute",
                "start": 0.0,
                "stop": 1.0,
            },
        ),
        "status": "OK",
        "key": "from_measurement_set_reader-ce5db27c84b49fb09f16262cb7c63b97",
        "worker": "tcp://127.0.0.1:37719",
    },
    {
        "startstops": (
            {
                "action": "compute",
                "start": 0.5,
                "stop": 1.5,
            },
        ),
        "status": "OK",
        "key": "from_measurement_set_reader-959219fda677c89f2231df55b34fbccc",
        "worker": "tcp://127.0.0.1:36221",
    },
    {
        "startstops": (
            {
                "action": "compute",
                "start": 2.0,
                "stop": 4.0,
            },
        ),
        "status": "OK",
        "key": "worker_ducc_invert-8eb7b120e745e99e2ea6b655faab8fe2",
        "worker": "tcp://127.0.0.1:37719",
    },
    {
        "startstops": (
            {
                "action": "compute",
                "start": 2.5,
                "stop": 4.5,
            },
        ),
        "status": "OK",
        "key": "worker_ducc_invert-a3bb0ad353f1ee2a7bc8a78156adc808",
        "worker": "tcp://127.0.0.1:36221",
    },
    {
        "startstops": (
            {
                "action": "transfer",
                "start": 5.0,
                "stop": 5.5,
                "source": "tcp://127.0.0.1:46153",
            },
            {
                "action": "compute",
                "start": 6.0,
                "stop": 7.0,
            },
        ),
        "status": "OK",
        "key": "integrate_weighted_images-a18572ecb467f210f42de9163ec9f158",
        "worker": "tcp://127.0.0.1:37719",
    },
]

EXPECTED_METRICS_DICTS = [
    {
        "duration": 1.0,
        "key": "from_measurement_set_reader-ce5db27c84b49fb09f16262cb7c63b97",
        "name": "from_measurement_set_reader",
        "start": 0.0,
        "status": "OK",
        "stop": 1.0,
        "worker": "tcp://127.0.0.1:37719",
    },
    {
        "duration": 1.0,
        "key": "from_measurement_set_reader-959219fda677c89f2231df55b34fbccc",
        "name": "from_measurement_set_reader",
        "start": 0.5,
        "status": "OK",
        "stop": 1.5,
        "worker": "tcp://127.0.0.1:36221",
    },
    {
        "duration": 2.0,
        "key": "worker_ducc_invert-8eb7b120e745e99e2ea6b655faab8fe2",
        "name": "worker_ducc_invert",
        "start": 2.0,
        "status": "OK",
        "stop": 4.0,
        "worker": "tcp://127.0.0.1:37719",
    },
    {
        "duration": 2.0,
        "key": "worker_ducc_invert-a3bb0ad353f1ee2a7bc8a78156adc808",
        "name": "worker_ducc_invert",
        "start": 2.5,
        "status": "OK",
        "stop": 4.5,
        "worker": "tcp://127.0.0.1:36221",
    },
    {
        "duration": 2.0,
        "key": "integrate_weighted_images-a18572ecb467f210f42de9163ec9f158",
        "name": "integrate_weighted_images",
        "start": 5.0,
        "status": "OK",
        "stop": 7.0,
        "worker": "tcp://127.0.0.1:37719",
    },
]


def test_task_metrics_parsing_and_json_export():
    """
    Parse metrics from dask `get_task_stream()` data, export to JSON file and
    check that the JSON output is as expected.
    """
    metrics = TaskMetrics(TASK_STREAM_DATA)

    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir) / "metrics.json"
        metrics.save_json(path)
        with open(path, "r") as file:
            metrics_dicts: list[dict] = json.load(file)

    assert metrics_dicts == EXPECTED_METRICS_DICTS
