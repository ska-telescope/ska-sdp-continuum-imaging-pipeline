from __future__ import annotations

import collections.abc
import json
import os
from dataclasses import dataclass, field
from typing import Union


@dataclass
class Task:
    """
    Information about a single task that is returned by the dask scheduler at
    the end of processing.
    """

    key: str
    """
    Unique identifier of the task.
    """

    worker: str
    """
    Address of the worker than ran the task.
    """

    status: str
    """
    Return status.
    """

    start: float
    """
    Start time as a UNIX timestamp.
    """

    stop: float
    """
    Stop time as a UNIX timestamp.
    """

    name: str = field(init=False)
    """
    Generic name of the task, usually the name of the underlying function that
    was called. Obtained by splitting the key on the last "-" character and
    discarding what's after.
    """

    duration: float = field(init=False)
    """
    Total duration in seconds, including input transfer time and processing
    time.
    """

    def __post_init__(self) -> None:
        self.name = self.key.rsplit("-", maxsplit=1)[0]
        self.duration = self.stop - self.start

    def as_dict(self) -> dict:
        """
        Convert to dictionary.
        """
        keys = ["key", "worker", "status", "start", "stop", "name", "duration"]
        return {key: getattr(self, key) for key in keys}

    @classmethod
    def _from_task_stream_entry(cls, entry: dict) -> Task:
        """
        Convert from one of the dictionaries in the list returned by
        dask's `get_task_stream()`.
        """
        # This is a tuple of dicts, each of them has three keys:
        # "action", "start", "stop"
        # "action" is a string, possibilities include: "compute", "transfer"
        # "start" and "stop" are timestamps
        startstops: tuple[dict] = entry["startstops"]
        start = min(item["start"] for item in startstops)
        stop = max(item["stop"] for item in startstops)
        return Task(
            key=entry["key"],
            worker=entry["worker"],
            status=entry["status"],
            start=start,
            stop=stop,
        )


class TaskMetrics(collections.abc.Sequence[Task]):
    """
    Provides parsing of task info from the output of dask's
    `get_task_stream()` and exporting the data to JSON for later analysis.

    The JSON output can be loaded using e.g. `pandas.read_json()`.
    """

    def __init__(self, task_stream_data: list[dict]) -> None:
        """
        Parse from dask Client task stream data. See example below.

        Example
        -------

        ```
        from dask.distributed import get_task_stream

        with get_task_stream() as ts:
            # dask computation

        task_list = TaskList(ts.data)
        ```
        """
        self._task_list = list(
            map(Task._from_task_stream_entry, task_stream_data)
        )

    def __len__(self) -> int:
        return len(self._task_list)

    def __getitem__(self, index: int) -> Task:
        return self._task_list[index]

    def to_json(self, **kwargs) -> str:
        """
        Convert to JSON representation; keyword arguments are passed to
        `json.dumps()`.
        """
        return json.dumps([task.as_dict() for task in self], **kwargs)

    def save_json(self, path: Union[str, os.PathLike], **kwargs) -> None:
        """
        Save JSON representation to given file path; keyword arguments are
        passed to `json.dumps()`.
        """
        with open(path, "w") as file:
            file.write(self.to_json(**kwargs))
