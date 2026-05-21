#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User-facing job group context.

Wraps ``JobGroupContext`` so the work it dispatches is also surfaced
to the UI client through the UI Worker protocol. Apps reach this via
``UIWorker.user_job_group(...)`` rather than constructing it directly.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from pipecat.bus.ui_messages import (
    BusUITaskGroupCompletedMessage,
    BusUITaskGroupStartedMessage,
)
from pipecat.pipeline.job_context import JobGroupContext

if TYPE_CHECKING:
    from pipecat.workers.ui.ui_worker import UIWorker


class UserJobGroupContext(JobGroupContext):
    """Job group whose lifecycle is forwarded to the UI client.

    Behaves exactly like ``JobGroupContext`` for the dispatching code.
    Additionally, on enter the context registers the group with its
    parent ``UIWorker`` and publishes a ``BusUITaskGroupStartedMessage``.
    The worker forwards any subsequent ``BusJobUpdateMessage`` /
    ``BusJobResponseMessage`` whose ``job_id`` matches a registered
    group as ``BusUITaskUpdateMessage`` / ``BusUITaskCompletedMessage``.
    On exit the context publishes ``BusUITaskGroupCompletedMessage`` and
    deregisters.

    Workers don't need to know about the UI surface: any
    ``send_job_update`` they emit against the group's ``job_id`` is
    forwarded automatically.

    Example::

        async with self.user_job_group(
            "researcher_a", "researcher_b",
            payload={"query": query},
            label=f"Research: {query}",
            cancellable=True,
        ) as tg:
            async for event in tg:
                ...
            results = tg.responses
    """

    def __init__(
        self,
        worker: UIWorker,
        worker_names: tuple[str, ...],
        *,
        name: str | None = None,
        payload: dict | None = None,
        timeout: float | None = None,
        cancel_on_error: bool = True,
        label: str | None = None,
        cancellable: bool = True,
    ):
        """Initialize the UserJobGroupContext.

        Args:
            worker: The parent ``UIWorker`` that owns this job group.
            worker_names: Names of the workers to send the job to.
            name: Optional job name for routing to named ``@job``
                handlers on the workers.
            payload: Optional structured data describing the work.
            timeout: Optional timeout in seconds covering both the
                ready-wait and job execution.
            cancel_on_error: Whether to cancel the group if a worker
                errors. Defaults to True.
            label: Optional human-readable label surfaced to the
                client (e.g. ``"Research: Radiohead"``). The client UI
                uses it to title the in-flight task card.
            cancellable: Whether the client may request cancellation
                of this group via the reserved ``__cancel_task`` event.
                Defaults to True.
        """
        super().__init__(
            worker,
            worker_names,
            name=name,
            payload=payload,
            timeout=timeout,
            cancel_on_error=cancel_on_error,
        )
        self._ui_worker = worker
        self._label = label
        self._cancellable = cancellable

    @property
    def label(self) -> str | None:
        """The group's human-readable label, if any."""
        return self._label

    @property
    def cancellable(self) -> bool:
        """Whether the client may request cancellation."""
        return self._cancellable

    async def __aenter__(self) -> UserJobGroupContext:
        await super().__aenter__()
        job_id = self.job_id
        self._ui_worker._register_user_job_group(
            job_id=job_id,
            worker_names=list(self._worker_names),
            label=self._label,
            cancellable=self._cancellable,
        )
        await self._ui_worker.send_bus_message(
            BusUITaskGroupStartedMessage(
                source=self._ui_worker.name,
                target=None,
                task_id=job_id,
                agents=list(self._worker_names),
                label=self._label,
                cancellable=self._cancellable,
                at=int(time.time() * 1000),
            )
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        job_id = self._group.job_id if self._group else None
        try:
            return await super().__aexit__(exc_type, exc_val, exc_tb)
        finally:
            if job_id:
                self._ui_worker._unregister_user_job_group(job_id)
                await self._ui_worker.send_bus_message(
                    BusUITaskGroupCompletedMessage(
                        source=self._ui_worker.name,
                        target=None,
                        task_id=job_id,
                        at=int(time.time() * 1000),
                    )
                )
