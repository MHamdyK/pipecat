#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Bus carriers for the UI Worker protocol.

These dataclasses are the on-the-bus shape that ``UIWorker`` (see
``pipecat.workers.ui``) and the native RTVI⇄bus bridge in
``PipelineWorker`` exchange. They are NOT the on-the-wire format the
client sees; that lives in ``pipecat.processors.frameworks.rtvi.models``
(``UIEventMessage``, ``UICommandMessage``, ``UITaskMessage``, ...). The
bridge translates between the two.

- ``BusUIEventMessage`` and ``BusUICommandMessage`` carry client
  events and server commands respectively.
- ``BusUITaskGroupStartedMessage``, ``BusUITaskUpdateMessage``,
  ``BusUITaskCompletedMessage``, and ``BusUITaskGroupCompletedMessage``
  carry the four phases of a user-facing task group's lifecycle. The
  "task" naming here mirrors the fixed RTVI ``ui-task`` wire protocol;
  the server-side mechanism that drives them is a job group (see
  ``UIWorker.user_job_group``).

The carriers live in the ``bus`` layer (rather than alongside
``UIWorker``) because both ``PipelineWorker`` (in ``pipecat.pipeline``)
and ``UIWorker`` (in ``pipecat.workers``) reference them, and
``pipeline`` must not import from ``workers``.
"""

from dataclasses import dataclass
from typing import Any

from pipecat.bus.messages import BusDataMessage

#: Internal ``event_name`` used by the UI bridge when republishing a
#: ``ui-snapshot`` wire message onto the bus as a
#: ``BusUIEventMessage``. ``UIWorker``'s bus dispatch matches on this
#: name to route the snapshot into ``_latest_snapshot`` storage. The
#: leading double underscore marks the name as internal so app-defined
#: ``@on_ui_event`` handlers can't collide with it.
_UI_SNAPSHOT_BUS_EVENT_NAME = "__ui_snapshot"

#: Internal ``event_name`` used by the UI bridge when republishing a
#: ``ui-cancel-task`` wire message onto the bus as a
#: ``BusUIEventMessage``. ``UIWorker``'s bus dispatch matches on this
#: name to route to ``cancel_job_group``. Internal; not part of the
#: public wire format.
_UI_CANCEL_TASK_BUS_EVENT_NAME = "__cancel_task"


@dataclass
class BusUIEventMessage(BusDataMessage):
    """A UI event sent from the client to a server-side worker.

    Emitted by the native UI bridge in ``PipelineWorker`` when the
    client dispatches an event via
    ``PipecatClient.sendUIEvent(event, payload)``. ``UIWorker``
    subclasses dispatch these to ``@on_ui_event(name)`` handlers.

    Parameters:
        event_name: App-defined event name.
        payload: App-defined payload. Schemaless by design.
    """

    event_name: str = ""
    payload: Any = None


@dataclass
class BusUICommandMessage(BusDataMessage):
    """A UI command sent from a server-side worker to the client.

    Published by ``UIWorker.send_command(name, payload)``. The native
    UI bridge in ``PipelineWorker`` translates this to an
    ``RTVIUICommandFrame(command=command_name, payload=payload)`` and
    pushes it through the pipeline.

    Parameters:
        command_name: App-defined command name.
        payload: App-defined payload (already a plain dict by the time
            it lands on the bus).
    """

    command_name: str = ""
    payload: Any = None


# ---------------------------------------------------------------------------
# UI task lifecycle
# ---------------------------------------------------------------------------


@dataclass
class BusUITaskGroupStartedMessage(BusDataMessage):
    """A user-facing task group has been dispatched.

    Published by ``UIWorker.user_job_group(...)`` on entry. The bridge
    forwards it to the client as a ``ui-task`` envelope with
    ``kind = "group_started"``.

    Parameters:
        task_id: Shared task identifier for the group.
        agents: Names of the workers the work was dispatched to.
        label: Optional human-readable label for the group.
        cancellable: Whether the client may request cancellation.
        at: Epoch milliseconds when the group started.
    """

    task_id: str = ""
    agents: list[str] | None = None
    label: str | None = None
    cancellable: bool = True
    at: int = 0


@dataclass
class BusUITaskUpdateMessage(BusDataMessage):
    """Per-task progress for a user-facing task group.

    Forwarded by the ``UIWorker`` whenever a worker emits a
    ``BusJobUpdateMessage`` whose ``job_id`` matches a registered user
    task group. The bridge forwards to the client as a ``ui-task``
    envelope with ``kind = "task_update"``.

    Parameters:
        task_id: The shared task identifier.
        agent_name: The worker that produced the update.
        data: The worker's update payload, forwarded verbatim.
        at: Epoch milliseconds when the update was emitted on the bus.
    """

    task_id: str = ""
    agent_name: str = ""
    data: Any = None
    at: int = 0


@dataclass
class BusUITaskCompletedMessage(BusDataMessage):
    """A worker in a user-facing task group has completed.

    Forwarded by the ``UIWorker`` whenever a worker's
    ``BusJobResponseMessage`` arrives for a registered user task group.
    The bridge forwards to the client as a ``ui-task`` envelope with
    ``kind = "task_completed"``.

    Parameters:
        task_id: The shared task identifier.
        agent_name: The worker that produced the response.
        status: Completion status as a string (``JobStatus`` value).
        response: The worker's response payload.
        at: Epoch milliseconds when the response was received.
    """

    task_id: str = ""
    agent_name: str = ""
    status: str = ""
    response: Any = None
    at: int = 0


@dataclass
class BusUITaskGroupCompletedMessage(BusDataMessage):
    """A user-facing task group has completed.

    Published when ``UIWorker.user_job_group(...)`` exits, after every
    worker has responded (or the group has been cancelled). The bridge
    forwards to the client as a ``ui-task`` envelope with
    ``kind = "group_completed"``.

    Parameters:
        task_id: The shared task identifier.
        at: Epoch milliseconds when the group completed.
    """

    task_id: str = ""
    at: int = 0
