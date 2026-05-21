#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""UI worker: LLM worker that observes and drives a GUI app.

Composes the RTVI wire protocol for client UI events, accessibility-tree
snapshots, and server-emitted UI commands (in
``pipecat.processors.frameworks.rtvi.models``) with an opt-in
``ReplyToolMixin`` exposing the canonical bundled reply tool
(``answer`` + optional ``scroll_to`` / ``highlight`` / ...).

The RTVI⇄bus bridge that connects a ``UIWorker`` to the client is built
into ``PipelineWorker`` and is active whenever RTVI is enabled — there
is no decorator or bridge to wire up.
"""

from pipecat.bus.ui_messages import (
    BusUICommandMessage,
    BusUIEventMessage,
    BusUITaskCompletedMessage,
    BusUITaskGroupCompletedMessage,
    BusUITaskGroupStartedMessage,
    BusUITaskUpdateMessage,
)
from pipecat.workers.ui.ui_event_decorator import on_ui_event
from pipecat.workers.ui.ui_prompts import UI_STATE_PROMPT_GUIDE
from pipecat.workers.ui.ui_tools import ReplyToolMixin
from pipecat.workers.ui.ui_worker import UIWorker

# Built-in UI command payload models (Toast, Navigate, ScrollTo,
# Highlight, Focus, Click, SetInputValue, SelectText) live in
# ``pipecat.processors.frameworks.rtvi.models``. Import them from there
# directly.

__all__ = [
    "BusUICommandMessage",
    "BusUIEventMessage",
    "BusUITaskCompletedMessage",
    "BusUITaskGroupCompletedMessage",
    "BusUITaskGroupStartedMessage",
    "BusUITaskUpdateMessage",
    "ReplyToolMixin",
    "UIWorker",
    "UI_STATE_PROMPT_GUIDE",
    "on_ui_event",
]
