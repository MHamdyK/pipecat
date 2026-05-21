#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pointing — the UIWorker acts on the page to direct the user's attention.

The UIWorker composes ``ReplyToolMixin``, which exposes one bundled LLM
tool: ``reply(answer, scroll_to=None, highlight=None, ...)``. One tool
call per turn; the required ``answer`` argument is enforced by the API
schema so the model cannot forget the spoken reply.

When the user asks "where's the iPhone 17?", the UIWorker's LLM finds
the matching ref in the snapshot and emits one ``reply`` call with
``answer="Here's the iPhone 17."`` plus ``scroll_to`` and ``highlight``
set to that ref. The mixin dispatches the UI commands and completes the
job.

Architecture::

    Main worker (PipelineWorker, owns transport + RTVI):
      transport.in → STT → user_agg → LLM → TTS → transport.out → assistant_agg
        └── answer_about_screen(query) tool
              └── params.pipeline_worker.job("ui", name="respond", payload={query})

    PointingWorker (ReplyToolMixin + UIWorker):
      └── inherited: reply(answer, scroll_to=None, highlight=None, ...)

Run::

    uv run python bot.py

Then open the client at ``http://localhost:5173`` (see ``README.md``).

Requirements:

- OPENAI_API_KEY
- DEEPGRAM_API_KEY
- CARTESIA_API_KEY
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.job_context import JobError
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.workers.ui import ReplyToolMixin, UIWorker

load_dotenv(override=True)

MAIN_NAME = "main"

transport_params = {
    "daily": lambda: DailyParams(audio_in_enabled=True, audio_out_enabled=True),
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


VOICE_PROMPT = """\
You are the voice layer of a screen-aware assistant. A separate UI \
layer sees the page and writes the spoken reply.

For every user utterance that could involve the page, call \
``answer_about_screen`` with the user's request verbatim. The tool's \
response is the spoken reply, already TTS-ready.

Only respond directly for pure pleasantries (greetings, thanks, \
goodbyes). Keep direct replies to one short spoken sentence."""


# The UI wire-format guide (UI_STATE_PROMPT_GUIDE) is appended to the LLM's
# system instruction automatically by UIWorker, so this prompt only needs the
# app-specific behavior.
UI_PROMPT = """\
You help the user find and look at items on a long page of phone \
listings. The current ``<ui_state>`` block is in your context.

## Tool: reply

Every turn calls ``reply`` exactly once. One tool call per turn, no \
chaining.

``reply(answer, scroll_to=None, highlight=None)``:

- ``answer`` (REQUIRED): the spoken reply, plain language, one \
short sentence. No markdown, no symbols, no specs read aloud.
- ``scroll_to`` (OPTIONAL): a single snapshot ref like ``"e5"``. \
Set this when at least one phone you want to point at is tagged \
``[offscreen]`` in ``<ui_state>``. Pick the most relevant ref \
(typically the first match).
- ``highlight`` (OPTIONAL): a list of snapshot refs like ``["e5"]`` \
or ``["e5", "e8", "e47"]``. Each ref pulses on screen \
simultaneously. Use a single-element list for one phone, multi-element \
for several.

## Decision rules

**Highlight every phone you name in your answer.** This is the most \
reliable rule: whatever specific phones appear in the spoken text \
should also pulse on screen. One phone named → \
``highlight=["e5"]``. Three named → ``highlight=["e5", "e8", "e47"]``. \
None named (a generic answer like "I don't see any matches") → \
omit ``highlight``.

When any highlighted phone is tagged ``[offscreen]`` in \
``<ui_state>``, also set ``scroll_to`` to the ref of the most \
relevant one (typically the first in the list, or the one the user \
asked about most directly).

## Examples

- "Where's the iPhone 17?" (offscreen) → \
``reply(answer="Here's the iPhone 17.", scroll_to="e5", highlight=["e5"])``
- "Show me the Pixel 9 Pro." (offscreen) → \
``reply(answer="Here's the Pixel 9 Pro.", scroll_to="e14", highlight=["e14"])``
- "Tell me about the iPhone 17 Pro." (offscreen) → \
``reply(answer="It's Apple's 2025 flagship with a 120Hz ProMotion display and periscope zoom.", scroll_to="e8", highlight=["e8"])``
- "Which one is the Nothing phone?" (visible) → \
``reply(answer="This one, the Nothing Phone 3.", highlight=["e29"])``
- "Show me the Galaxy S25." (visible) → \
``reply(answer="Here's the Galaxy S25.", highlight=["e17"])``
- "Show me all the Apple phones." (all visible) → \
``reply(answer="Here are the three Apple phones.", highlight=["e5", "e8", "e47"])``
- "Highlight the Apple phones." (mix: e5 and e8 visible, e47 offscreen) → \
``reply(answer="Highlighting the Apple phones now.", scroll_to="e47", highlight=["e5", "e8", "e47"])``
- "Which phones are from Google?" → \
``reply(answer="The Pixel 9, Pixel 9 Pro, and Pixel 9a are from Google.", highlight=["e11", "e14", "e50"])``
- "What's the cheapest one?" (no specific phones named) → \
``reply(answer="The iPhone 16e is the most budget-friendly option here.", highlight=["e47"])``"""


class PointingWorker(ReplyToolMixin, UIWorker):
    """UIWorker that points at items using the bundled ``reply`` tool.

    Composes ``ReplyToolMixin``, which exposes a single
    ``reply(answer, scroll_to=None, highlight=None, ...)`` LLM tool. One
    tool call per turn; the required ``answer`` argument is enforced by
    the API schema so the model cannot forget the spoken reply (the
    failure mode chainable tools have with smaller models).

    ``keep_history=False`` (the ``UIWorker`` default) clears the LLM
    context at the start of every job, so each turn sees only the
    current ``<ui_state>`` and the user's query — stale snapshots from
    prior turns would otherwise contradict the current viewport.
    """

    def __init__(self):
        llm = OpenAILLMService(
            api_key=os.environ["OPENAI_API_KEY"],
            settings=OpenAILLMService.Settings(system_instruction=UI_PROMPT),
        )
        super().__init__("ui", llm=llm)


async def answer_about_screen(params: FunctionCallParams, query: str):
    """Ask the screen-aware UI worker to point at and answer about the page.

    Args:
        query (str): The user's request, passed verbatim.
    """
    logger.info(f"answer_about_screen('{query}')")
    try:
        async with params.pipeline_worker.job(
            "ui", name="respond", payload={"query": query}, timeout=10
        ) as t:
            pass
    except JobError as e:
        logger.warning(f"ui job failed: {e}")
        await params.result_callback("Something went wrong on my side.")
        return

    speak = (t.response or {}).get("speak")
    await params.result_callback(speak or "I'm not sure how to answer that.")


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting pointing bot")

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice=os.getenv("CARTESIA_VOICE_ID", "71a7ad14-091c-4e8e-a314-022ece01c121"),
        ),
    )
    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(system_instruction=VOICE_PROMPT),
    )
    llm.register_direct_function(answer_about_screen, cancel_on_interruption=False, timeout_secs=30)

    context = LLMContext(tools=ToolsSchema(standard_tools=[answer_about_screen]))
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            aggregators.user(),
            llm,
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )

    worker = PipelineWorker(
        pipeline,
        name=MAIN_NAME,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        context.add_message(
            {
                "role": "developer",
                "content": (
                    "Greet the user briefly. Tell them they can ask to find "
                    "or scroll to any phone on the list. One short sentence."
                ),
            }
        )
        await worker.queue_frame(LLMRunFrame())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.add_workers(PointingWorker(), worker)

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
