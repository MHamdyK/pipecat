#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Form-fill — the UIWorker changes form state by voice.

The page renders a job-application form. The user dictates field values
("my name is John Smith", "email is john at gmail dot com"), checks
boxes ("I agree to the terms"), and submits ("send it"). The UIWorker
maps each spoken value to the right input ref and writes it.

Same skeleton as ``pointing`` / ``deixis``. ``FormWorker`` composes
``ReplyToolMixin``: the ``reply(answer, scroll_to, fills, click)`` bundle
covers the state-changing actions — ``fills`` writes input values
(many at once), ``click`` toggles checkboxes and presses submit.

Architecture::

    Main worker (PipelineWorker, owns transport + RTVI):
      transport.in → STT → user_agg → LLM → TTS → transport.out → assistant_agg
        └── answer_about_screen(query) tool
              └── params.pipeline_worker.job("ui", name="respond", payload={query})

    FormWorker (ReplyToolMixin + UIWorker):
      └── inherited: reply(answer, scroll_to, fills, click)

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
You are the voice layer of a form-fill assistant. A separate UI \
layer sees the form and writes the spoken reply.

For every user utterance involving the form (filling fields, \
checking boxes, submitting), call ``answer_about_screen`` with the \
user's request verbatim. The tool's response is the spoken reply, \
already TTS-ready.

Only respond directly for pure pleasantries (greetings, thanks, \
goodbyes). Keep direct replies to one short spoken sentence."""


# The UI wire-format guide (UI_STATE_PROMPT_GUIDE) is appended to the LLM's
# system instruction automatically by UIWorker, so this prompt only needs the
# app-specific behavior.
UI_PROMPT = """\
You help the user fill out a job application form by voice. The \
current ``<ui_state>`` block is in your context. Each input has a \
ref (e.g. ``e5``) and a label. Use the labels to decide which input \
gets which value.

## Tool: reply

Every turn calls ``reply`` exactly once. One tool call per turn.

``reply(answer, scroll_to=None, fills=None, click=None)``:

- ``answer`` (REQUIRED): a short spoken reply confirming what you \
did or asking for missing info. One short sentence. Plain language.
- ``scroll_to`` (OPTIONAL): a single snapshot ref. Use when a field \
the user wants to see is tagged ``[offscreen]``.
- ``fills`` (OPTIONAL): a list of ``{"ref": "eN", "value": "..."}`` \
objects. Each entry writes ``value`` into the input at ``ref``. \
You can fill many fields in one turn (e.g. first name + last name \
+ email when the user says "my name is John Smith, mark at \
daily dot co").
- ``click`` (OPTIONAL): a list of refs to click. Use for \
checkboxes (terms, newsletter) and the submit button. Order matters: \
click checkboxes before submit.

## Decision rules

- **User dictates field values** → match each value to the input \
whose label fits, set ``fills``, confirm in ``answer``.
- **User says "check" / "agree" / "yes" for a checkbox** → resolve \
the matching checkbox ref, set ``click=[ref]``.
- **User says "submit" / "send it"** → confirm any required fields \
are filled (especially the terms checkbox if needed), then \
``click=[submit_ref]``. If terms isn't checked yet but the user said \
submit, click both: ``click=[terms_ref, submit_ref]``.
- **User asks "what have I entered?" / "what's left?"** → read the \
current values from ``<ui_state>`` (the walker emits each input's \
current value), summarize in ``answer``. No fills, no clicks.

## Spelling and disambiguation

When the user says something like "john at gmail dot com", convert \
to ``mark@daily.co``. "five five five one two three four" → \
``5551234``. "five years" → ``5``. Don't read these conversions \
back to the user verbatim; just confirm naturally ("got it, your \
email is mark@daily.co").

## Examples

(refs are illustrative; use the actual refs from the current \
``<ui_state>``)

- "My name is John Smith." → \
``reply(answer="Got it, John Smith.", fills=[{"ref":"e5","value":"Mark"}, {"ref":"e7","value":"Backman"}])``
- "Email is john at gmail dot com." → \
``reply(answer="Email saved.", fills=[{"ref":"e9","value":"mark@daily.co"}])``
- "I have five years of experience and I love working on \
real-time voice agents." → \
``reply(answer="Five years and your interest noted.", fills=[{"ref":"e15","value":"5"}, {"ref":"e17","value":"I love working on real-time voice agents."}])``
- "I agree to the terms." → \
``reply(answer="Terms accepted.", click=["e22"])``
- "Submit it." (terms not yet checked) → \
``reply(answer="Submitting.", click=["e22","e26"])``
- "What have I entered?" → \
``reply(answer="John Smith, mark@daily.co, 5 years experience. The cover letter and terms aren't done yet.")``"""


class FormWorker(ReplyToolMixin, UIWorker):
    """UIWorker that fills form fields and toggles controls via ``reply``.

    Composes ``ReplyToolMixin``, which exposes a single
    ``reply(answer, scroll_to=None, fills=None, click=None, ...)`` LLM
    tool. ``fills`` writes values into inputs (many in one turn) and
    ``click`` toggles checkboxes / presses submit — the state-changing
    half of the standard action set.
    """

    def __init__(self):
        llm = OpenAILLMService(
            api_key=os.environ["OPENAI_API_KEY"],
            settings=OpenAILLMService.Settings(system_instruction=UI_PROMPT),
        )
        super().__init__("ui", llm=llm)


async def answer_about_screen(params: FunctionCallParams, query: str):
    """Ask the screen-aware UI worker to fill the form / answer about it.

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
    logger.info("Starting form-fill bot")

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
                    "Greet the user briefly. Tell them they can dictate field "
                    "values and you'll fill them in. One short sentence."
                ),
            }
        )
        await worker.queue_frame(LLMRunFrame())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.add_workers(FormWorker(), worker)

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
