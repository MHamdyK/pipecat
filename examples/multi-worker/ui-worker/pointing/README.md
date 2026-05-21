# pointing

The UIWorker finds items on the page and points at them. A grid of
phone listings tall enough that several rows sit below the fold; the
user asks for one by name and the worker scrolls it into view and
flashes it.

## What it shows

- The `scroll_to` and `highlight` UI commands round-tripping
  end-to-end: the `UIWorker` emits them, the native bridge in
  `PipelineWorker` translates them to RTVI frames, and the client
  handler resolves the snapshot ref and acts on the live DOM.
- `ReplyToolMixin`'s visual fields — `reply(answer, scroll_to=...,
  highlight=[...])`. One tool call per turn; `answer` is required so
  the model can't forget the spoken reply.
- The `[offscreen]` state tag the client emits, and the LLM reading it
  to decide whether a scroll is needed before highlighting.

## What it adds vs. `hello-snapshot`

`hello-snapshot` proved the worker can *read* the page. This one proves
it can *act* on the page. Same skeleton (voice LLM in the main pipeline
delegating to a `UIWorker` via a `respond` job); the new parts are the
`scroll_to` / `highlight` commands and the client handlers for them.

## Run

Two terminals.

**Terminal 1 — bot:**

```bash
cd examples/multi-worker/ui-worker/pointing
uv run python bot.py
```

The bot starts on `http://localhost:7860`.

**Terminal 2 — client:**

```bash
cd examples/multi-worker/ui-worker/pointing/client
npm install            # one-time
npm run dev
```

Open `http://localhost:5173` and click **Connect**.

## What to try

The page renders 20 phone cards in a responsive grid; the bottom rows
usually land below the fold. Try:

- _"Where's the iPhone 17?"_ — the worker scrolls the card into view and
  flashes it.
- _"Scroll to the Pixel 9 Pro."_ — same flow, different ref.
- _"Which one is the Nothing phone?"_ — if it's already visible, the
  worker just highlights without scrolling.
- _"Which phones are from Google?"_ — a descriptive question; the worker
  highlights each phone it names.
- _"What's the cheapest one?"_ — the worker names and highlights it.

Watch the bot logs: each turn shows the main LLM calling
`answer_about_screen`, then the UIWorker's LLM emitting one `reply`
(scroll/highlight + the spoken answer).

## Requirements

- `OPENAI_API_KEY`
- `DEEPGRAM_API_KEY`
- `CARTESIA_API_KEY`

A `.env` in the example folder is the easiest way to set these (see
`examples/multi-worker/env.example`).

## What this example _doesn't_ show

Form filling (see `form-fill/`), selection-based deixis (see `deixis/`),
async task cards (see `async-tasks/`), or custom command handlers beyond
the standard `scroll_to` / `highlight`.
