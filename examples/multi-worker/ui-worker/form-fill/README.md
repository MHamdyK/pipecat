# form-fill

The UIWorker fills form inputs and clicks buttons by voice. The page
renders a job application with text fields, a textarea, checkboxes, and
a submit button. Tell the worker your name, email, and the rest; when
you're ready, say "submit."

## What it shows

- The **state-changing actions**: `set_input_value` for writing into
  inputs, `click` for checkboxes and submit. Both are bundled into the
  same `ReplyToolMixin` that pointing and deixis use — `fills` is a list
  of `{"ref", "value"}` so the LLM can fill several fields in one turn
  ("my name is John Smith" fills first AND last name in one call), and
  `click` is a list so checkboxes and submit run in order.
- That `FormWorker` is a one-line composition:
  `class FormWorker(ReplyToolMixin, UIWorker)`. Same shape as pointing
  and deixis; the visual fields (`highlight`, `select_text`) just stay
  `null` here, and the prompt steers the LLM toward `fills` / `click`.

## What it adds vs. `pointing` and `deixis`

Those exercise the visual / attention-pointing fields of `reply`. This
one exercises the state-changing fields (`fills`, `click`). Same
composition, same mixin — different fields per turn, driven by the
prompt.

## Run

Two terminals.

**Terminal 1 — bot:**

```bash
cd examples/multi-worker/ui-worker/form-fill
uv run python bot.py
```

The bot starts on `http://localhost:7860`.

**Terminal 2 — client:**

```bash
cd examples/multi-worker/ui-worker/form-fill/client
npm install            # one-time
npm run dev
```

Open `http://localhost:5173` and click **Connect**.

## What to try

- _"My name is John Smith."_ — fills first and last name in one call.
- _"My email is john at gmail dot com."_ — converts the spoken form to
  `mark@daily.co` and fills the email field.
- _"I have five years of experience and I love working on real-time
  voice agents."_ — fills two fields in one call.
- _"Agree to the terms."_ — clicks the terms checkbox.
- _"What have I entered so far?"_ — reads back current values from
  `<ui_state>` (no fills, no clicks).
- _"Submit it."_ — clicks submit. If terms isn't ticked yet, the worker
  clicks both in order: terms, then submit.

## Requirements

- `OPENAI_API_KEY`
- `DEEPGRAM_API_KEY`
- `CARTESIA_API_KEY`

A `.env` in the example folder is the easiest way to set these (see
`examples/multi-worker/env.example`).

## What this example _doesn't_ show

Selection-based deixis (see `deixis/`) or async task cards (see
`async-tasks/`).
