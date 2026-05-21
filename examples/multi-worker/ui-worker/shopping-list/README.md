# shopping-list

Every voice turn drives the UI; speech is incidental. The user builds a
shopping list by talking — "add milk and eggs", "check off the bread",
"drop the last one", "what's left?" — and the list updates on screen
every turn. The assistant may also say something back. The two halves run
in parallel on separate workers; neither calls the other as a tool.

This is the **bridge-free** pattern for "every input acts, may speak."

## What it shows

- **A standard voice pipeline + a UIWorker, no bridging.** The voice
  layer is an ordinary `transport → STT → LLM → TTS` pipeline whose LLM
  just converses (no tools, never touches the list). Its user aggregator
  fires `on_user_turn_stopped` once per user turn, and that handler
  dispatches the transcript to the UIWorker as a `respond` job
  (`worker.job("ui", name="respond", payload={"query": transcript})`) —
  a bus message. The UIWorker does all the list work, silently, on its
  own worker, so its LLM output never reaches TTS.
- **Snapshot-driven action.** Before each UIWorker inference, the current
  `<ui_state>` is auto-injected (via the LLM's `on_before_process_frame`
  hook), so the worker resolves "the milk", "the last one", "the checked
  ones" against the live list.
- **Custom UI commands.** A single bundled `update_list` tool maps the
  request to `add_item` / `set_checked` / `remove_item` commands (plus the
  standard `highlight`, used to *show* what's left). Each item is a
  checkbox whose accessible name is the item text, so the snapshot exposes
  every item's label and checked state.

## What it adds vs. the prior demos

The other demos use the **request/response delegation** shape: a voice LLM
decides, via a tool call, when to consult the UIWorker. This one removes
that decision — **every** user turn goes to the UIWorker automatically,
and the voice LLM runs independently for conversation. The wiring is a
single `on_user_turn_stopped` handler instead of a delegating tool.

## Run

Two terminals.

**Terminal 1 — bot:**

```bash
cd examples/multi-worker/ui-worker/shopping-list
uv run python bot.py
```

The bot starts on `http://localhost:7860`.

**Terminal 2 — client:**

```bash
cd examples/multi-worker/ui-worker/shopping-list/client
npm install            # one-time
npm run dev
```

Open `http://localhost:5173` and click **Connect**.

## What to try

- _"Add milk and a dozen eggs."_ — both items appear; the voice
  acknowledges.
- _"Add bread, butter, and coffee."_ — three more land.
- _"Check off the bread."_ — it gets ticked and struck through.
- _"Actually, drop the butter."_ — removed.
- _"Clear the ones I've already got."_ — removes everything checked.
- _"What's left?"_ — the unchecked items pulse, and the voice tells you.
- _"Hi there!"_ — the voice greets you; the list is unchanged (the
  UIWorker no-ops).

You can also type in the box to add items by hand — they go through the
same snapshot path, so the worker sees them next turn.

## Requirements

- `OPENAI_API_KEY`
- `DEEPGRAM_API_KEY`
- `CARTESIA_API_KEY`

A `.env` in the example folder is the easiest way to set these (see
`examples/multi-worker/env.example`).

## What this example _doesn't_ show

The voice layer is **blind to the screen** — it converses and answers from
the dialogue so far, but it doesn't read the live list (only the UIWorker
sees the snapshot). For "what's left?" the UIWorker highlights the answer
visually while the voice gives a from-memory summary. The list is also not
persisted — refresh and it's gone.
