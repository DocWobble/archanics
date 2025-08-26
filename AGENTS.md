# AGENTS.md — Prototype-First Game Compass (Terminal Only)

Codex must treat this file as its constant compass. The project’s only proof of progress is a
**running command-line prototype** that accumulates **green lights** (capabilities) one by one.
The prototype never contains business logic; it only calls public facades provided by the core game.

We are building a **turn/action-based, terminal-playable game** (genre-agnostic). No GUI windows.
Output is text and optional ASCII art. Assets are optional and appear conditionally if present.

---

## Core Principles

1) **Every task adds one capability**
   - Implement or extend a single, importable **public facade** in core (function/class).
   - Add a matching **probe** in `prototype.py` that calls the facade and verifies work via observable effects.
   - Prototype never fakes success; it only succeeds by exercising the core facade.

2) **Green lights prove real IO or compute**
   - A probe is complete only if it produces **observable evidence**: a file written, a JSON state updated,
     a deterministic transcript appended, or a terminal interaction round-trip performed.
   - `print("OK")` alone is not evidence. Success must follow from the facade’s real behavior.

3) **Prototype is the running changelog**
   - On success, the probe prints a line:  
     `"[OK] name(): one-line description of what this capability does and why it matters"`
   - Append structured metadata for each probe to `_artifacts/proto.json`
     (e.g., timestamp, seed, turn count, bytes written, durations).

4) **Stable facades, flexible internals**
   - Keep a thin, stable API surface between core and prototype. Internals can evolve freely.
   - If internals change, update the prototype to preserve the same green light.

5) **Bounded, deterministic, offline**
   - Each probe completes in a few seconds and remains deterministic for a given seed/input.
   - Use only local resources; localhost is allowed, external network is not.
   - If a capability cannot run (e.g., missing optional codec), print `[SKIP] reason` and continue.

6) **Behavioral invariants, not brittle snapshots**
   - Probes check properties: file exists and is non-empty, schema fields present, turn counter advanced,
     status code is 200 for a local call, ASCII frame width/height match, etc.
   - Avoid exact byte matches unless the artifact is intentionally canonical (e.g., replay hash).

7) **No regressions**
   - Existing green lights must remain green. Replacing a capability requires adding an equal or stronger light.

8) **Shadow comparisons for improvements**
   - When optimizing or adding an alternate path (e.g., faster loop, cached content),
     run **baseline and new path** on the same seed, log both, and print a comparative result.

9) **Assets are optional, never blocking**
   - If an art/sound asset exists under `assets/`, load and display/play it.
   - If not present, substitute a **procedural/ASCII placeholder** and still produce the artifact.

10) **Definition of Done**
    - A task is complete only if `prototype.py` (or `prototype.rs`) produces **one more `[OK]` line**
      than before, and that line corresponds to a real core facade invoked by the probe.

---

## Ignition Checklist (genre-agnostic, terminal-playable)

Codex should grow these lights in any sensible order. Each item is a facade the prototype can call and verify.

1. `game.init(seed)` → `[OK] init()`  
   Evidence: `_artifacts/sys.json` with seed/platform; deterministic RNG draw recorded.

2. `game.state_new()` → `[OK] new_state()`  
   Evidence: `_artifacts/state.json` with initial location/board/actors (empty or minimal is fine).

3. `game.turn_step(input)` → `[OK] turn()`  
   Evidence: `_artifacts/turns.log` appended; `_artifacts/state.json` shows valid state transition.

4. `game.describe()` → `[OK] describe()`  
   Evidence: `_artifacts/scene.txt` (text or ASCII art). If `assets/scene.txt` exists, include it; else synthesize.

5. `game.command_parse(line)` → `[OK] command()`  
   Evidence: `_artifacts/commands.jsonl` records `(input, parsed, result)`; invalid commands yield a clear message.

6. `game.render_frame()` → `[OK] render()`  
   Evidence: `_artifacts/frame.txt` or `_artifacts/frame.ppm` (ASCII or PPM). Dimensions asserted.

7. `game.replay_export()` / `game.replay_import()` → `[OK] replay()`  
   Evidence: `_artifacts/replay.json` round-trips to the same hash for the same seed and command list.

8. `game.assets_probe()` → `[OK] assets()`  
   Evidence: `_artifacts/assets.json` listing present/used assets; falls back to placeholders if missing.

9. `game.event_log()` → `[OK] events()`  
   Evidence: `_artifacts/events.jsonl` with structured entries (turn, actor, action, outcome).

10. `game.loop_script(script_path)` → `[OK] script_loop()`  
    Evidence: run a canned script of commands; produce `_artifacts/transcript.txt` and final state summary.

11. `game.loop_interactive(stdin)` (bounded) → `[OK] interactive()`  
    Evidence: simulated input stream (not human keyboard) processes N steps; transcript shows prompts and results.

12. `game.alt_path_enable(flag)` (optimization/variant) → `[OK] alt_path()`  
    Evidence: baseline vs alt timing or state equivalence; print comparison, keep both valid.

> These are **mechanics**, not genres. A text adventure, roguelike, puzzle, tactics, or sim can all satisfy them.

---

## Prototype Rules

- Keep each probe ≤ ~15 lines and run time ≤ a few seconds.  
- Probes live only in `prototype.*`, call only core facades, and write artifacts under `_artifacts/`.  
- Each probe prints exactly one line on success and a brief rationale.  
- Probes must accept a `--seed` and `--fast` flag where applicable to maintain determinism and bounds.

---

## Suggested Facade Shape (language-agnostic)

- `init(seed: int) -> SysInfo`
- `state_new() -> GameState`
- `command_parse(line: str) -> Parsed`
- `turn_step(parsed: Parsed) -> TurnResult`
- `describe(state: GameState) -> str`
- `render_frame(state: GameState, size: (w,h)) -> FrameInfo`
- `replay_export(state: GameState, path) -> ReplayInfo`
- `replay_import(path) -> GameState`
- `assets_probe(path="assets/") -> AssetsInfo`
- `event_log() -> Iterator[Event]`
- `loop_script(path) -> RunInfo`
- `alt_path_enable(flag: str) -> AltInfo`

The exact language and modules are up to Codex; the **names and responsibilities** should remain stable so the prototype can rely on them.

---

## What Counts as Evidence (examples)

- File exists and size > 0; JSON deserializes and contains expected keys.  
- Transcript contains the parsed command and a result token (e.g., `OK`, `INVALID`, `BLOCKED`).  
- ASCII/PPM frame has declared width/height; first line header matches.  
- Replay re-loads to an equivalent state hash.  
- Timing comparisons print baseline and alt durations with the same seed.

---

## Positive Defaults

- If input or assets are missing, generate a **minimal placeholder** and proceed.  
- If a capability cannot run in this environment, print `[SKIP]` with a specific reason and move on.  
- Prefer tiny, deterministic samples over realism. Add realism later behind an option.

---

## Definition of Done (strict)

A task is **Done** only when:
1) A new public facade exists or has been extended meaningfully,  
2) `prototype` invokes it and prints **one additional `[OK]` line**, and  
3) The probe writes its artifact(s) and prior lights remain green.

The shortest path to success is always to make the core facade work and let the prototype prove it.

