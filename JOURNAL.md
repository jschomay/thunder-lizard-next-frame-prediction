# Thunder Lizard Frame Prediction — Journal

A dated log of ideas, decisions, and results. Testable ideas are listed in [HYPOTHESES.md](HYPOTHESES.md).

Goal: train a model that predicts the next frame of [Thunder Lizard](https://github.com/jschomay/thunder-lizard) from the previous frame(s) and the key input. Then "play the game inside the model." This is a learning project and a portfolio piece.

---

## 2026-10-01 — Kickoff

### What we learned about the game
- **Real-time, not turn-based.** `mainLoop` ticks every 70 ms, lava every 50 ms, and water every 200 ms. The player moves while a key is held, and movement speed depends on dino level. **Decision:** one training "step" = one main-loop tick, plus the key held during it (or NONE).
- **Viewport:** 45×45 (39×39 when 3 or more dinos are near), following the player across a 250×250 map. Cells scrolling in from the edge can't be predicted from the frame alone.
- **Lots of cosmetic randomness.** Water and ocean glyphs re-roll every tick, terrain glyph variants are chosen at random, and tail glyphs flicker.
- **Real randomness:** lava spread is random. Dino AI (awareness, pursue, flee) runs on hidden counters.
- `ROT.RNG` is seedable, so runs can be made reproducible.
- Logic is tied to `document`, Howler, and `ROT.Display`. The headless harness needs to stub these out.

### Decision: predict game *state*, render *visuals* afterward
The renderer is deterministic given the state (apart from cosmetic randomness). So the model predicts a **semantic state grid**, and a separate renderer turns it back into the full ASCII look:
- **Dinos:** predict only the body cell, with **direction** (4) and **level** (5). The renderer draws the 5-cell dino (head, legs, tail) from those. The model doesn't spend capacity learning drawing rules.
- **Terrain:** predict a discrete category (OCEAN, WATER, DIRT, GRASS, SHRUB, JUNGLE, LAVA, BURNT). The renderer picks glyph variants and colors, and animates water shimmer.
- **Consequence:** the headless harness emits the state grid directly instead of rendered glyphs, so training data and model output use the same format. A later upgrade could predict rendered frames for comparison.
- v1 simplifications: lock zoom at 45×45, drop score popups.

### Approaches to compare (we want to try all three)
| # | Approach | What | Fit |
|---|---|---|---|
| A | **CNN / U-Net** | Convolutions over the grid. Each output cell sees a local neighborhood, which grows with depth and downsampling. Input: k past state grids, one-hot encoded, plus the key. Output: per-cell class probabilities. | Best first fit. The game's rules are mostly local (one-cell moves, lava spreading to neighbors). Small (1–5M params), trains on a Mac in minutes to an hour. |
| B | **Transformer over cells/patches** | Cells (or small patches) become tokens, and attention lets any cell look at any other. | Better for long-range interactions such as dinos noticing the player about 10 cells away. Needs more data and compute. The natural v2. |
| C | **Fine-tuned small LLM on ASCII text** (e.g. SmolLM or Qwen from Hugging Face) | Serialize each frame plus the key as text and train the LLM to write the next frame's text. | Interesting as a comparison, but a poor fit: about 2k+ tokens per frame, generated one at a time, so it's slow. Its pretrained knowledge is about language, not grids. Expected to lose, and confirming that is a result in itself. |

Out of scope for now: diffusion world models (GameNGen, DIAMOND).

**Where models come from:** A and B are architectures we write in PyTorch and train **from scratch**, since nothing pretrained knows this game. Hugging Face is used for C (the pretrained LLM) and for **publishing**: dataset (`datasets`), weights (`huggingface_hub`), and a public demo (Space).

### Staged plan (cheap validation first)
1. **Player only:** fixed map, no dinos, no lava. Learn movement and scrolling. Should get close to 100% accuracy. Validates the plumbing.
2. **+ Lava:** learn stochastic spread (probabilities, not certainty).
3. **+ A few dinos (about 5):** infer hidden state and momentum from past frames.
4. **Full game + autoregressive "dream" play** in the browser (ONNX).

### Metrics
- Baseline: **copy the last frame**. Most cells don't change, so raw accuracy is misleading.
- Report accuracy on **changed cells** and on **dino cells**, plus quality over N-step rollouts (drift).

### Open questions
- Reuse the same model across stages? See H4.
- How many past frames (k) are needed? See H3.

## 2026-10-01 — Whiteboard review notes

- **Tagline:** "Playing the game inside a neural net." Avoid the word "dream."
- **Training data source (open):** expected to need human play. Plan: a **scripted heuristic bot** for volume, **human play recordings** as a smaller realistic set, and held-out evaluation, mixed. Concern: *distribution shift*. The model only learns situations that appear in its data, so a random bot that never levels up or reaches the map edge leaves those cases unlearned. See H6.
- **Evaluation:** automatic metrics (changed-cell accuracy, rollout drift) plus **human "feel" judgments**. Plan a blind side-by-side test: "which one is the real game?" GameNGen used human raters the same way. See H7.
- **Compute:** Mac MPS for small models. **Colab** for bigger runs (transformer, LLM fine-tune).
- **Want next:** a page on precedents (World Models, GameNGen, DIAMOND, Genie): how they work and how they compare to this project. Each milestone needs its own detailed write-up.

## 2026-10-05 — Milestone 1: headless harness

### What was built
- `game/` is a **git submodule** of the game repo, on a local branch `headless` (not pushed). It has two changes:
  1. `lavaTick()` extracted from the lava timer loop, so the harness can step lava manually.
  2. **Perf fix:** the systems called `defineQuery` on every tick, which registers a new bitECS query each frame. Moving the queries to module level made one step go from **12 ms to 0.1 ms** (~120×). This bug also affects the browser game, which slows down the longer it runs; consider upstreaming the fix.
- `harness/headless.ts`: `HeadlessGame` stubs out the DOM, sound, and display, disables the timer loops, and provides `step(key)` (one 70 ms main tick, with lava at its real 50 ms rate ≈ 1.4 lava ticks per step) and `observe()` (a 45×45 state grid centered on the player, zoom locked).
- `harness/record.ts`: a sticky random-walk bot records episodes to `data/stage<N>/ep_<seed>/{obs.npy, actions.npy, meta.json}`.
  - `obs` has shape `[T+1, 2, 45, 45]`. Layer 0 is the terrain class (8). Layer 1 is the entity class (42: player and NPC × level 1–5 × direction 4, plus skeleton). The vocabulary is fixed across all stages.
  - Run: `cd game && npx vite-node --root .. ../harness/record.ts -- --stage 1 --episodes 200 --ticks 500`
- **Determinism verified:** a given seed produces byte-identical data whether it's recorded alone or after other episodes. bitECS global state (entity ids and component arrays) is reset between episodes.
- Throughput: about 1,200 ticks/s for stage 1. Setup (about 0.4 s for map generation) dominates. Current data: stage 1 has 100k ticks; stages 2–4 have about 10k each.

### Finding: the input → movement delay is hidden state, even in stage 1
Sanity check on 50 stage-1 episodes:

| key | avg changed cells | ticks with no change |
|---|---|---|
| NONE | 0.1% | 97% |
| UP/RIGHT/DOWN/LEFT | 1.4–1.9% | 56–66% |

A level-1 player moves only **every 2nd tick**, and a key press is **buffered** until the next move tick. *(Correction: this first said "every 3rd tick", inferred from `turnsToSkip = 2` in the code. Measuring scroll in the recorded frames shows a strict every-other-tick pattern in both bot and human data: `.D.D.D.D`. Why the code yields 2 rather than 3 isn't confirmed yet.)* So about 60% of ticks with a key held change nothing, and about 3% of NONE ticks still move (a buffered press). **From a single frame, the model can't know whether this tick is a move tick.** It needs history to infer the movement phase. That makes H3 relevant from stage 1, not just for dinos. Prediction: k=1 plateaus around the move-or-not coin flip on scrolled cells, and k≥2 solves stage 1 almost completely.

### Whiteboard
- Added the **Precedents** page (World Models, GameGAN, GameNGen, DIAMOND, Genie), with a comparison table and the ideas we're borrowing. GameGAN (NVIDIA 2020, Pac-Man) is the closest precedent.
- Applied the tagline. The whiteboard server restarted on a new port, so earlier marks don't carry over; their content was logged in the previous entry.

### Next
- Python environment (`uv`), then `notebooks/01-explore-frames.ipynb` to look at the data visually.
- Milestone 2: copy-last-frame baseline and metrics.
- A heuristic bot to replace the random walker (H6).

## 2026-10-05 (later) — Recording your play; Python side; how-it-works tabs

**Status check:** no model has been trained yet. Everything so far produces and checks *data*. The movement-rhythm finding came from measuring scroll in the recordings, not from training.

### Recording human play
- `cd game && npm run record` opens the game with `?record`. Each main-loop tick it stores `observe()` plus the key, using **the same encoder as the bot harness** (now in `game/src/recorder.ts`; the harness imports it). Output is byte-identical to before the refactor (verified).
- A session saves to `data/human/ep_<timestamp>/` when you die, win, or close/reload the tab. A vite dev-server plugin writes the `.npy` files.
- The action is the **last arrow keydown in the tick**, including key-repeat events, which is exactly what the game's movement system sees. Verified in Chrome: 30 key-repeat events over about 1 s became 16 LEFT ticks, and the recorded frames scroll left every other tick.
- `window.recorder` in the console shows the live session (`recorder.actions.length`).
- Gotcha: the player can die within about a second of spawning if a predator is close. That's real, so short sessions happen.

### Python (uv)
- `uv` installed. `pyproject.toml` declares the `tlfp` package with numpy, torch (Apple GPU/MPS works), matplotlib, jupyterlab, and wandb.
- `tlfp/data.py`:
  - `Episode` memory-maps `.npy` files, so they're read lazily.
  - `FrameDataset` is the sliding window `(k frames, key) → next frame`, padding the first frames.
  - `split_by_episode` holds out whole episodes so validation is honest.
- `tlfp/export_viewer.py` exports episodes for the whiteboard Data tab.

### Whiteboard
- **How it works** tab: pipeline diagram (built vs planned), one tick in code, a real frame with hoverable numbers, the on-disk layout, and the Python pieces, including an interactive sliding-window strip over real ticks.
- **Data** tab: replay any exported episode (bot or yours), with changed-cell outlines and a key-vs-change timeline.

## 2026-10-05 — First human data: 2 games (428 ticks)

In both games the player ate a level-1 dino and leveled up to 2 (at ticks 66 and 24). A predator then caught the player twice (2 → 1 → 0). Final score −1000 = +1000 for the level-up meal − 2 × 1000 for being caught. *(Corrected: this first said "no prey eaten", inferred from the final score and level. The recorded frames show the level changes.)* They lasted 268 and 160 ticks (≈19 s and 11 s).

### You vs the bot (stage 4, full game)
| | You | Bot |
|---|---|---|
| NONE ticks | **62%** | 21% |
| key-run length (median) | **1 tick (taps)** | 14 ticks (holds) |
| NONE-run length (median) | 2 | 13 |
| DOWN share | 1% (spawn location) | 21% |
| dinos in view (mean) | 3.8 | 4.2 |
| nearest dino distance (median) | 12 | 14 |

Observations:
1. **You tap; the bot holds.** Typical input looks like `U..U.U....L..L.L.U.U`. When you do hold a key, the OS key-repeat shows up (`R..RRRRRRRRR`): a press, a short gap, then a key every tick. The bot never produces either pattern. **This is the distribution shift H6 is about,** concretely measured.
2. **The input → movement delay is everywhere in your data.** At the start of a key press, the player moves on the *same* tick about half the time and on the *next* tick the other half (you: 38 vs 40; bot: 34 vs 55). Since you tap, the next tick is often NONE, so **many NONE ticks show movement**. With k = 1 the model sees "no key, yet the player moved" and can't explain it; with history it can (more support for H3).
3. ~~Direction bias~~: only 1% DOWN, but that came from where the player spawned, not from play style. Not a real pattern.
4. **Every game starts with an idle stretch** (about 15–22 NONE ticks during the intro zoom).
5. **Eating and level-ups are rare:** one per human game. The random bot hasn't been checked for them yet. Rare events need deliberate coverage.

### Implications
- Make the bot more human-like: a **"tapper" mode** that samples key-run and NONE-run lengths from your sessions, plus key-repeat-style holds (supports H6).
- Coverage targets: eating smaller dinos, leveling up, lava deaths, and winning at the map edge. Either aim for these in human play, or give the heuristic bot a chase-prey behavior.

## 2026-10-05 — Milestone 2: baselines and metrics; a harness bug found via human data

### Bug: dino AI was mostly frozen in the harness
`awarenessSystem` only runs for dinos **inside the game's viewport** (`level.getViewport()`). The harness had stubbed out the zoom animation, so the viewport stayed at its **6×6 intro size**, and almost no dinos ever noticed the player or prey. **Found because** the human data showed dino movement on 51% of non-scroll ticks versus 7% for the bot.
- Fix (per your direction: "zoomed out all the time is fine, zooming in is just for effect"): the viewport is fixed at 45×45. The harness applies the intro zoom-out instantly and disables dynamic zoom. The in-game recorder also disables dynamic zoom while recording, so human and bot dino behavior match.
- Stages 3–4 were regenerated. Stages 1–2 have no dinos and were verified byte-identical.
- Your first 2 recorded games used dynamic zoom (viewport sometimes 39×39), so dino awareness there differed slightly. They're fine as data, but noted.
- Lesson: anything stubbed in the harness can silently change game logic. **Human recordings double as a fidelity check on the harness.**

### Metrics (`tlfp/metrics.py`)
- `cell_acc` is about 98% for *any* baseline, so it's useless on its own.
- `changed_acc` (accuracy on cells that really changed) turned out to be **gameable**. A baseline that scrolls on every key tick scored 93.6% while getting only 24% of frames exactly right, because invented changes aren't penalized.
- Added **`dynamic_acc`**: accuracy over cells that changed in reality **or** in the prediction. This is now the **headline metric**.

### Baselines (`tlfp/baselines.py`, stage-1 validation episodes)
| baseline | history used | cell | **dynamic** | changed | exact frame |
|---|---|---|---|---|---|
| copy_last | – | 98.4% | **0.0%** | 0.0% | 64.6% |
| shift_by_key (scroll every key tick) | 1 frame | 97.8% | **40.1%** | 93.6% | 24.2% |
| shift_rhythm (scroll only if the last tick didn't move; handles the buffered key) | 2 frames | 99.3% | **68.6%** | 95.8% | 57.9% |

The hand-written rule using 2 frames beats the 1-frame rule by 28 points, an early hint for H3. The remaining errors are mostly the newly revealed edge row, which no rule can know. *(Baselines were run before the dino fix; only stage 1 is quoted here, and it's unaffected.)*

**A learned model is worth something if it beats 68.6% dynamic accuracy on stage 1.**
