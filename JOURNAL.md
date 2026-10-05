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

A level-1 player moves only **every 3rd tick** (`turnsToSkip = 2`), and a key press is **buffered** until the next move tick. So about 60% of ticks with a key held change nothing, and about 3% of NONE ticks still move (a buffered press). **From a single frame, the model can't know whether this tick is a move tick.** It needs history to infer the movement phase. That makes H3 relevant from stage 1, not just for dinos. Prediction: k=1 plateaus around the move-or-not coin flip on scrolled cells, and k≥3 solves stage 1 almost completely.

### Whiteboard
- Added the **Precedents** page (World Models, GameGAN, GameNGen, DIAMOND, Genie), with a comparison table and the ideas we're borrowing. GameGAN (NVIDIA 2020, Pac-Man) is the closest precedent.
- Applied the tagline. The whiteboard server restarted on a new port, so earlier marks don't carry over; their content was logged in the previous entry.

### Next
- Python environment (`uv`), then `notebooks/01-explore-frames.ipynb` to look at the data visually.
- Milestone 2: copy-last-frame baseline and metrics.
- A heuristic bot to replace the random walker (H6).
