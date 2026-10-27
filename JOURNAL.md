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

The hand-written rule using 2 frames beats the 1-frame rule by 28 points, an early hint for H3. Where shift_rhythm's remaining 31% of errors come from (measured): **85% are timing errors** (it scrolled on the wrong tick: the hand-written rhythm logic is only approximate), 5% are the newly revealed edge row (unknowable), and 10% other. So most of the gap is *learnable*, and a model has a lot of room to beat the rule. *(Corrected: this first claimed the errors were "mostly the edge row", unmeasured.)* *(Baselines were run before the dino fix; only stage 1 is quoted here, and it's unaffected.)*

**A learned model is worth something if it beats 68.6% dynamic accuracy on stage 1.**

## 2026-10-05 — H3 run 1: k = 1 (stage 1, CNN 220k params, 5 epochs, 16 min)
W&B: https://wandb.ai/jschomay-na/thunder-lizard-frame-prediction/runs/q7xa1xgl
- Val dynamic accuracy **never got above 22.6%** and ended at **6.7%**. It swung between 5% and 22%, with no upward trend after epoch 1. Exact-frame accuracy ended at 14%.
- That's below *both* hand-written rules (1-frame rule 40.1%, 2-frame rule 68.6%) and below copy-last on exact frames (64.6%).
- Consistent with H3 (one frame isn't enough), but **not conclusive until the k = 2 run under the same settings** finishes.
- Unexplained so far: why it does worse than the 1-frame *rule*, and why the curve is so jumpy. Two guesses, both unverified: (a) with move/no-move a coin flip, the per-cell argmax blends the two outcomes into an inconsistent frame; (b) the learning rate is too high or constant, with no decay. To check, look at actual predicted frames, and compare with k = 2's curve shape.

## 2026-10-05 — H3 run 2: k = 2 (same settings, 19 min)
W&B: https://wandb.ai/jschomay-na/thunder-lizard-frame-prediction/runs/zj8n6xp0

| | k = 1 | k = 2 |
|---|---|---|
| best val dynamic acc | 22.6% | **65.5%** (epoch 4.74) |
| trend | flat and noisy from epoch 1 | rising steadily through epoch 4.7 |
| best exact-frame acc | 19.9% | 46.5% |

- **H3 supported.** The only difference between the runs is one extra past frame. With it, the model reaches 3× the dynamic accuracy and keeps improving; without it, it stalls.
- k = 2 **hasn't beaten the hand-written 2-frame rule yet** (65.5% vs 68.6%), but it was still climbing.
- **Training instability:** at the very last step the loss spiked (0.09, about 3× normal), and the final eval collapsed to 19.6% dynamic / 0% exact. Together with the jumpy k = 1 curve, this points at the optimizer settings. A constant learning rate of 1e-3 with no gradient clipping can kick a model out of a good spot. The script also only keeps the final weights, so the best point (65.5%) is lost.
- Fixes before the next run: learning-rate warmup and cosine decay, gradient clipping, saving the best checkpoint by val dynamic accuracy, and more epochs.

## 2026-10-06 — Expectation vs reality after the first training runs

**Predicted** at kickoff: stage 1 would reach "close to 100% accuracy in minutes", and k ≥ 2 would "solve stage 1 almost completely". **Actual:** 65.5% dynamic accuracy after 19 minutes, still below a 10-line hand-written rule. The *direction* was right (H3); the *difficulty* was underestimated.

Still feasible: the signal is real, and the failures look like training setup rather than a dead end. Changes to the plan:
1. **Training hygiene** (needed): LR warmup/decay, gradient clipping, best-checkpoint saving.
2. **Look at predictions before tuning:** build a Predictions tab (model vs truth, wrong cells marked) so fixes target the real failure mode.
3. **The loss is dominated by easy cells:** ~98% of cells never change, so averaged cross-entropy mostly rewards copying. Candidate fix: up-weight cells that changed. → H9
4. **Make "copy" the default:** structure the model to predict *what changes* relative to the last frame (a residual or copy prior) instead of re-drawing every cell from scratch. → H10
5. **Compute:** about 2.6 min per epoch for 250k params on stage 1 is slower than assumed. Try half precision, and use Colab for later stages. The full game will need far more compute than the early estimate.

## 2026-10-06 — Run: k = 2 + training fixes + H9 (changed ×5) + H10 (copy prior), 10 epochs, 50 min
W&B: https://wandb.ai/jschomay-na/thunder-lizard-frame-prediction/runs/24sko1uj
- **Stable:** no crash, steady curves. **Fast start:** 64.5% dynamic accuracy after 1 epoch (the previous run had 54%).
- **Plateau:** best **67.1%** dynamic accuracy, still just under the hand-written rule (68.6%). Exact frames 50.6% (copy-last 64.6%). Accuracy on cells that really changed is 94.6%.
- Took 50 minutes vs the estimated 35–40.

### Where the errors are (validation, measured)
| tick type | ticks | share of wrong cells |
|---|---|---|
| no real move, but model **invented a scroll** | 1,284 | **81%** |
| real move, model moved too | 3,430 | 13% (mostly the unknowable edge row) |
| real move, model missed it | 83 | 5.5% |
| no move, model right | 5,203 | 0.6% |

- **The model learned the rhythm:** only 59 of the invented scrolls are rhythm mistakes (the previous tick moved).
- The other 1,225 are ticks where the player *tried* to move and **didn't**. The cell it pressed toward was **WATER (541) or OCEAN (428)** in 75% of cases, GRASS/JUNGLE/SHRUB in most of the rest.
- Likely explanations (not yet verified): ocean blocks movement (the model hasn't learned that), and water slows movement (+2 turns skipped), giving a slower rhythm that 2 frames can't reveal. Changed-cell ×5 weighting (H9) may also bias the model toward predicting movement.
- Next: check the game's movement rules for water/ocean; then compare k = 3–4 (slower water rhythm) and runs without ×5 weighting.

**Predictions tab** live: model vs truth vs rule on held-out episodes.

## 2026-10-06 — Review of the Predictions tab (with you)
- Your read of the tab: errors are almost all **wrong timing** or **edge cells**. The scrolling itself looks close to perfect. The dynamic-accuracy number is harsh: one mistimed tick counts every scrolled cell (~150) as wrong, which is exactly why H7 (metrics vs feel) exists.
- Edge cells come from procedural generation and don't need to be exact; they need to be *plausible*.
- Movement rules confirmed by you: **ocean blocks the player; water slows the player.** If training can't get these, the rules could be simplified.
- Open question → next experiment: in a rollout (model fed its own output), do newly revealed borders form island shapes or **smear**? Prediction: argmax smears (repeating the edge is the single most likely guess), and sampling (H5) or a generative model is needed for plausible new terrain.

## 2026-10-06 — First "play in the net" rollouts (no training)
`tlfp/rollout.py` plus the whiteboard **Play in the net** tab: 6 held-out episodes, 150 ticks each. All predictors start from the same real frames and replay the real keys, then only ever see their own output.

**Smear question, answered:**
- **argmax:** coherent but *stretched*. Newly revealed terrain continues the edge as long diagonal bands, a smooth smear rather than islands.
- **rule:** smears too, as straight stripes (it literally repeats the edge row).
- **sampled:** no smear, but **static**. Each cell is drawn independently, so terrain becomes speckle noise. Independent per-cell sampling can't make coherent shapes (H5 refuted in its simple form).

**The player disappears (argmax) or multiplies (sampled):**
| | player lost (count = 0) | ticks with >1 player |
|---|---|---|
| argmax | **4 of 6 rollouts**, often within 1–22 ticks | 0 |
| sampled | 4 of 6 at some point | up to 150 of 150 ticks; up to **17 players** |
| rule | never | never |

- Likely cause (unverified): the player's probability is **split across 4 facing classes** (×5 levels). Each one individually loses to "empty", so argmax picks empty even when "some player is here" is the most likely event. Sampling has the opposite problem: every cell rolls independently, so several cells can each come up "player".
- Fix ideas: factor the entity output into *what's here* (empty/player/dino/skeleton), then *its facing/level*. Or treat the player specially (exactly one, placed at the highest-probability cell).
- Per-step accuracy hid all of this; **rollouts are a necessary evaluation** (supports H7/H8).

## 2026-10-06 — Speed, continuity, and priorities (discussion)
- **Inference speed (measured):** stage-1 CNN, single frame: **2.1 ms CPU / 1.5 ms Apple GPU** (~1 GFLOP per frame). 10 fps real-time play needs 100 ms/frame, about 50× headroom natively. Browser (onnxruntime-web) not yet measured; even 10× slower would be ~20 ms. **Design constraint:** keep per-frame compute within ~10× of today's.
- **Continuity** (leave an area, come back, and it has changed): the model only sees k frames, so off-screen terrain is forgotten. Precedents struggle here too (GameNGen ~3 s of memory, Oasis notoriously, Genie 3 among the first to hold minutes; GameGAN used an explicit memory module). **Out of scope to solve; in scope to measure** (left-N/right-N rollouts scored on returning terrain). Possible later: a self-written memory map built only from the model's own predictions.
- **No grounding in the real map.** Decided against: it would make this something other than next-frame prediction.
- **Sampling "flutter"** kept as a possible fun secondary mode (smoother version: lower temperature, or correlated noise).
- **Entities:** the player is the easy case of the dino problem (split probabilities, multiple instances, appear/disappear, stochastic AI). Design the entity output (H11) for dinos and validate it on the player.
- **Priorities:** (1) water/ocean timing via short 3-epoch comparisons; (2) stage 2 lava (stochastic, tests H4 warm start); (3) stage 3 dinos with factored entity output, probably a U-Net; plus cheap side checks (continuity metric, browser speed).

## 2026-10-06 — Short comparisons: k = 4 and no weighting (3 epochs each, ~15 min each)
| run | k | changed weight | best dynamic | exact frame | changed acc |
|---|---|---|---|---|---|
| s1-3ep-k2-cw5 (control) | 2 | 5 | 65.9% | 39.3% | 94.3% |
| **s1-3ep-k4-cw5** | **4** | 5 | **71.6%** | 45.9% | 94.9% |
| s1-3ep-k2-cw1 | 2 | 1 | 65.0% | 50.6% | 88.8% |
| *hand-written rule* | | | *68.6%* | *57.9%* | |

- 🎉 **k = 4 is the first learned model to beat the hand-written rule** (71.6% vs 68.6%), in 3 epochs, also beating the 10-epoch k = 2 run (67.1%).
- Error analysis on k = 4: invented scrolls toward **water fell 541 → 322 (−40%)**, consistent with water's slower rhythm needing more history. Toward **ocean barely changed (428 → 410)**: blocking isn't a history problem (the ocean cell is visible in the current frame), and the model still hasn't learned it. Missed real moves fell 83 → 20.
- **No weighting (cw = 1)** vs ×5: about the same dynamic accuracy (65.0% vs 65.9%), but a different trade-off. Fewer invented changes (exact frames 39% → 51%), more missed changes (changed acc 94% → 89%). The ×5 weighting tilts the model toward predicting change; it doesn't make it better overall.
- Odd: both k = 2 runs invent a scroll on exactly the same 1,225 "tried to move, didn't" ticks. It looks like they learned the same simple rule (key pressed and the last tick didn't move → scroll).
- Open: why ocean blocking isn't learned. The visual signal is right there. Possibly too few blocked examples relative to normal moves.

## 2026-10-06 — Lava is slow but memoryless (measured, stage-2 data)
Per-tick chance that a cell next to active lava becomes lava (non-scroll ticks): DIRT 11.5% (~9 ticks avg), GRASS 2.4% (~42), SHRUB 1.2% (~80), JUNGLE 0.7% (~140), WATER 0.2% (~450), BURNT 0%.
- The game rolls fresh dice every tick, so spread is **memoryless**: the best prediction needs only the current frame, not 20+ frames of history. k = 4 is enough.
- Implications: argmax will **freeze** lava (stay ≫ spread on every tick). **Per-cell sampling is the *correct* model for lava** (spread really is independent per cell), unlike scrolling terrain, where it produced static. Per-tick accuracy can't measure this. Use **calibration** (predicted odds vs the table above) and **rollout growth curves** (lava area over 100+ ticks, averaged over many runs, vs real).
- Generate ~100k stage-2 ticks before training; spread events are rare.

## 2026-10-06 — Stage 2 (lava): warm start vs fresh (H4), and calibration (H12)
Data: 200 episodes, 96,867 ticks. Both runs: k = 4, copy prior, no weighting, 3 epochs, ~17 min each.
W&B: fresh https://wandb.ai/jschomay-na/thunder-lizard-frame-prediction/runs/1ne1f31p · warm https://wandb.ai/jschomay-na/thunder-lizard-frame-prediction/runs/9l1rdvgx

| val dynamic acc | after 0.25 ep | after 1 ep | final (3 ep) |
|---|---|---|---|
| fresh | 39.0% | 67.5% | 72.5% |
| **warm (from s1-3ep-k4-cw5)** | **70.7%** | 72.2% | **73.7%** |

- **H4 supported:** the warm-started model beats the fresh model's *final* score within a quarter epoch, and finishes ahead.

**Calibration** (val, cells next to lava, non-scroll ticks; predicted P(lava) vs actual):
| terrain | actual | fresh predicts | warm predicts |
|---|---|---|---|
| DIRT | 11.5% | 17.8% | **12.0%** |
| GRASS | 2.5% | 6.6% | 8.6% |
| SHRUB | 1.4% | 7.5% | 6.7% |
| JUNGLE | 1.0% | 11.7% | 2.3% |
| WATER | 0.23% | 4.2% | 4.0% |
- **Not calibrated:** the model over-predicts spread everywhere except DIRT (warm), by 2–17×. Ordering is roughly right (DIRT highest, WATER lowest-ish). A sampled rollout would spread lava several times too fast.
- Possible causes (unverified): too little training (3 epochs; rare events take longer); interaction with the copy prior. Standard post-hoc fix: **temperature scaling** (fit one number that sharpens or softens the probabilities on validation data).

## 2026-10-07 — Lava rollouts (H12, second half)
- New eval set `data/eval-lava-still`: 12 episodes, player placed 12 cells from lava and **standing still** (`record.ts --scenario lava-still`), so only lava changes. The real game: lava+burnt area grows ~150 cells in 20 ticks; the player is usually overtaken in 80–300 ticks.
- Rollouts of `s2-warm-k4` (no keys), new lava+burnt cells, mean:

| tick | real | argmax | hybrid (dice on lava + neighbors) | sample (dice everywhere) |
|---|---|---|---|---|
| 20 | 150 | **0** | 52 | 79 |
| 40 | 262 | 0 | 93 | 164 |
| 80 | 376 | 0 | 140 | 293 |

- **argmax freezes lava** completely, as predicted. **Sampling spreads it, but too slowly** (sample ≈ 50–78% of real; hybrid ≈ 35%).
- First hybrid version only rolled dice for cells *next to* lava, so lava never burned out (found by eye in the Play tab). Fixed: burning out is also a per-cell dice roll.
- **Why it under-spreads despite over-predicting spread odds (measured):** the model gives **frontier lava** (which still has a burnable neighbor) a **4.4% chance per tick of burning out; the game's true rate is ~0%** (lava can only burn out when fully enclosed by lava/ocean). For enclosed lava it's well calibrated (14.2% vs 13.6%). Premature burnout kills the spreading front faster than the inflated spread odds can make up.
- Temperature scaling wouldn't fix this. It's a structural error ("can't burn out while it has a burnable neighbor"), not overall over/under-confidence. Candidates: longer training, or more lava-frontier examples.
- Your call: exact lava rates aren't required; plausible-looking is enough for the POC.

## 2026-10-07 — Status checkpoint (before stage 3)
**Where we are:** stage 1 (movement) and stage 2 (lava) done with a plain CNN.
- Best stage-1 model: `checkpoints/s1-3ep-k4-cw5.pt` (k = 4, 71.6% dynamic, beats the hand rule's 68.6%).
- Best stage-2 model: `checkpoints/s2-warm-k4.pt` (warm-started from stage 1, 73.7%).
- Hypotheses: H3 🟢, H4 🟢, H5 🔴 (simple form), H9 🔴, H12 🟡; open: H1, H2, H6, H7, H8, H10, H11.
- Known gaps: ocean blocking not learned; new-edge terrain smears (accepted for POC); continuity out of scope (measure later); lava spreads too slowly when sampled (accepted); **the player vanishes or duplicates in rollouts** (→ H11).

**Stage 3 plan (dinos):**
1. Generate ~100k ticks with 5 dinos (`record.ts --stage 3`). Dino AI was fixed earlier (fixed 45×45 viewport).
2. Implement **H11**: split the entity head into *what's here* (empty/player/dino/skeleton) + *facing* + *level*, with facing/level loss only on occupied cells.
3. Implement a **U-Net** whose full-size path equals the current CNN, plus a zero-initialized shrink/grow branch, so it can warm-start from `s2-warm-k4`.
4. Runs (warn the user before each): CNN + H11 vs U-Net + H11, 3 epochs each, ~20–25 min each.
5. Evaluate dinos: error analysis by entity type, rollouts (do dinos persist, move plausibly, chase or flee?), player survival in rollouts (H11).
6. Keep the speed budget: per-frame compute within ~10× of today's ~1 GFLOP (2 ms CPU).

## 2026-10-07 — Stage 3 data
- With 5 dinos, a dino was on screen in only 18% of ticks and moved in 12% (your own games: 51% moving). Raised stage 3 to **25 dinos**: now 52% on screen and 38% moving.
- More dinos means more early deaths (only 64 of 200 games reached 500 ticks), so I generated two batches (seeds 0 and 1000): `data/stage3-100k` + `data/stage3-100k-b` = 400 games, **101,585 ticks**.

## 2026-10-07 — H11 split heads + U-Net branch (code)
- `--split-entity` (H11): three per-cell questions: what's here (4), facing (4), level (5). Facing and level are only scored on cells that hold a dino. The answers recombine into the usual 42 entity scores, so metrics and rollouts are unchanged. Checked that all 42 ids round-trip.
- `--unet`: a shrink-and-grow branch (45→23→12→back) added onto the full-size features, with its last layer zero-initialized. **Verified:** with stage-2 weights loaded, its output is identical to `s2-warm-k4`.
- `--init` now copies only the weights that match; new parts start fresh.
- Speed (CPU, 1 frame): CNN 3.3 ms → U-Net+split 4.7 ms (about 1.4×, well inside the 10× budget). Parameters: 0.30M → 0.57M.

## 2026-10-07 — Run s3-cnn-split (CNN + H11, warm from s2-warm-k4, 3 epochs, 14.4 min)
- Validation dynamic accuracy **68.4%**. On stage 3 the hand rule gets 57.2% and copy-last 0% (the rule knows nothing about dinos or lava).
- Breakdown on the validation split, compared with the stage-2 model applied unchanged:

| | s2-warm-k4 (never saw dinos) | s3-cnn-split |
|---|---|---|
| player cell exactly right | 89.1% | **94.5%** |
| player cell right when it moved | 23.3% | **48.3%** |
| frames with no player predicted (of 6,924) | 452 | **64** |
| dino cell exactly right | 0% | 28.2% |
| dino cell right when it moved | 0% | 9.8% |
| dino cells predicted / real dino cells | 0% | 74.7% |

- H11 looks good for the player in one-step predictions (frames missing the player fell 7×). Rollouts are still needed to confirm it.
- Dinos are partly learned. The model places many of them but often gets the exact class wrong (possibly facing or level) and rarely gets a moving dino right. Next: break dino errors down by kind / facing / level, then run rollouts, then the U-Net run (s3-unet-split, ~20 min going by this run's speed).

### Correction: s3-cnn-split is *worse* than doing nothing new
- On the same stage-3 validation games, the untouched **s2-warm-k4 scores 73.1%** dynamic accuracy (changed 84.2%, exact 13.6%), versus 68.4% for s3-cnn-split. Most of the score comes from terrain scrolling and lava, not from the few dino cells.
- Cause (from the log): the warm start collapsed to 49% at 0.25 epoch, then slowly recovered to 68%. The split heads started random, and at lr 1e-3 their early gradients scrambled the pretrained body.
- Fix for the next run: initialize the split heads from the old 42-way head, and/or lower the LR for warm starts (~3e-4). Also, always score the starting checkpoint on the new data as a baseline before training.

## 2026-10-07 — s3-cnn-split-v2 also got worse, and the cause
- v2 (split heads grouped from the 42 scores, lr 3e-4): starts at 73.1%, best 68.3% (0.25 epoch), ends at 66.6%. It also ran 3× slower (47 min) with the GPU at 100%; still to be explained.
- Comparison on the stage-3 validation set, s2-warm-k4 → v2:
  - terrain changed-cell acc 85.5% → 77.6%
  - entity changed-cell acc 48.7% → 44.0%
  - occupied entity acc 34.1% → 40.2%
- It's equally worse on *training* episodes (terrain 84.7% → 76.5%), so this is not overfitting.
- Validation loss: terrain went **up** (0.021 → 0.029; on changed cells 0.35 → 0.62) while entity went **down 3×** (0.022 → 0.006). Training traded terrain away to improve entities.
- **Likely cause: a loss-weighting bug.** The facing and level losses are averaged over dino cells only (roughly 1 in 1,000 cells), but terrain is averaged over every cell. So each dino cell counts about 1,000× more than a terrain cell. With gradient clipping, terrain's share of each update shrinks to almost nothing. The first s3 run had the same problem.
- Fix: average facing and level over all cells, like the rest of the loss.
- Note: 3 epochs was chosen for speed, not tested. Revisit once the loss is fixed.
- **Fixed** both: facing/level losses are now averaged over all cells. The slowdown came from the logsumexp grouping on MPS (391 ms/step vs 150 ms for the flat head). Replaced it with one matrix multiply that gives the same result: split now takes 158 ms/step, U-Net+split 215 ms/step.

## 2026-10-07 — s3-cnn-split-v3 (both fixes, lr 3e-4, 17 min)
- No collapse this time: start 73.1%, then 71.8–73.5% the whole run, ending at **73.4%**. Speed was normal.
- Compared with s2-warm-k4 on the stage-3 validation set:

| | s2-warm-k4 | v3 |
|---|---|---|
| dynamic | 73.1% | 73.5% |
| terrain changed-cell acc | 85.5% | 83.3% |
| exact frames | 13.6% | **21.5%** |
| frames with no player predicted | 452 | **59** |
| player exact | 89.1% | 94.6% |
| player right when it moved | 23.3% | 16.2% |
| dino predicted as a dino at all | 0% | 18.9% |
| dino right when it moved | 0% | 1.1% |

- **H11 (one-step):** the player almost never vanishes now. Rollouts are still needed.
- **Dinos are barely learned.** The model erases about 80% of real dinos. The stage-2 model learned that dino classes never appear, and with correct weighting dinos are only ~0.1% of the loss, so the gradient to unlearn that is tiny.
- Next run: `--dino-weight 20` (cells with a dino now or next count 20×) → s3-cnn-split-dw20.
- Process note: the monitor delivered all the eval lines in one batch at the end, so the live check-ins didn't actually happen.

## 2026-10-07 — s3-cnn-split-dw20 (dino cells weighted 20×, 18.5 min)
- Dynamic accuracy stayed at 70.5–72% the whole run. The best checkpoint was saved at epoch 1 (step 1472, 71.9%).
- Compared with v3:

| | v3 | dw20 |
|---|---|---|
| dynamic | 73.5% | 71.9% |
| terrain changed-cell acc | 83.3% | 83.3% |
| frames with no player (one-step) | 59 | **26** |
| player right when it moved | 16.2% | **35.0%** |
| real dinos predicted as a dino | 18.9% | **40.7%** |
| dino right when it moved | 1.1% | **14.7%** |
| predicted / real dino cells | 0.36 | 1.07 |

- Terrain is unchanged and dinos are much better. Dynamic accuracy is lower only because wrong dino guesses now count against it. That makes **dynamic accuracy a poor way to pick the best checkpoint at stage 3**: it favors models that ignore dinos. train.py now also saves `<name>-last.pt`.
- **Rollouts** (4 validation games × 150 ticks, the model fed its own output; average per frame):

| | real | v3 argmax | dw20 argmax | dw20 hybrid | dw20 sampled |
|---|---|---|---|---|---|
| players | 0.99 | 1.00 | 1.13 (max 3) | 1.01 | 6.7 (max 29) |
| frames with no player | 0.5% | **0%** | **0%** | 0% | 1.3% |
| dinos | 1.49 | 0.04 (vanish) | 4.8 (multiply) | 1.35 | 37 |

- **H11 is supported for argmax:** the player never vanished in any rollout (it used to vanish in 4 of 6). Sampling still multiplies entities.
- With v3, dinos vanish in rollouts; with dw20 they multiply under argmax. Hybrid happens to land close to the real count; that's a coincidence of the mode, not a fix.
- Next: pick checkpoints by an entity-aware metric; then the U-Net + split + dino weight run.

## 2026-10-07 — Checkpoint: where the CNN stands
- The CNN (split output + dino weight 20) is a **solid stage-3 baseline**; we're done tuning it for now.
  - Terrain changed-cell accuracy is 83.3% vs 85.5% for the stage-2 model: a small, stable cost.
  - The player persists, and dinos are partly learned (41% placed, 15% of moves right).
- **Success at stage 3:** judge by terrain accuracy, dino accuracy and rollouts, not by dynamic accuracy, which penalizes attempting dinos.
- **Checkpoint picking is not solved.** The dw20 run only kept its epoch-1 version (the "-last" saving was added after it started). Before the next run: pick the best checkpoint by a combined terrain + dino score, and confirm with rollouts.
- **H11:** the player never vanishes under argmax (4 games). Caveats: dw20 sometimes shows 2–3 players, and sampling multiplies everything.
- Next: U-Net + split + dino weight 20, directly comparable to s3-cnn-split-dw20.

## 2026-10-07 — H13: per-entity fates (s3-cnn-fates)
- **Labels** (`tlfp/moves.py`): match each entity across ticks after removing the scroll (found from terrain). 99% of dinos get a clear label: stay 59%, moved one cell 40%, die/gone <1%. No dinos spawn during play (checked the game code); 100 of 121 unmatched arrivals were at the screen edge.
- **Model:** a 7-way fate head (stay/up/right/down/left/die/gone) on the CNN, read at cells that hold an entity. Warm start from dw20, 3 epochs, 16 min.
- **One-step fate accuracy** (validation):
  - player: 90%; when it moved, right 82% of the time
  - dino: 63%; when it moved, right **27%** of the time (per-cell output: 15%)
  - but it predicts a move for only 20% of dinos vs 40% real, so argmax makes dinos too still
- **Rollout mode `fates`:** terrain as in hybrid. Entities are moved by their fate: argmax for the player, a random draw for dinos (in the game dino moves are partly random). New dinos are drawn only in the outermost ring, with the dino odds divided by the training dino weight.
  - Without that odds correction the edge kept filling with dinos (an average of 83 on screen); a 2-cell ring was still too many (5.9).
- **Rollouts** (8 validation games × 150 ticks):

| | real | argmax (this model) | fates |
|---|---|---|---|
| players per frame | 1.00 | 4.24 (max 53) | **1.00** |
| dinos per frame | 1.47 (max 4) | 8.67 (max 182) | **2.67 (max 12)** |
| average gap in dino count vs real | 0 | 8.0 | **2.0** |
| ticks where dinos changed | 82% | 73% | 58% |

- **H13 partly supported.** No more duplication or vanishing from per-cell decisions. The remaining drift comes from the edge spawns: counts creep up in some games (to 10–12 by tick 150) and drop to 0 in one.
- This model's own per-cell argmax got much worse (max 53 players): training the fate head changed the shared body. Not investigated.

### What the net decides vs what our code decides
| Net | Code |
|---|---|
| scroll (via terrain), lava spread/burnout, each entity's fate, facing/level, edge-dino odds | the output is game state drawn by a fixed renderer; lava cells roll dice; entities move one cell to their target, with collisions; dinos only appear in the outer ring; the player stays centred; odds corrected for the dino weight |

The net makes the game-logic decisions; the code applies bookkeeping. "Pure" world models (GameNGen, DIAMOND, Genie) learn all of it from pixels, at far greater data and compute.

## 2026-10-08 — U-Net + split + fates (s3-unet-fates)
- Warm start from s3-cnn-fates; the U-Net branch starts at zero, so it began at exactly 71.92%. Training: 3 epochs, 19 min, ended at 72.4% dynamic. Evaluated the final checkpoint (`-last`).
- Added a **chase/flee breakdown** to `tlfp/eval_fates.py`: for dino moves, by distance to the player, the share toward / away / stay. In the real game, dinos move toward the player about 2× as often as away, at every distance.
- **One-step fates** (validation), CNN fates → U-Net fates:
  - dino right when it moved: 26.6% → **39.3%**
  - dino overall: 62.7% → 67.5%
  - predicted to move: 20% → 26% (real 40%)
  - player right when it moved: 82.0% → 85.2%

| distance | real (toward / away / stay) | CNN | U-Net |
|---|---|---|---|
| near ≤6 | 27 / 13 / 60 | 15 / 6 / 79 | **23 / 7 / 70** |
| mid 7–13 | 32 / 10 / 58 | 16 / 6 / 77 | **21 / 7 / 72** |
| far >13 | 24 / 15 / 61 | 12 / 7 / 81 | **16 / 10 / 75** |

- The U-Net predicts more chasing at every distance, and is closest to real for near dinos.
- **Rollouts** (fates mode, same 8 games): dinos per frame 3.95 (max 14) vs CNN 2.67 (max 12) and 1.47 real. Dinos changed on 61% of ticks (CNN 58%, real 82%). Terrain drift is about the same (33.5% vs 35.1%; mostly unknowable new terrain).
  - The extra dinos come from the edge spawns (the per-cell dino odds), not from the moves, which can't duplicate. **Edge spawning is now the weak point.**
- Speed: 4.5 ms/frame on CPU vs 3.3 for the CNN (1.4×, inside the budget).
- **Conclusion:** the U-Net helps dino behaviour one step at a time (H13 + range). In rollouts, the edge-spawn rule hides the gain. Next: fix spawns (e.g. only on rows/columns newly revealed by a scroll, or a learned "arrivals" head trained on real arrivals), then compare rollouts without spawn noise.
- **Spawn rule comparison** (fates rollouts, 8 games; real = 1.47 dinos per frame):

| spawn rule | CNN dinos (max) / off-by | U-Net dinos (max) / off-by |
|---|---|---|
| outer ring every tick | 2.67 (12) / 1.96 | 3.95 (14) / 3.14 |
| only the row/column a scroll just revealed | 1.20 (5) / 0.70 | **1.36 (5) / 0.71** |
| no spawns | 1.06 (4) / 0.65 | 1.24 (4) / 0.64 |

  "Revealed" is now the default (`--spawn`). Dinos can walk in from just outside the view without a scroll, so a few arrivals are missed; that's acceptable for now. Whiteboard rollouts were regenerated with it.

## 2026-10-09 — Playable app (`app/`)
- Standalone page, no game engine: `app/index.html` + `main.js` + `model.onnx` (s3-unet-fates-last, 2.3 MB) + `seeds.json` (12 starting states from held-out games, 4 real frames each). Built with `uv run python -m tlfp.export_onnx <ckpt>`, which checks the ONNX output matches PyTorch.
- Each tick runs the same step as the `fates` rollout mode (hybrid lava, dino fates drawn at random, the player's fate is the top choice, new dinos only on the edge a scroll reveals) and draws it with the game's own glyphs and colours.
- The game runs one tick every 70 ms (14.3 per second). It shows an FPS meter (ticks/s) and the net's time per frame.
- **Speed:**
  - WebAssembly CPU: 150–250 ms per frame (too slow).
  - **WebGPU: about 15 ms for the net, 16 ms per full tick** (net + apply + draw), so up to ~60 ticks/s, about 4× headroom. It falls back to the CPU if WebGPU is missing.
  - Gotcha: Chrome slows timers in hidden tabs to 1 per second, so measure with the tab visible.
- Run locally: `cd app && python3 -m http.server 8765`, then open http://localhost:8765/.
- H8 reframed: preventing divergence from the real game is a non-goal; H8 is about degradation (smear, static, impossible states), judged by plausibility statistics and feel.

## 2026-10-09 — First play-test findings (Jeff, playing app/)
Overall: "it is magical". What breaks the illusion:
1. **Lava works quite well.**
2. **"Most likely" dino moves barely move.** Argmax nearly always picks "stay" (the net predicts a move for 26% of dinos vs 40% real). Dice mode (the default) draws moves from the probabilities.
3. **Moving through uniform non-water terrain sometimes doesn't scroll at all.** Water scrolls well.
4. **Dinos sometimes "pin" to the player** instead of scrolling with the world.
5. **Smearing fills the screen with one terrain type**, so scrolling stops being visible.
6. **Ocean boundaries scroll worse** than other terrain boundaries.
7. **Glyphs pin to screen coordinates** instead of moving with the world (in the real game they're tied to world tiles).

**Diagnosis:**
- 3, 4 and 7 share one cause. The app (and `rollout.py`) work out the scroll by matching the net's new terrain to the last one. In textureless or smeared terrain every shift matches equally, so no scroll is detected, and dinos and glyphs don't shift. (Glyphs are already tied to world position, but that position only updates when a scroll is detected.)
- 5 and 6: the net redraws all 2,025 terrain cells every tick; errors compound and uniform areas spread. Long smooth ocean edges give it the least to go on.

**Plan (H14):** take the scroll from the player's predicted fate (85% right on moves); shift already-visible terrain by it; the net only draws the newly revealed row/column and lava changes. This trades more bookkeeping in code for no smear and consistent scrolling. The purer alternative (H8 noise training so the net itself stops smearing) stays open as a comparison.

## 2026-10-09 — H14 implemented (app + `rollout.py` fates mode) — later reverted, see next entry
- The scroll comes from the player's predicted fate. Already-visible terrain is shifted by it; the net draws only the newly revealed row/column, plus dice for lava and its neighbours. The terrain-matching scroll detector was removed from the app.
- **New finding:** with the player's fate on argmax, the player barely moved (scrolled on 15% of key-held ticks vs 44% real). From a standstill the net gives "move" a constant ~31% chance (holding RIGHT for 60 ticks, identical every tick), so argmax never starts moving, and the frames stay identical. Fix: dice for *whether* the player moves, the net's top choice for *which way*.
- **Rollouts** (8 games × 150 ticks, U-Net fates):

| | real | hybrid (net redraws terrain) | fates + H14 |
|---|---|---|---|
| isolated "static" terrain cells | 0.02% | 1.05% | **0.00%** |
| biggest terrain type, share of screen | 40% (max 59%) | 38% (max 66%) | 43% (max 63%) |
| scrolled on ticks with a key held | 44% | — | **41%** |

- In the app: 30 ticks holding RIGHT → 15 columns scrolled, then 30 ticks DOWN → 11 rows. Glyphs, dinos and terrain all move with the world.
- **Net vs code:** the code now also shifts already-seen terrain and rolls dice for whether the player moves; the net still decides the move odds and direction, the revealed terrain and lava.

## 2026-10-09 — H14 reverted: direction check
- Jeff: "I want frame prediction, this is moving away too far to game engine." H14 (code shifts terrain, scroll from the player's fate, dice for whether the player moves) was built, measured, then **reverted** (`git revert`). Its numbers are kept above as a record of the trade-off (that entry was accidentally removed by the `git revert`, and was restored).
- **Principle going forward:** the net predicts the frame. Code is limited to encoding the input, decoding the output (argmax / sampling / temperature) and drawing. Fixes for smearing, scrolling and pinning should come from the model and its training, not from game logic in code.
- Open question: H13 (fates: code moves entities, edge-only spawns) also puts game bookkeeping in code. To be decided.

## 2026-10-09 — Terrain sampling slider in the app
- Decision (Jeff): keep dino fates (unavoidable to stop dinos multiplying); otherwise leave the app as is (frame prediction; no H14).
- Added a "sampling" slider, a terrain temperature from 0 to 1.5. 0 = argmax (lava still rolls dice); above 0, every terrain cell is drawn from softmax(logits / T).
- The net is very confident about terrain (the copy prior). Cells per tick that differ from the top choice, and isolated "static" cells after 40 ticks of moving right:

| T | differ from top choice per tick | static after 40 ticks |
|---|---|---|
| 0.5 | 2 | 0% |
| 0.75 | 6 | 0% |
| 1.0 | 12 | 0.05% |
| 1.5 | 52 | 1.1% |

- So "a tiny bit of sampling" means about 0.5–1.0.

## 2026-10-09 — App polish from Jeff's whiteboard markup
- A random start (one of the 12 seeds, never the same twice in a row) on every load and restart; the start dropdown is gone.
- Dino moves always use dice; the dropdown is gone.
- Terrain sampling is fixed at temperature 0.2; the slider is gone.
- **Restart bug:** a tick still in flight when you pressed restart wrote its frame into the new game. Fixed with a generation counter: stale ticks are dropped. Tested by restarting mid-tick 5 times.
- The subtitle is shortened, with a link to the original game (enegames.itch.io/thunder-lizard-eat-or-be-eaten).
- Fixed: a page-wide `canvas` style gave the whiteboard's transparent layer a dark background, hiding the game under the overlay. `?wb=<server>` loads the markup overlay (dev only).

## 2026-10-09 — Release, scorecard, coast data, H8 setup
- **The app is released as is** (Jeff): random starts, dice for dinos, terrain temperature 0.2, U-Net fates model. `checkpoints/best-so-far.pt` = s3-unet-fates-last. Stage 4 skipped: dinos spawn at the revealed edge by the net's odds, so there's no global dino count to tune, and there are plenty.
- **Scorecard** (`tlfp/scorecard.py`): one fixed report on the stage-3 validation games for any fates model; saves `scorecards/<name>.json`. best-so-far:

| metric | best-so-far | real |
|---|---|---|
| terrain changed-cell acc | 83.3 | |
| player move right | 85.2 | |
| ocean block (stay when pushing into ocean) | **99.7** | |
| dino move right | 39.3 | |
| dinos predicted to move | 26.4% | ~40% |
| near dinos toward/away/stay | 21/7/72 | 29/12/60 |
| rollout static cells | 0.023% | 0.02% |
| rollout biggest terrain type (worst frame) | **74%** | ~59% |
| rollout dinos per frame | 1.37 | ~1.5 |
| rollout scroll when key held | **28%** | 44% |
| CPU ms/frame | 4.5 | |

- **Coast scenario** (`--scenario coast`: start next to ocean; CoastBot pushes into it): 80 games, 15.6k ticks, 78% of key presses push into ocean. But the normal data already has 12% ocean pushes, and the one-step ocean block is 99.7%. **Ocean blocking isn't a data problem.** The coast data is kept but unused.
- **H8 implemented** (`--corrupt P`): half the examples get damaged inputs at up to 3% of cells (speckle or smear on terrain in all frames; dinos dropped in the older frames only). No noise-level input, so the model and app stay compatible.
- Running: **s3-unet-more** (control: 3 more epochs, no damage; also answers "does longer training help?") and **s3-unet-h8** (same + `--corrupt 0.5`). The targets are rollout max_share (smear) and scroll_when_key.

## 2026-10-09 — Published (early WIP)
- **Live at https://thunder-lizard-frame-prediction.vercel.app/** (Vercel project `thunder-lizard-frame-prediction`, deployed from `app/` with the CLI; there's no build step).
- Added a "Runs best in Chrome or Edge (WebGPU)" WIP line. `app/.vercelignore` keeps the Vercel link files and `.env.local` (an OIDC token the CLI created) out of the upload; checked that `/.env.local` returns 404.
- Gotcha: Vercel CLI 63 fails on Node 26 (`fetch failed`); run it with Node 24 (`~/.nvm/versions/node/v24.18.0/bin/npx vercel ...`).
- Checked live: the model loads, runs on WebGPU and plays.
- Redeploy: `cd app && ~/.nvm/versions/node/v24.18.0/bin/npx vercel deploy --prod`.

## 2026-10-09 — Longer training vs H8 (scorecards)
Both runs: warm start from best-so-far, 3 more epochs on the stage-3 data (19–22 min each). Last checkpoints scored:

| metric | best-so-far | **more** (control) | h8 (`--corrupt 0.5`) | real |
|---|---|---|---|---|
| terrain changed acc | 83.3 | 85.1 | **85.9** | |
| player move right | 85.2 | 90.0 | **92.7** | |
| ocean block | 99.7 | 99.8 | 99.7 | |
| dino move right | 39.3 | **45.8** | 43.5 | |
| dinos predicted to move | 26.4 | **29.6** | 27.8 | ~40 |
| near dinos toward/away/stay | 21/7/72 | **23/8/68** | 22/8/70 | 29/12/60 |
| rollout static % | 0.023 | 0.018 | 0.029 | 0.02 |
| rollout biggest terrain type | 74 | 71 | 71 | ~59 |
| rollout dinos per frame | 1.37 | **1.50** | 1.24 | ~1.5 |
| rollout scroll when key held | 28 | **44** | 46 | 44 |
| CPU ms | 4.5 | 5.3* | 5.1* | |

*CPU timing taken while the other run trained; noisy.

- **Longer training helps a lot,** especially rollout scrolling: 28% → 44%, matching the real game (Jeff's "sometimes doesn't move at all"). Dinos also got better (45.8% of moves right).
- **H8 is inconclusive.** Small gains on terrain and player one-step, slightly worse dinos, and **no change to its target**: the biggest terrain type is still 71% vs ~59% real, and static is unchanged. Not adopted for now. The rollout metrics come from 8 games and are noisy.
- Candidate new best: **s3-unet-more-last**, pending Jeff's play-test (exported to a local test copy of the app, not deployed).
- **Adopted (Jeff play-tested: "good"):** s3-unet-more-last is the new `checkpoints/best-so-far.pt` (the previous one is kept as `best-prev-unet-fates.pt`), exported to `app/` and redeployed.

## 2026-10-09 — Hypotheses closed, H15 added
- Closed (Jeff OK'd): H7 🟢 (informal), H8 🔴 in this form, H12 🟡 closed as partly supported, H13 🟢. Still open for the architecture chapter: H1, H2, H6, H10.
- The LinkedIn WIP post is going viral (~30k impressions). A commenter: "Temporal consistency requires recurrence… ConvRNN, ConvGRU, ConvLSTM." Added as **H15**: a recurrent hidden state across ticks instead of a fixed 4-frame window. Plausible here because the game has hidden state the frames don't show (movement phase, dino timers and targets, terrain that scrolled away), which a recurrent state could carry. Doable as a body swap within the CNN family; slower training (backprop through time).

## 2026-10-09 — H10 closed: the copy prior
- A clean comparison: stage 1, k=4 CNN, 1 epoch **from scratch**, identical except `--copy-prior`.

| epoch | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|
| dynamic, copy prior | 42.1% | 59.7% | 64.2% | **64.7%** |
| dynamic, none | 33.3% | 56.6% | 60.7% | 62.6% |
| exact frames, copy prior | 29.9% | 43.6% | 46.4% | **47.2%** |
| exact frames, none | 0.0% | 1.2% | 6.7% | 10.4% |

- **H10 🟢 supported:** it learns faster and gets far more whole frames exactly right. Without it, the net has to learn "copy every unchanged cell" before it gets anything fully right.
- H15 (s3-unet-gru) is training now; see the whiteboard "Memory (H15)" page.

## 2026-10-09 — H15 first run: s3-unet-gru (ConvGRU memory, 8-tick sequences)
- Warm start from best-so-far; the memory output starts at zero, so the start is identical (74.72%). 3 epochs, 37 min (about 1.8× a normal run).

| metric | best-so-far (s3-unet-more) | **s3-unet-gru** | real |
|---|---|---|---|
| terrain changed acc | 85.1 | 86.4 | |
| player move right | 90.0 | 92.2 | |
| dino move right | 45.8 | **49.6** | |
| dinos predicted to move | 29.6 | 31.4 | ~40 |
| near dinos toward/away/stay | 23/8/68 | 25/9/67 | 29/12/60 |
| rollout static % | 0.018 | **0.007** | 0.02 |
| rollout biggest terrain type | 71 | **64** | ~59 |
| rollout dinos per frame | 1.50 | 1.38 | ~1.5 |
| rollout scroll when key held | 44 | 45 | 44 |
| CPU ms/frame | 5.3 | 7.2 | |

- The first real movement on smearing (biggest terrain type 71 → 64, static down). Dinos are better too.
- **Confound:** the GRU run also got 3 more epochs than best-so-far, and we know extra training helps. The one-step metrics run *without* memory (empty memory each call), so those gains are probably from the extra training. The rollout numbers use the memory. Running the control **s3-unet-more2** (the same 3 epochs, no memory) before drawing a conclusion.
- **Control s3-unet-more2** (same 3 epochs, no memory) and both rescored on **24 rollout games** (8 was too noisy):

| metric | more2 (control) | gru (H15) |
|---|---|---|
| terrain changed acc (one-step, GRU with empty memory) | **88.1** | 86.4 |
| player move right | **93.2** | 92.2 |
| dino move right | 48.9 | 49.6 |
| near dinos toward/away/stay | 25/9/67 | 25/9/67 |
| rollout static % (24 games) | 0.100 | 0.096 |
| rollout biggest terrain type (24) | 68 | **66** |
| rollout dinos per frame (24) | 1.25 | **1.43** (real ~1.5) |
| rollout scroll when key held (24) | 41 | 39 |
| CPU ms | 5.0 | 6.1 |

- **H15 verdict for this form: 🟡 weak / inconclusive.** Most of the smear gain came from the extra training. The memory adds a small edge on smear (66 vs 68) and keeps the dino count closer to real (1.43 vs 1.25), at 1.2–1.8× compute, with app plumbing still to build. One-step numbers are slightly lower for the GRU, but they're measured with an empty memory, which it never saw in training, so that's not a fair test of it.
- Not tested yet, and where memory should matter most: a **continuity test** (walk away, come back) and longer training sequences (8 ticks ≈ 0.6 s of game time).
- Candidate new best: **s3-unet-more2** (simpler; best one-step numbers), pending Jeff's play-test.
- **Play-test of s3-unet-more2 (Jeff):** "feels the same" as the live model, so the live model stays (no redeploy). Walking away and back loses the earlier terrain completely. That's expected for every current model (a 4-frame window; even the H15 memory is screen-sized, so terrain that scrolls off is lost). Plan: add a continuity score to the scorecard (walk N away, walk back, measure the match).
- **On the LinkedIn claim** ("temporal consistency requires recurrence"): right in direction, "requires" overstated (windows and attention are the alternatives). Our H15 was the easy version: the memory was added on top of a model that already works from 4 frames, it trained only on real inputs, 8-tick sequences, a screen-sized memory. A fairer test: 1 frame + memory from scratch vs the 4-frame window; training on its own predictions over 32+ ticks; a memory bigger than the screen, plus the continuity score.

## 2026-10-12 — H15b: 89×89 memory canvas, 128-tick sequences (s3-gru-canvas)
- Following the LinkedIn commenter's advice: a ConvGRU memory canvas about 2× the view (`--mem-pad 22`), 128-tick sequences with truncated backprop in 32-tick chunks, plus new **backtracking bot data** (`--scenario backtrack`: walk out, retrace; 402 games, 87k ticks; mixed 50/50 with stage-3). Width kept at 64: the full 128-width version was ~11× live compute (blows the browser budget). 1 epoch, 63 min.
- New **continuity score** (scorecard): roll out from a real moment, replay the real keys through a walk-away-and-back, compare the model's frame on return with where it started (lava excluded). Real game 95.4%. Backfilled: best-so-far 62.2, h8 65.9, more2 70.2, gru(8-tick) 70.6.
- Live signal: chunk losses. In the second half of the epoch, chunks that start with memory had 11% lower loss than chunk 1 (no difference in the first half), so the memory was being used.
- **Scorecard (24 games):**

| | more2 (no memory) | s3-gru-canvas | real |
|---|---|---|---|
| continuity | 70.2 | **70.0** | 95.4 |
| biggest terrain type (smear) | 68 | **86** ❌ | ~59 |
| dinos per frame | 1.25 | 1.22 | ~1.5 |
| scroll when key held | 41 | 41 | 44 |
| terrain changed acc | 88.1 | 88.3 | |
| dino move right | 48.9 | 47.7 | |
| CPU ms/frame | 5.0 | **18.2** | |

- **Negative result:** no continuity gain, worse smear in rollouts, 3.6× slower.
- Likely why: with real frames as training input, remembering terrain only pays off on the single edge row revealed when scrolling back (~45 of 2,025 cells, on some ticks), so the loss barely rewards it. One epoch is also short for a new memory.
- Scorecards are now pushed to W&B as `score/*`.

## 2026-10-21 — Architecture chapter, part 1: masked diffusion for terrain (H16, H17)

**Why diffusion first:** smearing is the most visible flaw. Each terrain cell is picked alone, so when the net is unsure the most common type wins everywhere. Masked diffusion builds the frame in a few passes, and each pass sees the cells already fixed. The transformer (long history for continuity) comes second, because H15 suggested the training, not the architecture, is what doesn't reward remembering.

**What was built (`--diffuse`):**
- The U-Net stays. It gets one extra input: a draft of the next frame's terrain, where some cells are known and the rest are "unknown". A new layer mixes in the draft and starts at zero, so the warm start begins exactly as best-so-far.
- **Training:** hide a random share of the real next terrain (cosine: mostly large shares) and score only the hidden cells.
- **Generating:** start all unknown; 4 passes; each pass keeps the most confident guesses (cosine schedule) and leaves the rest unknown. Lava and its neighbours are still rolled with dice.
- Only terrain is diffused. Dinos and the player keep their fates and dice.
- Scorecard: `--steps N` sets the passes; one-step terrain uses the full decode.
- Whiteboard: new **Diffusion (H16)** tab with a toy (argmax smear vs sampling speckle vs passes), masked vs noise, the U-Net diagram, hide-and-fill on a real frame, the one-shot model's confidence on real held-out moments, and pass-by-pass views.
- **Colab:** not needed. A 3-epoch run takes about 24 minutes on the Mac.

**Seen on real data before training:** on a scroll moment, the one-shot model's 163 wrong cells are a missed scroll (coastlines off by one), and it was *confident* about them. Diffusion targets cells the model is unsure of, so it probably can't fix mistimed scrolls.

**Results** (s3-diffuse-last vs control s3-unet-more2-last, same 3 extra epochs from best-so-far; 24 rollout games; real: static 0.02, max_share ~59, continuity ~95):

| decode | static % | max_share % | continuity | terrain_changed | CPU ms/frame |
|---|---|---|---|---|---|
| control (one-shot) | 0.100 | 68 | 70.2 | 88.1 | 5.0 |
| diffusion, 1 pass | 0.237 | 69 | 68.0 | 87.8 | 4.9 |
| 2 passes | 0.161 | 71 | 69.5 | 87.6 | 9.9 |
| **4 passes** | 0.101 | 70 | 67.9 | 86.1 | 26.3 |
| 8 passes | 0.075 | 73 | 67.3 | 85.3 | 40.8 |
| 4 passes, sampled T=0.5 | 0.084 | 73 | 70.2 | 86.1 | ~45* |
| 4 passes, sampled T=1.0 | 0.249 | 77 | 71.5 | 86.1 | ~45* |

*two scorecards ran at once; timings inflated. Dino and player numbers are unchanged throughout (as designed).

- **More passes make terrain cleaner but more smeared.** Stray cells fall steadily with passes (0.24 → 0.08%), so later cells do agree with earlier ones. But they agree on the majority, and the biggest terrain type grows (69 → 73%).
- **Sampling instead of argmax (the MaskGIT way) makes it worse**, not better: 77% at T=1. The extra randomness compounds over 150 ticks.
- On the 8 real one-step moments, diffusion is a little better on the big scroll errors (163 → 151, 113 → 96 wrong cells) but these are mostly missed scrolls either way.
- **Reading:** the smear isn't one frame's indecision, which is what diffusion fixes. It builds up over the rollout as the net is fed its own slightly-off frames (the same "exposure" problem H8 tried and failed to fix with fake damage). A per-frame sampler can't fix an accumulation problem.
- **H16 🔴 in this form. H17:** passes cost time linearly (~5 ms each on CPU); 4 passes would fit the browser's budget, but there's no gain to pay for.

## 2026-10-22 — Does diffusion have potential? (90-minute session)

**1. Random cell order (decode only, MaskGIT's fix).** Noise on the ranking of which cells are fixed first, fading to 0 by the last pass; values stay argmax.
- `max_share` (the single worst frame) proved too noisy to judge by: strength 1 gave 76, strength 4 gave 68. **New scorecard measure `share_mean`:** the biggest terrain type averaged over all rollout frames, with `share_mean_real` for the real game over the same ticks.

| model / decode | share_mean (real 40.2) | max_share | static % | continuity | terrain_changed |
|---|---|---|---|---|---|
| control s3-unet-more2 (one-shot) | 42.1 | 68 | 0.100 | 70.2 | 88.1 |
| s3-diffuse, 4 passes | 43.6 | 70 | 0.101 | 67.9 | 86.1 |
| s3-diffuse, 4 passes, random order 4 | **41.6** | 68 | **0.077** | 70.4 | 87.4 |

- **On average the smear is mild:** 42% vs 40% real. The 68–70% frames are rare spikes.
- Random order turns diffusion from slightly worse to slightly better than the control on smear and stray cells, at equal continuity. The gaps are small (about 1 point), close to the noise.

**2. GameNGen-style damage with a damage-level input.** In training, 70% of examples get their input frames damaged (speckle/smear on up to 15% of cells; older-frame dinos dropped, capped at 30%), and the net is told the damage level (a new zero-initialized input). Rollouts run at level 0. Two runs, 1 epoch each from best-so-far, in parallel: with diffusion (s3-diffuse-gng) and without (s3-gng).
- First tried 3 epochs each, but in parallel a quarter epoch took 8 minutes (~95 min total), more than the machine's 90 minutes; restarted at 1 epoch.

**Results (24 rollout games; real: share_mean 40.2, static 0.02, continuity ~95, dino moves ~40%):**

| model / decode | share_mean | static % | continuity | dino_moves_pred | dino_move_right | CPU ms |
|---|---|---|---|---|---|---|
| control s3-unet-more2 | 42.1 | 0.100 | 70.2 | 30.6 | 48.9 | ~5 |
| s3-diffuse + random order | **41.6** | 0.077 | 70.4 | 31.2 | 49.5 | ~26 |
| s3-gng (damage + level, one-shot, 1 ep) | 42.7 | 0.054 | 68.6 | 27.5 | 43.7 | ~5 |
| s3-diffuse-gng, 4 passes | 44.6 | 0.069 | 70.4 | 27.4 | 43.5 | ~26 |
| s3-diffuse-gng + random order | 42.8 | **0.046** | **71.6** | 27.3 | 43.6 | ~26 |

(CPU ms for diffusion is ~5 ms × passes; the scorecard timings were inflated by parallel runs.)

- **Damage with a level input halves stray cells** (0.10 → 0.05), the first training change to move `static` (H8 in its simple form didn't). It doesn't touch the smear, and it **costs the dinos** (moves predicted 31 → 27%, right move 49 → 44%). Dropping dinos from older frames probably taught it that dinos flicker; next time damage terrain only.
- **Diffusion on top adds ~1 point** of continuity and a little less static, within the noise of 24 games.
- **Random order is required** for diffusion here: without it, the passes drift toward the majority terrain (44.6 vs 42.8).

**Verdict:** at this scale, diffusion gives at most ~1–2 points on smear/static/continuity, for 4× the compute per frame. The big gaps (continuity 70 vs 95, dino moves 30 vs 40%) don't move at all. The smear we were chasing is mostly rare spikes (42% vs 40% on average). Diffusion isn't broken; it targets a problem we mostly don't have. GameNGen's damage idea is the useful takeaway, without diffusion.

Not tried: rollouts at a small damage level (GameNGen used one), damage on terrain only, longer training.

## 2026-10-27 — Transformer memory (H18, H19): build, speed checks, Colab

- **Design (hybrid):** the CNN + U-Net stays and still reads the last 4 frames. A new **memory branch** reads a strided history (ages 0–7, then 8, 16 … 128: 23 frames). Each frame goes through a small CNN stem, is cut into patches, and becomes tokens tagged with screen position and age. 25 "next frame" query tokens (plus the key) attend to all of them in a 4-layer transformer (d=256), and the result is added to the CNN's features. Its output starts at zero, so the warm start begins as best-so-far. Screen positions only (option A); no world coordinates.
- **H19 weighting:** `--revealed-weight` multiplies the loss on the row/column a scroll reveals; detection matches `find_scroll` on 1,398/1,398 ticks.
- **Speed on the Mac (MPS):** 5×5 patches (81 tokens/frame): 10 s/step at batch 16 (~15 h/epoch), out of memory at 32. 9×9 patches (25 tokens/frame): ~1 h/epoch. A 5-step test then **made the Mac unresponsive** (attention on MPS uses ~28 GB of shared memory). Rule from now on: transformer runs go to Colab.
- **Speed in the browser (WebGPU, untrained branch alone):** 9×9 patches 48 ms per tick (~64 ms with the CNN: fits the 70 ms tick); 5×5 patches 145 ms (doesn't). → 9×9 patches.
- **Colab (free T4):** training now supports CUDA, fp16 mixed precision with loss scaling, `--out-dir` (Google Drive), saving `-last` with optimizer state at every evaluation, and `--resume`. Notebook: `colab/train_transformer.ipynb`. Data + starting checkpoint: a 16.5 MB zip (1.15 GB unpacked; the grids compress ~70×).
- Replaced `F.fold/unfold` with reshapes (same result, exports to ONNX).

**First results (2026-10-27).** Transformer: 2 epochs on Colab (T4 for the first quarter epoch at 1.06 s/step, then an A100 at ~0.05 s/step, about 20× faster thanks to bfloat16 + fast attention; resumed from Drive). Control: the CNN with the same data and revealed-cell ×10, 2 epochs on the Mac (~25 min). 24 rollout games, real values on the right.

| | best-so-far line (more2) | CNN control | **transformer** | real |
|---|---|---|---|---|
| continuity | 70.2 | 71.8 | **73.9** | ~95 |
| share_mean (smear) | 42.1 | 41.2 | **40.3** | 40.2 |
| static % | 0.100 | **0.082** | 0.109 | 0.02 |
| terrain_changed (one step) | 88.1 | 88.9 | **90.0** | |
| dino_move_right | 48.9 | 51.6 | **52.1** | |
| dinos per frame | 1.25 | 1.43 | 1.36 | ~1.5 |
| scroll_when_key | 41 | 41 | 40 | 44 |

- The memory branch is used (its output layer grew from 0 to a weight norm of 13), and one-step validation improves ~1 point (dynamic 80.6 vs 79.4).
- **Continuity +2.1 over the control**, far from the ≥80 target. Smear is now at the real level. Stray cells slightly worse.
- The control alone (backtrack data + revealed ×10) beat the old best on nearly everything: the data and weighting help without memory.
- H19 isn't separated yet: both runs used the revealed weighting.
- **H18: 🟡 small gain.** Next: look at what newly revealed patches attend to, to see whether the net finds the old view at all.
