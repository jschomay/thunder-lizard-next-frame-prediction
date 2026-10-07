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
