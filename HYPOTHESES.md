# Hypotheses

Status: 🟡 open · 🟢 supported · 🔴 refuted. Each one links to the journal entry or notebook with the evidence.

| ID | Hypothesis | How to test | Status |
|---|---|---|---|
| H1 | Predicting a semantic state grid (dino = body cell + direction + level), then rendering, beats predicting rendered frames, for the same model size. | Train approach A on both encodings. Compare changed-cell accuracy and rollout quality. | 🟡 |
| H2 | CNN (A) beats transformer (B) on small data. B wins on dino behavior at scale. LLM (C) is worst and slowest. | Same dataset and metrics across A/B/C. Vary dataset size. | 🟡 |
| H3 | History is needed: k=1 can't infer the hidden movement phase (the player moves every 2nd tick with buffered input), so it plateaus. k≥2 nearly solves stage 1. Dinos need 2–4 frames for momentum. | Sweep k ∈ {1,2,3,4}. Compare changed-cell accuracy on stage 1 and dino-cell accuracy on stage 3. *(Revised 2026-10-05 after the harness finding.)* Early hint: a hand-written 2-frame rule beats the 1-frame rule on stage 1, 68.6% vs 40.1% dynamic accuracy. **Result 2026-10-05: k=1 peaked at 22.6% dynamic acc; k=2 reached 65.5% and was still rising.** k=3/4 not yet tested. | 🟢 supported (k=1 vs 2) |
| H4 | Warm-starting each stage from the previous stage's weights trains faster than starting from scratch. | Stage N from scratch vs. from the stage N-1 checkpoint. Compare loss curves. | 🟡 |
| H5 | Sampling from the predicted probabilities (instead of argmax) gives believable lava spread over rollouts. | Compare lava growth stats between real and dreamed rollouts. **2026-10-06:** independent per-cell sampling gives speckle noise and duplicated players, not believable terrain. | 🔴 refuted (simple form) |
| H6 | A model trained only on random-bot play does poorly on human play. Mixing in heuristic-bot and human data fixes it. *(2026-10-05: the shift is measured. Humans tap (median key run 1 tick vs the bot's 14), idle 62% vs 21%.)* | Train on bot-only vs. mixed data. Evaluate on a held-out set of human play. | 🟡 |
| H7 | Per-step accuracy doesn't predict human-judged "feel." Rollout metrics predict it better. | Blind side-by-side human ratings vs. automatic metrics across checkpoints. | 🟡 |
| H8 | Adding noise to context frames during training (GameNGen) reduces rollout drift. | Train with/without context corruption. Compare accuracy at rollout steps 10/50/100. | 🟡 |
| H9 | Up-weighting changed cells in the loss improves dynamic accuracy (the plain loss is dominated by the ~98% of cells that never change). | Same run with and without changed-cell loss weighting. | 🟡 |
| H10 | A "copy by default" model (predicts changes relative to the last frame) learns faster and gets far higher exact-frame accuracy than one that redraws every cell. | Residual/copy-prior head vs plain head, same budget. | 🟡 |
| H11 | Splitting the entity output into *what's here* (empty/player/dino/skeleton) and *its facing/level* stops the player vanishing in argmax rollouts. | Rollout player-survival rate, split head vs flat 42-class head. | 🟡 |
