# Hypotheses

Status: 🟡 open · 🟢 supported · 🔴 refuted. Each one links to the journal entry or notebook with the evidence.

| ID | Hypothesis | How to test | Status |
|---|---|---|---|
| H1 | Predicting a semantic state grid (dino = body cell + direction + level), then rendering, beats predicting rendered frames, for the same model size. | Train approach A on both encodings. Compare changed-cell accuracy and rollout quality. | 🟡 |
| H2 | CNN (A) beats transformer (B) on small data. B wins on dino behavior at scale. LLM (C) is worst and slowest. | Same dataset and metrics across A/B/C. Vary dataset size. | 🟡 |
| H3 | 2–4 past frames are needed to predict dino motion; 1 is enough for the player and terrain. | Sweep k ∈ {1,2,4}. Compare accuracy on dino cells. | 🟡 |
| H4 | Warm-starting each stage from the previous stage's weights trains faster than starting from scratch. | Stage N from scratch vs. from the stage N-1 checkpoint. Compare loss curves. | 🟡 |
| H5 | Sampling from the predicted probabilities (instead of argmax) gives believable lava spread over rollouts. | Compare lava growth stats between real and dreamed rollouts. | 🟡 |
