# Thunder Lizard, imagined by a neural net

Can a small neural net learn a game well enough that you can play it **inside the net**, with no game engine running at all?

<p align="center">
  <img src="media/tl-frame-predict-poc.gif" width="420" alt="Playing Thunder Lizard inside the neural net: every frame is predicted by the model">
  <br><em>Every frame here is drawn by the model. It sees the last few frames and the key you're pressing, and predicts the next one.</em>
</p>

**▶ Play it in your browser: [thunder-lizard-frame-prediction.vercel.app](https://thunder-lizard-frame-prediction.vercel.app/)** (best in Chrome or Edge; runs on your GPU via WebGPU)

This is an early, open learning project: I'm new to machine learning, and this repo is where I try ideas, keep score, and write down what worked and what didn't.

## The original game

<p align="center">
  <img src="https://img.itch.zone/aW1hZ2UvMzQxMDA0OC8yMzM3Mzk2OS5qcGc=/347x500/Q3s4mx.jpg" width="300" alt="Thunder Lizard title screen">
  <img src="https://img.itch.zone/aW1hZ2UvMzQxMDA0OC8yMzM3Mzk3MS5qcGc=/347x500/lF%2Bucb.jpg" width="300" alt="Thunder Lizard island map">
</p>

[**Thunder Lizard: Eat or Be Eaten**](https://enegames.itch.io/thunder-lizard-eat-or-be-eaten) is an ASCII game I made: you're a dinosaur on a volcanic island. Eat smaller dinos, avoid bigger ones, and stay ahead of the spreading lava. The game lives in [`game/`](https://github.com/jschomay/thunder-lizard) (a submodule).

## How it works

1. **Record.** A headless version of the game is played by bots for hundreds of thousands of ticks. Every tick is saved as a 45×45 grid: one layer for terrain (ocean, water, dirt, grass, shrub, jungle, lava, burnt) and one for entities (the player and the dinos, with their facing and level, plus skeletons), along with the key pressed.
2. **Train.** A convolutional net (a CNN with a U-Net branch, about 600k parameters) learns: *given the last 4 frames and a key, what's the next frame?*
3. **Play.** In the browser, the model runs in a loop on its own output. The first frames come from a real game; everything after that is the net's imagination. Code only turns the model's output into a frame (picking or sampling each cell) and draws it.

This makes it a small **world model**, or neural game engine: the same idea as [GameNGen](https://gamengen.github.io/) (DOOM inside a diffusion model) and Genie, at toy scale on a laptop.

**Ground rules I set for the project:** the net should predict frames, not run a game behind the scenes. So fixes have to come from the model and its training, not from game logic added to the code. It's also fine for the imagined game to drift away from the real one, as long as it still feels like the game.

## What we're trying

Each idea is written up as a hypothesis, tested, and scored:

- **[HYPOTHESES.md](HYPOTHESES.md):** every idea with its status (🟢 supported, 🔴 refuted, 🟡 open) and the evidence.
- **[JOURNAL.md](JOURNAL.md):** the lab notebook: each run, its numbers, the surprises, the mistakes and the decisions.

Some highlights so far:

| Idea | Result |
|---|---|
| More history (4 frames vs 1) | 🟢 needed: the player moves every other tick, which one frame can't show |
| "Copy the last frame by default" | 🟢 much faster learning; exact frames 47% vs 10% |
| Split "what's here" from "facing / level" | 🟢 the player stopped vanishing |
| Predict each dino's *fate* (stay / move / die) | 🟢 dinos stopped duplicating and vanishing |
| A recurrent memory (ConvGRU) for continuity | 🔴 no gain: training barely rewards remembering |
| Masked diffusion, drawing each frame in passes | 🔴 1–2 points at 4× the compute |
| Damaging the input frames during training (GameNGen-style) | 🟡 fewer stray cells, but dinos got worse |
| A transformer reading 128 ticks of history | 🟡 better scores (continuity 70 → 79), no difference in play |
| Training on the model's own rollouts | 🔴 traded smear for smoothing; no difference in play |

Every model is compared on a fixed **scorecard** ([`tlfp/scorecard.py`](tlfp/scorecard.py)) of things that matter for play, not just accuracy:
- **Continuity:** walk away and come back. Is it the same place?
- **Smear:** does one terrain type slowly take over the screen?
- **Static:** stray single cells.
- **Scrolling:** does the view move when you press a key?
- **Dino movement.**
- **Speed:** milliseconds per frame.

<p align="center">
  <img src="media/wandb-parallel-coordinates.png" width="800" alt="Weights & Biases parallel-coordinates chart comparing runs across settings and scorecard metrics">
  <br><em>Each line is one training run (Weights & Biases), traced from its settings on the left to its scorecard on the right. Colour is continuity. Improving one score often cost another.</em>
</p>

## What we learned

The net imagines the game well for a few seconds: movement, scrolling, lava and dinos all work. **What it can't do is stay coherent while you travel.** After you move some distance, the terrain smears and places you come back to have changed. Every attempt to fix that (diffusion, recurrent memory, a transformer with long history, training on its own rollouts) improved the scorecard but didn't change how it plays.

In hindsight:
- **Measure what you feel, first.** One-step accuracy and short rollouts kept saying "better" while playing said "same". A score for long walks came last; it should have come first.

<p align="center">
  <img src="media/wandb-dynamic-acc.png" width="640" alt="Validation accuracy curves for ten runs, mostly rising over training">
  <br><em>Accuracy on predicting a single frame from real ones kept climbing run after run. How the game felt to play barely moved.</em>
</p>

- **Train longer.** Most runs got 2–3 epochs because they ran on a laptop. On a cloud GPU (Colab A100) the CNN trains an epoch in about a minute, and more epochs were the most reliable gain we found.
- **Stability before memory.** Remembering doesn't help if what you remember is already smeared.
- **Scale.** This is ~200k frames and up to 6M parameters. GameNGen used 900M frames and 860M parameters and still held only ~3 seconds of memory. Long-horizon coherence is an open problem for much bigger models too.

The full story, with every number and wrong turn, is in [JOURNAL.md](JOURNAL.md).

## Repo layout

| Path | What's there |
|---|---|
| `tlfp/` | Python: model, training, rollouts, scorecard, export to ONNX |
| `harness/` | Bot players that record training data from the headless game |
| `app/` | The browser app ([live](https://thunder-lizard-frame-prediction.vercel.app/)): ONNX Runtime Web on WebGPU, ~16 ms per frame |
| `.whiteboard/` | Interactive explainer pages used while working: how the model works, inside the layers, predictions, diffusion, and more |
| `scorecards/` | Scorecard results per model |

## Running it

```bash
uv sync
# record data (from game/, see harness/record.ts for scenarios)
cd game && npx vite-node --root .. ../harness/record.ts -- --stage 3 --episodes 200 --ticks 500
# train
uv run python -m tlfp.train --data data/stage3-100k --k 4 --copy-prior --split-entity --fates --unet --dino-weight 20 --name my-run
# score, then export to the browser app
uv run python -m tlfp.scorecard checkpoints/my-run-last.pt --eps 24
uv run python -m tlfp.export_onnx checkpoints/my-run-last.pt
cd app && python3 -m http.server
```

The CNN trains on a Mac (Apple GPU) in about 20 minutes per 3 epochs. Transformer runs need a cloud GPU: [`colab/train_transformer.ipynb`](colab/train_transformer.ipynb) runs a list of experiments on Colab and scores them there. Runs are logged to Weights & Biases.

Built with a lot of help from Claude (Claude Code).
