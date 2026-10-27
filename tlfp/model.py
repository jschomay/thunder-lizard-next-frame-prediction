"""Approach A: a plain CNN that predicts the next state grid.

Input per cell: k frames x (8 terrain + 42 entity one-hot) + 5 key one-hot (same in every cell).
Output per cell: 8 terrain scores and 42 entity scores (logits) for the next frame.

Options for stage 3:
  split_entity (H11): instead of one 42-way guess, ask three questions per cell, "what's here"
      (empty / player / dino / skeleton), "which way does it face" (4) and "what level" (5).
      The three answers are recombined into the same 42 entity scores, so everything
      downstream (metrics, rollouts) is unchanged. Splitting stops the player's probability
      being spread over 20 look-alike classes, where "empty" can win even when P(player) > 50%.
  unet: a shrink-and-grow branch that sees the whole screen at 1/4 size, added to the
      full-size features. Its last layer starts at zero, so a warm-started model begins
      exactly as the plain CNN and learns to use the branch.
  memory (H18): a transformer branch that looks back over a long, strided history (e.g. the last
      8 frames, then every 8th back to 128). Each old frame is cut into 5x5 patches (81 tokens per
      frame, tagged with screen position and age); 81 "next frame" query tokens attend to all of
      them, and the result is added to the CNN's features. Its output starts at zero, so a warm
      start begins exactly as the CNN. The CNN still sees only the last k frames.
  diffuse (H16): masked diffusion for terrain. The net also sees a draft of the next frame's
      terrain in which some cells are known and the rest are "unknown" (MASK). Training hides a
      random share of the real next frame and scores only the hidden cells; generating starts
      fully hidden and fills the most confident cells over a few passes (diffuse_terrain), so
      later choices can agree with earlier ones instead of every cell guessing alone.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tlfp.data import N_ENTITY, N_TERRAIN

N_KEYS = 5
N_KIND, N_DIR, N_LEVEL = 4, 4, 5  # kind: 0 empty, 1 player, 2 dino, 3 skeleton
MASK = N_TERRAIN  # "unknown" terrain in a diffusion draft
WORLD_PAD = 200  # H18 world positions: how far (cells) a remembered patch may be from the current screen


def split_ids(e: torch.Tensor):
    """entity id -> (kind, dir, level); dir/level are 0 where there's no dino"""
    dino = (e >= 1) & (e <= 40)
    kind = torch.where(e == 0, 0, torch.where(e == 41, 3, torch.where(e <= 20, 1, 2)))
    rel = torch.where(e <= 20, e - 1, e - 21).clamp(min=0)
    return kind, torch.where(dino, rel % 4, 0), torch.where(dino, rel // 4, 0)


def combine(kind: torch.Tensor, dir_: torch.Tensor, level: torch.Tensor) -> torch.Tensor:
    """log-probs [B,4|4|5,H,W] -> 42 entity log-probs [B,42,H,W] (id = base + level*4 + dir)"""
    k, d, l = kind.log_softmax(1), dir_.log_softmax(1), level.log_softmax(1)
    dl = (l[:, :, None] + d[:, None]).flatten(1, 2)   # [B, 20, H, W], index level*4 + dir
    return torch.cat([k[:, :1], k[:, 1:2] + dl, k[:, 2:3] + dl, k[:, 3:4]], 1)


def _group_matrix() -> torch.Tensor:
    """[42, 13] 0/1 matrix: which kind (4), facing (4) and level (5) group each entity id belongs to"""
    g = torch.zeros(N_ENTITY, N_KIND + N_DIR + N_LEVEL)
    k, d, l = split_ids(torch.arange(N_ENTITY))
    g[torch.arange(N_ENTITY), k] = 1
    dino = (k == 1) | (k == 2)
    g[torch.arange(N_ENTITY)[dino], N_KIND + d[dino]] = 1
    g[torch.arange(N_ENTITY)[dino], N_KIND + N_DIR + l[dino]] = 1
    return g


GROUPS = _group_matrix()


def group_entity(e: torch.Tensor):
    """42 scores -> (kind 4, dir 4, level 5) log-probabilities. Each group's probability is the sum
    of its members', e.g. "player" = all 20 player classes together. One matrix multiply (fast on MPS)."""
    p = torch.einsum("bchw,cg->bghw", e.softmax(1), GROUPS.to(e.device))
    lp = p.clamp(min=1e-12).log()
    return lp[:, :N_KIND], lp[:, N_KIND:N_KIND + N_DIR], lp[:, N_KIND + N_DIR:]


def build_model(c: dict) -> "CNN":
    return CNN(c["k"], c["width"], c["depth"], copy_prior=c.get("copy_prior", False),
               split_entity=c.get("split_entity", False), unet=c.get("unet", False),
               fates=c.get("fates", False), recurrent=c.get("recurrent", False),
               mem_pad=c.get("mem_pad", 0), diffuse=c.get("diffuse", False),
               noise_level=c.get("noise_level", False), memory=c.get("memory"))


class ConvGRUCell(nn.Module):
    """A GRU where every gate is a 3x3 convolution: each cell keeps a memory vector that the net
    can update, keep or overwrite each tick, and neighbours can share memory."""

    def __init__(self, width: int):
        super().__init__()
        self.gates = nn.Conv2d(2 * width, 2 * width, 3, padding=1)
        self.cand = nn.Conv2d(2 * width, width, 3, padding=1)

    def forward(self, x, h):
        z, r = torch.sigmoid(self.gates(torch.cat([x, h], 1))).chunk(2, 1)
        n = torch.tanh(self.cand(torch.cat([x, r * h], 1)))
        return (1 - z) * h + z * n


def encode_input(context: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """context: int [B, k, 2, 45, 45], keys: int [B] -> float [B, k*50 + 5, 45, 45]"""
    B, k, _, H, W = context.shape
    terrain = F.one_hot(context[:, :, 0], N_TERRAIN)          # [B, k, H, W, 8]
    entity = F.one_hot(context[:, :, 1], N_ENTITY)            # [B, k, H, W, 42]
    frames = torch.cat([terrain, entity], dim=-1)             # [B, k, H, W, 50]
    frames = frames.permute(0, 1, 4, 2, 3).reshape(B, k * 50, H, W)
    key = F.one_hot(keys, N_KEYS)[:, :, None, None].expand(B, N_KEYS, H, W)  # copied into every cell
    return torch.cat([frames, key], dim=1).float()


class MemoryBranch(nn.Module):
    """H18: patches of every history frame -> transformer -> per-cell features for the next frame."""

    def __init__(self, n_frames: int, width: int, d: int = 256, layers: int = 4, heads: int = 8, patch: int = 5,
                 stem: int = 32, world: bool = False):
        super().__init__()
        # world (option B): tag each patch with where it is *now*, i.e. its screen position shifted by
        # how far the view has scrolled since that frame (rel), so the same place gets the same tag
        self.world = world
        if world:
            self.wrow, self.wcol = nn.Embedding(2 * WORLD_PAD + 45, d), nn.Embedding(2 * WORLD_PAD + 45, d)
        self.p, self.n = patch, (45 // patch) ** 2
        self.stem = nn.Sequential(nn.Conv2d(N_TERRAIN + N_ENTITY, stem, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(stem, stem, 3, padding=1))
        self.embed = nn.Linear(stem * patch * patch, d)
        self.pos = nn.Parameter(torch.randn(self.n, d) * 0.02)        # screen position of a patch
        self.age = nn.Parameter(torch.randn(n_frames + 1, d) * 0.02)  # which history frame (+1: the query)
        self.key = nn.Embedding(N_KEYS, d)
        self.query = nn.Parameter(torch.randn(self.n, d) * 0.02)      # "what goes here next frame?"
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(d, width * patch * patch)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.width = width

    def patches(self, x):  # [N, C, 45, 45] -> [N, 81, C*p*p] (reshapes only: exports to ONNX)
        N, C, p, g = len(x), x.shape[1], self.p, 45 // self.p
        return x.reshape(N, C, g, p, g, p).permute(0, 2, 4, 1, 3, 5).reshape(N, g * g, C * p * p)

    def unpatch(self, y):  # [B, 81, W*p*p] -> [B, W, 45, 45]
        B, p, g = len(y), self.p, 45 // self.p
        return y.reshape(B, g, g, -1, p, p).permute(0, 3, 1, 4, 2, 5).reshape(B, -1, 45, 45)

    def world_pos(self, rel):
        """rel: int [B, T, 2], each frame's screen origin in the current frame's screen coordinates
        -> [B, T, n, d] position tags of its patches (patch centres, shifted by rel)"""
        g = 45 // self.p
        c = torch.arange(g, device=rel.device) * self.p + self.p // 2
        rows = (c[:, None].expand(g, g).reshape(-1) + rel[..., :1]).clamp(-WORLD_PAD, 44 + WORLD_PAD) + WORLD_PAD
        cols = (c[None, :].expand(g, g).reshape(-1) + rel[..., 1:]).clamp(-WORLD_PAD, 44 + WORLD_PAD) + WORLD_PAD
        return self.wrow(rows) + self.wcol(cols)

    def tokens(self, context, keys, rel=None, oh=None):
        """all tokens: history patches then the next-frame queries, [B, (T+1)*n, d].
        oh: the same frames already one-hot, float [B, T, 50, 45, 45] (the browser builds it)"""
        if oh is None:
            oh = torch.cat([F.one_hot(context[:, :, 0], N_TERRAIN), F.one_hot(context[:, :, 1], N_ENTITY)], -1)
            oh = oh.permute(0, 1, 4, 2, 3).float()
        B, T = oh.shape[:2]
        f = self.stem(oh.reshape(B * T, -1, 45, 45))
        if self.world:
            rel = torch.zeros(B, T, 2, dtype=torch.long, device=oh.device) if rel is None else rel
            pos, qpos = self.world_pos(rel), self.world_pos(torch.zeros_like(rel[:, :1]))[:, 0]
        else:
            pos, qpos = self.pos, self.pos
        tok = self.embed(self.patches(f)).reshape(B, T, self.n, -1) + pos + self.age[:T, None]
        q = (self.query + qpos + self.age[T] + self.key(keys)[:, None]).reshape(B, 1, self.n, -1)
        return torch.cat([tok, q], 1).flatten(1, 2)

    def forward(self, context, keys, rel=None, oh=None):
        x = self.tf(self.tokens(context, keys, rel, oh))[:, -self.n:]   # read the query tokens
        return self.unpatch(self.out(x))


def pick_context(model, history: list) -> np.ndarray:
    """the frames a model reads from a rollout history (oldest first); missing old frames repeat the earliest"""
    n = len(history)
    return np.stack([history[max(0, n - 1 - a)] for a in model.offsets])


class CNN(nn.Module):
    def __init__(self, k: int = 2, width: int = 64, depth: int = 6, copy_prior: bool = False,
                 split_entity: bool = False, unet: bool = False, fates: bool = False,
                 recurrent: bool = False, mem_pad: int = 0, diffuse: bool = False,
                 noise_level: bool = False, memory: dict | None = None):
        super().__init__()
        self.k = k
        # frame ages the model reads, oldest first (0 = the last frame); the CNN body uses the last k
        self.offsets = sorted(set(range(k)) | set(memory["offsets"] if memory else []), reverse=True)
        self.memory = MemoryBranch(len(self.offsets), width, **{x: v for x, v in memory.items() if x != "offsets"}) if memory else None
        # copy prior (H10): add a learned bonus to the class each cell had in the last frame,
        # so "nothing changes" is the default and the network only has to learn changes
        self.copy_prior = copy_prior
        if copy_prior:
            self.copy_strength = nn.Parameter(torch.tensor([5.0, 5.0]))
        in_ch = k * (N_TERRAIN + N_ENTITY) + N_KEYS
        layers: list[nn.Module] = [nn.Conv2d(in_ch, width, 3, padding=1), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Conv2d(width, width, 3, padding=1), nn.ReLU()]
        self.body = nn.Sequential(*layers)
        # 1x1 convs: per-cell readout of the features into class scores
        self.terrain_head = nn.Conv2d(width, N_TERRAIN, 1)
        self.split_entity = split_entity
        # split mode reuses the same 42 scores, but groups them into the three questions (see
        # group_entity), so a warm start from a flat-head model begins with the same predictions
        self.entity_head = nn.Conv2d(width, N_ENTITY, 1)
        # H13: per entity, "what happens to you?" (stay/up/right/down/left/die/gone), read at the
        # cells that hold an entity in the last frame (see tlfp/moves.py)
        self.fates = fates
        if fates:
            self.fate_head = nn.Conv2d(width, 7, 1)
            nn.init.zeros_(self.fate_head.weight); nn.init.zeros_(self.fate_head.bias)
        # H15: a memory carried from tick to tick (ConvGRU) on top of the k-frame window. Its output
        # layer starts at zero, so a warm-started model begins exactly as before. The memory lives
        # on the model: set .carry = True and .state = None at the start of a sequence/rollout;
        # with carry off (one-step evaluation) every call starts from an empty memory.
        self.recurrent, self.carry, self.state = recurrent, False, None
        # H15b: the memory canvas extends mem_pad cells past each screen edge (22 -> 89x89, about 2x
        # the view), so what scrolls off screen can stay in memory; the net must learn to shift it
        self.mem_pad = mem_pad
        if recurrent:
            self.gru = ConvGRUCell(width)
            self.mem_out = nn.Conv2d(width, width, 1)
            nn.init.zeros_(self.mem_out.weight); nn.init.zeros_(self.mem_out.bias)
        # H16: the draft (8 terrain + unknown one-hot, plus the share still unknown) mixed into the
        # features before the U-Net; starts at zero, so a warm start begins as the one-shot model
        self.diffuse, self.steps = diffuse, 4
        if diffuse:
            self.draft_in = nn.Sequential(nn.Conv2d(N_TERRAIN + 2, width, 3, padding=1), nn.ReLU(),
                                          nn.Conv2d(width, width, 3, padding=1))
            nn.init.zeros_(self.draft_in[-1].weight); nn.init.zeros_(self.draft_in[-1].bias)
        # GameNGen-style: the net is told how damaged its input frames are (0 = clean; rollouts use 0
        # or a small value), so it can learn to repair damage. Starts at zero.
        self.noise_level = noise_level
        if noise_level:
            self.level_in = nn.Linear(1, width)
            nn.init.zeros_(self.level_in.weight); nn.init.zeros_(self.level_in.bias)
        self.rollout_level = 0.0
        self.unet = unet
        if unet:  # 45 -> 23 -> 12 and back; each 3x3 at 12x12 sees ~1/4 of the screen
            self.down = nn.Sequential(
                nn.Conv2d(width, width, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
                nn.Conv2d(width, width, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
                nn.Conv2d(width, width, 3, padding=1), nn.ReLU())
            self.up = nn.Sequential(nn.Conv2d(2 * width, width, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(width, width, 1))
            nn.init.zeros_(self.up[-1].weight); nn.init.zeros_(self.up[-1].bias)

    def forward(self, context: torch.Tensor, keys: torch.Tensor):
        t, e = self.forward_parts(context, keys)[:2]
        if self.split_entity:
            e = combine(*e)
        return t, e

    def forward_parts(self, context: torch.Tensor, keys: torch.Tensor, draft: torch.Tensor | None = None,
                      level: torch.Tensor | None = None):
        """context: [B, len(offsets), 2, 45, 45], oldest first (= [B, k, ...] without memory)"""
        full = context
        context = context[:, -self.k:]
        """terrain logits, and entity logits (42) or (kind, dir, level) when split.
        draft (diffuse only): int [B, 45, 45] next-frame terrain, MASK where unknown; None = all unknown"""
        h = self.body(encode_input(context, keys))
        if self.memory is not None:
            h = h + self.memory(full, keys, getattr(self, "rel", None))  # rel: set by the caller (world mode)
        if self.noise_level:  # level: [B] damage level of the input frames; None = rollout_level
            if level is None:
                level = torch.full((len(h),), self.rollout_level, device=h.device)
            h = h + self.level_in(level[:, None].float())[:, :, None, None]
        if self.diffuse:
            if draft is None:
                draft = torch.full_like(context[:, -1, 0], MASK)
            d = F.one_hot(draft, N_TERRAIN + 1).permute(0, 3, 1, 2).float()
            share = d[:, -1:].mean((2, 3), keepdim=True).expand_as(d[:, -1:])
            h = h + self.draft_in(torch.cat([d, share], 1))
        if self.recurrent:
            p = self.mem_pad
            x = F.pad(h, (p, p, p, p)) if p else h
            prev = self.state if (self.carry and self.state is not None) else torch.zeros_like(x)
            mem = self.gru(x, prev)
            if self.carry:
                self.state = mem
            h = h + self.mem_out(mem[:, :, p:p + h.shape[-2], p:p + h.shape[-1]] if p else mem)
        if self.unet:
            g = F.interpolate(self.down(h), size=h.shape[-2:], mode="nearest")
            h = h + self.up(torch.cat([h, g], 1))
        last = context[:, -1]
        oh = lambda x, n: F.one_hot(x, n).permute(0, 3, 1, 2)
        t = self.terrain_head(h)
        if self.copy_prior:
            t = t + self.copy_strength[0] * oh(last[:, 0], N_TERRAIN)
        e = self.entity_head(h)
        if self.copy_prior:
            e = e + self.copy_strength[1] * oh(last[:, 1], N_ENTITY)
        e = group_entity(e) if self.split_entity else e
        return (t, e, self.fate_head(h)) if self.fates else (t, e)

    @torch.no_grad()
    def predict(self, context: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """Most likely next frame: int [B, 2, 45, 45]"""
        t, e = self(context, keys)
        return torch.stack([t.argmax(1), e.argmax(1)], dim=1)


@torch.no_grad()
def diffuse_terrain(model, context, keys, steps: int | None = None, sample=None, gen=None, temp: float = 1.0,
                    order_noise: float | None = None):
    """H16: generate next-frame terrain in a few passes. Each pass guesses every unknown cell, then
    keeps the most confident guesses and leaves the rest unknown (a cosine schedule: few cells are
    fixed early, more later). sample: bool [B,45,45] cells drawn from the probabilities (at temp)
    instead of argmax. order_noise (MaskGIT): random noise on the confidence ranking, fading to 0 by the
    last pass, so which cells are fixed first is partly random (the values stay argmax). Returns (terrain [B,45,45], drafts per pass, the last pass's outputs)."""
    steps = steps or model.steps
    order_noise = getattr(model, "order_noise", 0.0) if order_noise is None else order_noise
    B, _, _, H, W = context.shape
    draft = torch.full((B, H, W), MASK, dtype=torch.long, device=context.device)
    drafts = []
    for s in range(steps):
        out = model.forward_parts(context, keys, draft)
        p = out[0].softmax(1)
        choice = p.argmax(1)
        if sample is not None and sample.any():
            q = (out[0] / temp).softmax(1).permute(0, 2, 3, 1).reshape(-1, p.shape[1]).cpu()
            drawn = torch.multinomial(q, 1, generator=gen).reshape(B, H, W).to(choice.device)
            choice = torch.where(sample.to(choice.device), drawn, choice)
        conf = p.gather(1, choice[:, None])[:, 0]
        unknown = draft == MASK
        rank = conf
        if order_noise:
            u = torch.rand(conf.shape, generator=gen).to(conf.device).clamp(1e-9, 1 - 1e-9)
            rank = conf.log() + order_noise * (1 - s / steps) * -torch.log(-torch.log(u))
        rank = torch.where(unknown, rank, torch.full_like(rank, float("inf")))
        left = int(H * W * math.cos(math.pi / 2 * (s + 1) / steps))  # still unknown after this pass
        new = torch.where(unknown, choice, draft)
        if left > 0:
            low = rank.flatten(1).argsort(1)[:, :left]
            new = new.flatten(1).scatter(1, low, MASK).reshape(B, H, W)
        draft = new
        drafts.append((draft.clone(), conf.clamp(max=1.0)))
    return draft, drafts, out
