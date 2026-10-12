// Records episodes of bot play as .npy files for training.
//   npx vite-node --root .. ../harness/record.ts -- --stage 1 --episodes 50 --ticks 500
// Output per episode (data/stage<N>/ep_<seed>/):
//   obs.npy      uint8 [T+1, 2, 45, 45]  layer 0 = terrain class, layer 1 = entity class
//   actions.npy  uint8 [T]               index into KEYS (action[t] takes obs[t] -> obs[t+1])
//   meta.json    options, vocab, outcome

import { mkdirSync, writeFileSync } from "fs"
import { join } from "path"
import * as ROT from "../game/lib/rotjs"
import { HeadlessGame, KEYS, Key, TERRAIN, ENTITY_CLASSES, VIEW, EpisodeOptions } from "./headless"
import * as Terrain from "../game/src/entities/terrain"
import XY from "../game/src/xy"

const STAGES: Record<number, Omit<EpisodeOptions, "seed">> = {
  1: { numDinos: 0, lava: false },   // player only
  2: { numDinos: 0, lava: true },    // + lava
  3: { numDinos: 25, lava: true },   // + some dinos (5 left dinos off-screen 82% of ticks)
  4: { numDinos: 60, lava: true },   // full game
}

const args = Object.fromEntries(
  process.argv.slice(2).join(" ").split("--").filter(Boolean).map(a => a.trim().split(/\s+/)))
const stage = Number(args.stage ?? 1)
const episodes = Number(args.episodes ?? 10)
const ticks = Number(args.ticks ?? 500)
const firstSeed = Number(args.seed ?? 1)
const outDir = args.out ?? join(__dirname, "..", "data", `stage${stage}`)
// --scenario lava-still: player starts a few cells from lava and never moves (clean lava-growth test, H12)
// --scenario backtrack: walks out and comes back the same way (continuity training/testing, H15b)
// --scenario coast: player starts near ocean and often pushes into it (blocked moves were the top error)
const scenario = args.scenario

// "Sticky" random walker: holds a direction for a while, sometimes idles.
// Placeholder for the heuristic bot and human recordings (see H6).
class StickyBot {
  private key: Key = "NONE"
  private left = 0
  act(): Key {
    if (this.left-- <= 0) {
      this.key = ROT.RNG.getPercentage() <= 20 ? "NONE" : ROT.RNG.getItem(KEYS.slice(1) as unknown as Key[])!
      this.left = ROT.RNG.getUniformInt(3, 20)
    }
    return this.key
  }
}

// Like StickyBot, but most of the time it heads for the nearest ocean in view and keeps pushing
// into it, so the data has plenty of "key held, player blocked" ticks
class CoastBot extends StickyBot {
  constructor(private game: HeadlessGame) { super() }
  private key2: Key = "NONE"
  private left2 = 0
  act(): Key {
    if (this.left2-- > 0) return this.key2
    if (ROT.RNG.getPercentage() > 70) { this.left2 = 0; return super.act() }
    const [terrain, entity] = this.game.observe()
    const me = entity.findIndex(v => v >= 1 && v <= 20)
    const py = Math.floor(me / VIEW), px = me % VIEW
    let best: Key = "NONE", bd = 1e9
    for (let i = 0; i < VIEW * VIEW; i++) if (terrain[i] === 0) {
      const dy = Math.floor(i / VIEW) - py, dx = (i % VIEW) - px, d = Math.abs(dy) + Math.abs(dx)
      if (d < bd) { bd = d; best = Math.abs(dy) > Math.abs(dx) ? (dy < 0 ? "UP" : "DOWN") : (dx < 0 ? "LEFT" : "RIGHT") }
    }
    this.key2 = best === "NONE" ? super.act() : best
    this.left2 = ROT.RNG.getUniformInt(4, 15)
    return this.key2
  }
}

// Walks a direction for a while, then retraces its steps (same keys reversed, same timing), so the
// view returns to places it has seen. Occasionally idles or wanders like StickyBot.
class BacktrackBot extends StickyBot {
  private plan: Key[] = []
  act(): Key {
    if (!this.plan.length) {
      if (ROT.RNG.getPercentage() <= 25) return super.act()
      const opposite: Record<string, Key> = { UP: "DOWN", DOWN: "UP", LEFT: "RIGHT", RIGHT: "LEFT" }
      const legs = ROT.RNG.getUniformInt(1, 2)
      const out: Key[] = []
      for (let i = 0; i < legs; i++) {
        const k = ROT.RNG.getItem(["UP", "RIGHT", "DOWN", "LEFT"] as Key[])!
        for (let n = ROT.RNG.getUniformInt(10, 40); n > 0; n--) out.push(k)
      }
      const pause: Key[] = Array(ROT.RNG.getUniformInt(0, 8)).fill("NONE")
      this.plan = [...out, ...pause, ...[...out].reverse().map(k => opposite[k])]
    }
    return this.plan.shift()!
  }
}

// move the player onto a land cell next to ocean
function placeNearOcean(game: HeadlessGame) {
  const l = game.level
  const ocean = [...l.map.getTagged(Terrain.Ocean)].map(e => e.getXY())
  for (const c of ROT.RNG.shuffle(ocean)) {
    for (const [dx, dy] of [[2, 0], [-2, 0], [0, 2], [0, -2]]) {
      const xy = new XY(c.x + dx, c.y + dy)
      const t = l.map.at(xy)
      if ((t instanceof Terrain.Grass || t instanceof Terrain.Dirt) && !l.dinos.at(xy)) { l.playerDino.moveTo(xy); return }
    }
  }
}

// move the player onto a grass cell `dist` steps from the lava frontier
function placeNearLava(game: HeadlessGame, dist: number) {
  const l = game.level
  const lava = [...l.map.getTagged(Terrain.Lava)].map(e => e.getXY())
  for (const c of ROT.RNG.shuffle(lava)) {
    for (const [dx, dy] of [[dist, 0], [-dist, 0], [0, dist], [0, -dist]]) {
      const xy = new XY(c.x + dx, c.y + dy)
      if (l.map.at(xy) instanceof Terrain.Grass && !l.dinos.at(xy)) { l.playerDino.moveTo(xy); return }
    }
  }
}

function npy(path: string, data: Uint8Array, shape: number[]) {
  let header = `{'descr': '|u1', 'fortran_order': False, 'shape': (${shape.join(", ")}${shape.length === 1 ? "," : ""}), }`
  header += " ".repeat(64 - ((10 + header.length + 1) % 64)) + "\n"
  const pre = Buffer.alloc(10)
  pre.write("\x93NUMPY", 0, "latin1"); pre[6] = 1; pre[7] = 0; pre.writeUInt16LE(header.length, 8)
  writeFileSync(path, Buffer.concat([pre, Buffer.from(header, "latin1"), Buffer.from(data)]))
}

const started = Date.now()
let totalTicks = 0
for (let e = 0; e < episodes; e++) {
  const seed = firstSeed + e
  const game = new HeadlessGame({ seed, ...STAGES[stage] })
  const bot = scenario === "lava-still" ? { act: (): Key => "NONE" } : scenario === "coast" ? new CoastBot(game) : scenario === "backtrack" ? new BacktrackBot() : new StickyBot()
  if (scenario === "lava-still") placeNearLava(game, 12)
  if (scenario === "coast") placeNearOcean(game)
  const frame = 2 * VIEW * VIEW
  const obs = new Uint8Array((ticks + 1) * frame)
  const actions = new Uint8Array(ticks)

  const write = (t: number) => { const [a, b] = game.observe(); obs.set(a, t * frame); obs.set(b, t * frame + VIEW * VIEW) }
  write(0)
  let t = 0
  for (; t < ticks && !game.done; t++) {
    const key = bot.act()
    actions[t] = KEYS.indexOf(key)
    game.step(key)
    write(t + 1)
  }

  const dir = join(outDir, `ep_${String(seed).padStart(5, "0")}`)
  mkdirSync(dir, { recursive: true })
  npy(join(dir, "obs.npy"), obs.subarray(0, (t + 1) * frame), [t + 1, 2, VIEW, VIEW])
  npy(join(dir, "actions.npy"), actions.subarray(0, t), [t])
  writeFileSync(join(dir, "meta.json"), JSON.stringify({
    seed, stage, ...STAGES[stage], scenario: scenario ?? null, ticks: t, outcome: game.done || "timeout",
    keys: KEYS, terrain: TERRAIN, entityClasses: ENTITY_CLASSES,
  }, null, 2))
  totalTicks += t
  game.dispose()
}
const secs = (Date.now() - started) / 1000
console.log(`stage ${stage}: ${episodes} episodes, ${totalTicks} ticks in ${secs.toFixed(1)}s (${(totalTicks / secs).toFixed(0)} ticks/s) -> ${outDir}`)
