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

const STAGES: Record<number, Omit<EpisodeOptions, "seed">> = {
  1: { numDinos: 0, lava: false },   // player only
  2: { numDinos: 0, lava: true },    // + lava
  3: { numDinos: 5, lava: true },    // + a few dinos
  4: { numDinos: 60, lava: true },   // full game
}

const args = Object.fromEntries(
  process.argv.slice(2).join(" ").split("--").filter(Boolean).map(a => a.trim().split(/\s+/)))
const stage = Number(args.stage ?? 1)
const episodes = Number(args.episodes ?? 10)
const ticks = Number(args.ticks ?? 500)
const firstSeed = Number(args.seed ?? 1)
const outDir = args.out ?? join(__dirname, "..", "data", `stage${stage}`)

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
  const bot = new StickyBot()
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
    seed, stage, ...STAGES[stage], ticks: t, outcome: game.done || "timeout",
    keys: KEYS, terrain: TERRAIN, entityClasses: ENTITY_CLASSES,
  }, null, 2))
  totalTicks += t
  game.dispose()
}
const secs = (Date.now() - started) / 1000
console.log(`stage ${stage}: ${episodes} episodes, ${totalTicks} ticks in ${secs.toFixed(1)}s (${(totalTicks / secs).toFixed(0)} ticks/s) -> ${outDir}`)
