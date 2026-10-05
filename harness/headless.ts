// Runs Thunder Lizard with no browser and steps it one main-loop tick at a time.
// Each tick it records the state grid around the player and the key held during it.

import * as ROT from "../game/lib/rotjs"
import MainLevel from "../game/src/level"
import * as Terrain from "../game/src/entities/terrain"
import Dino from "../game/src/entities/dino"
import awarenessSystem from "../game/src/systems/awareness"
import movementSystem, { keypressCb } from "../game/src/systems/movement"
import deplacementSystem from "../game/src/systems/displacement"
import scoreSystem from "../game/src/systems/score"
import { Movement } from "../game/src/components"
import { MAP_SIZE } from "../game/src/constants"
import XY from "../game/src/xy"
import { removeEntity, deleteWorld, resetGlobals } from "bitecs"
import * as Components from "../game/src/components"

// ---- vocabulary (fixed across all stages so checkpoints stay compatible) ----

export const VIEW = 45
export const TERRAIN = ["OCEAN", "WATER", "DIRT", "GRASS", "SHRUB", "JUNGLE", "LAVA", "BURNT"] as const
export const KEYS = ["NONE", "UP", "RIGHT", "DOWN", "LEFT"] as const
export type Key = typeof KEYS[number]
// entity layer: 0 empty, 1..20 player (level 1-5 x dir 0-3), 21..40 npc dino, 41 skeleton
export const ENTITY_CLASSES = 42
export const entityId = (isPlayer: boolean, level: number, dir: number) =>
  (isPlayer ? 1 : 21) + (Math.min(level, 5) - 1) * 4 + dir
export const SKELETON = 41

// ---- browser stand-ins ----

const fakeEl = { classList: { add() { }, remove() { } }, innerHTML: "", style: {} }
;(globalThis as any).document ??= { querySelector: () => fakeEl, getElementById: () => fakeEl }
const silent: any = new Proxy({}, { get: () => () => 0 })
const sounds: any = new Proxy({}, { get: () => silent })

// The real loops run on timers; headless stepping drives them instead.
const proto = MainLevel.prototype as any
for (const m of ["waterLoop", "lavaloop", "mainLoop", "zoomOut", "zoomIn", "dynamicZoom", "dinoSfx", "drawMap"]) {
  proto[m] = () => { }
}

export interface EpisodeOptions {
  seed: number
  numDinos: number   // npc dinos kept (the game spawns 60; extras are removed)
  lava: boolean      // whether lava spreads
}

export class HeadlessGame {
  level: MainLevel
  done: false | "dead" | "win" = false
  private lavaClock = 0

  constructor(private opts: EpisodeOptions) {
    resetEcs()
    ROT.RNG.setSeed(opts.seed)
    const game: any = { sounds, soundIds: {}, display: { draw() { }, drawOver() { } }, updateSize() { } }
    this.level = new MainLevel(game)
    this.level.setGameOver = () => { this.done = "dead" }
    this.level.doWinState = () => { this.done = "win" }

    // thin out npc dinos for early stages
    const all = this.level.dinos.withIn({ x: 0, y: 0, w: MAP_SIZE, h: MAP_SIZE }) as Dino[]
    const npcs = all.filter(d => d !== this.level.playerDino)
    for (const d of ROT.RNG.shuffle(npcs).slice(opts.numDinos)) {
      this.level.dinos.remove(d)
      removeEntity(this.level.dinoEcsWorld, d.id)
    }
  }

  /** free this game's ECS worlds (call before starting another game in this process) */
  dispose() {
    deleteWorld(this.level.dinoEcsWorld)
    deleteWorld(this.level.terrainEcsWorld)
  }

  /** advance one main-loop tick (70ms) with `key` held */
  step(key: Key) {
    const l = this.level
    if (key !== "NONE") keypressCb.call(l, "Arrow" + key[0] + key.slice(1).toLowerCase())
    awarenessSystem(l.dinoEcsWorld)
    movementSystem(l.dinoEcsWorld)
    deplacementSystem(l.terrainEcsWorld)
    scoreSystem(l.dinoEcsWorld)
    // lava runs every 50ms vs 70ms main loop: 1.4 lava ticks per step on average
    if (this.opts.lava) {
      this.lavaClock += 70
      while (this.lavaClock >= 50) { l.lavaTick(); this.lavaClock -= 50 }
    }
    if (l.playerDino.dead) this.done = "dead"
  }

  /** 45x45 window around the player, clamped to the map (lock zoom; see journal) */
  viewOrigin(): XY {
    const p = this.level.playerDino.getXY()
    const clamp = (v: number) => Math.max(0, Math.min(MAP_SIZE - VIEW, v))
    return new XY(clamp(p.x - Math.floor(VIEW / 2)), clamp(p.y - Math.floor(VIEW / 2)))
  }

  /** returns [terrain, entity] layers, each VIEW*VIEW uint8 in row-major order */
  observe(): [Uint8Array, Uint8Array] {
    const terrain = new Uint8Array(VIEW * VIEW)
    const entity = new Uint8Array(VIEW * VIEW)
    const o = this.viewOrigin()
    for (let y = 0; y < VIEW; y++) for (let x = 0; x < VIEW; x++) {
      terrain[y * VIEW + x] = terrainClass(this.level, o.x + x, o.y + y)
    }
    for (const d of this.level.dinos.withIn({ x: o.x, y: o.y, w: VIEW, h: VIEW }) as Dino[]) {
      const { x, y } = d.getXY().minus(o)
      if (x < 0 || y < 0 || x >= VIEW || y >= VIEW) continue
      entity[y * VIEW + x] = d.dead ? SKELETON
        : entityId(d === this.level.playerDino, d.dominance, Movement.direction[d.id])
    }
    return [terrain, entity]
  }
}

// bitECS ids and component data are global; clear them so each episode depends only on its seed
function resetEcs() {
  resetGlobals()
  const zero = (v: any) => {
    if (ArrayBuffer.isView(v)) (v as Uint8Array).fill(0)
    else if (v && typeof v === "object") Object.values(v).forEach(zero)
  }
  Object.values(Components).forEach(zero)
}

function terrainClass(level: MainLevel, x: number, y: number): number {
  const t = level.map.at(x, y)
  if (t instanceof Terrain.Ocean) return 0
  if (t instanceof Terrain.Water) return 1
  if (t instanceof Terrain.Lava) return level.map.getTagged(Terrain.Lava).has(t) ? 6 : 7 // burnt-out lava is un-indexed
  if (t instanceof Terrain.Dirt) return 2
  if (t instanceof Terrain.Grass) return 3
  if (t instanceof Terrain.Shrub) return 4
  if (t instanceof Terrain.Jungle) return 5
  return 0
}
