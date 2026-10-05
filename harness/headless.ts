// Runs Thunder Lizard with no browser and steps it one main-loop tick at a time.
// Each tick it records the state grid around the player and the key held during it.

import * as ROT from "../game/lib/rotjs"
import MainLevel from "../game/src/level"
import awarenessSystem from "../game/src/systems/awareness"
import movementSystem, { keypressCb } from "../game/src/systems/movement"
import deplacementSystem from "../game/src/systems/displacement"
import scoreSystem from "../game/src/systems/score"
import { MAP_SIZE } from "../game/src/constants"
import type Dino from "../game/src/entities/dino"
import { removeEntity, deleteWorld, resetGlobals } from "bitecs"
import * as Components from "../game/src/components"

export { VIEW, TERRAIN, KEYS, ENTITY_CLASSES, entityId, SKELETON } from "../game/src/recorder"
export type { Key } from "../game/src/recorder"
import { Key, observe, viewOrigin } from "../game/src/recorder"

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

  viewOrigin() { return viewOrigin(this.level) }
  observe() { return observe(this.level) }
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
