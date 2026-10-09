// Thunder Lizard, imagined: the browser side of tlfp/rollout.py's "fates" mode.
// The net (app/model.onnx) predicts; this file only encodes its input, applies its predictions
// (rolling dice where the game itself is random) and draws the result.

const N = 45, CELLS = N * N, TICK_MS = 70;  // the real game's main loop runs every 70 ms
const NT = 8, NE = 42, NF = 7;
const LAVA = 6;
const MOVES = [[0, 0], [-1, 0], [0, 1], [1, 0], [0, -1]];  // stay, up, right, down, left (dy, dx)
const DIE = 5, GONE = 6;

// the game's own look (game/src/entities/*.ts)
const TERRAIN_LOOK = [
  { chs: "༄༅", fgs: ["#11d", "#11a"], flow: true },   // ocean
  { chs: "≋≈", fgs: ["#44f", "#66e"], flow: true },   // water
  { chs: "¸ʾ˚˓", fgs: ["#754"] },                      // dirt
  { chs: "ɷɞɜ", fgs: ["#2a2"] },                       // grass
  { chs: "ɰɱɵɛ", fgs: ["#172"] },                      // shrub
  { chs: "ƍʕʔʃʆψβϡ", fgs: ["#050"] },                  // jungle
  { chs: "༉", fgs: ["#e11", "#e43"], flow: true },    // lava
  { chs: "༉", fgs: ["#511"] },                         // burnt-out lava
];
const DINO_GLYPH = ["①", "②", "③", "④", "⑤"];
const DINO_COLOR = ["#69A1B5", "#A1663B", "#E2A651", "#C94A24", "#79547D"];
const PLAYER_COLOR = "#eee";

let session, SEEDS, K, DINO_W;
let hist = [];            // last K frames: {t: Uint8Array, e: Uint8Array}
let world = [0, 0];       // accumulated scroll, so each terrain cell keeps its glyph as it scrolls
let held = [];            // arrow keys currently held, most recent last
let running = false, busy = false, dead = false, tickNo = 0;
let timer = null;
const stats = { ticks: [], modelMs: [] };

const $ = id => document.getElementById(id);
const canvas = $("screen"), g = canvas.getContext("2d");

// ---------- loading ----------
(async function init() {
  try {
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
    const [seeds, s] = await Promise.all([
      fetch("seeds.json").then(r => r.json()),
      createSession(),
    ]);
    SEEDS = seeds.seeds; K = seeds.k; DINO_W = seeds.dino_weight; session = s;
    $("seed").innerHTML = SEEDS.map((sd, i) => `<option value="${i}">${i + 1}: ${sd.episode} tick ${sd.tick}</option>`).join("");
    reset();
    $("otitle").textContent = "Ready";
    $("otext").textContent = "Arrow keys to move. The first four frames come from a real game; everything after that is the net's imagination.";
    $("obtns").hidden = false;
  } catch (err) {
    $("otitle").textContent = "Couldn't load the model";
    $("otext").textContent = String(err);
    console.error(err);
  }
})();

// the GPU (WebGPU) if the browser has it, otherwise the CPU (WebAssembly)
let BACKEND = "wasm";
async function createSession() {
  if (navigator.gpu) {
    try { const s = await ort.InferenceSession.create("model.onnx", { executionProviders: ["webgpu"] }); BACKEND = "webgpu"; return s; }
    catch (err) { console.warn("WebGPU unavailable, using CPU", err); }
  }
  return ort.InferenceSession.create("model.onnx", { executionProviders: ["wasm"] });
}

function reset() {
  const sd = SEEDS[+$("seed").value || 0];
  hist = sd.frames.map(([t, e]) => ({ t: Uint8Array.from(t), e: Uint8Array.from(e) }));
  world = [0, 0]; dead = false; tickNo = 0;
  draw(hist[hist.length - 1]);
}

// ---------- input ----------
const KEYMAP = { ArrowUp: 1, ArrowRight: 2, ArrowDown: 3, ArrowLeft: 4 };
addEventListener("keydown", ev => {
  const k = KEYMAP[ev.key];
  if (k) { ev.preventDefault(); held = held.filter(x => x !== k); held.push(k); }
  if (ev.key === " ") { ev.preventDefault(); togglePause(); }
});
addEventListener("keyup", ev => { const k = KEYMAP[ev.key]; if (k) held = held.filter(x => x !== k); });
addEventListener("blur", () => { held = []; });
for (const b of document.querySelectorAll("#pad button")) {
  const k = +b.dataset.k;
  b.addEventListener("pointerdown", ev => { ev.preventDefault(); held = held.filter(x => x !== k); held.push(k); });
  for (const e of ["pointerup", "pointerleave", "pointercancel"]) b.addEventListener(e, () => { held = held.filter(x => x !== k); });
}
const currentKey = () => held.length ? held[held.length - 1] : 0;

// ---------- controls ----------
$("go").onclick = () => { $("overlay").hidden = true; start(); };
$("restart").onclick = () => { reset(); $("overlay").hidden = true; start(); };
$("seed").onchange = () => { reset(); $("overlay").hidden = true; start(); };
$("pause").onclick = togglePause;
function togglePause() {
  if (!session || dead) return;
  if (running) { stop(); $("pause").textContent = "resume"; }
  else { $("overlay").hidden = true; start(); }
}
function start() {
  stop(); running = true; $("pause").textContent = "pause";
  timer = setInterval(() => { if (!busy) tick(); }, TICK_MS);
}
function stop() { running = false; clearInterval(timer); timer = null; }

// ---------- one tick ----------
async function tick() {
  busy = true;
  const key = currentKey();
  const t0 = performance.now();
  const out = await runModel(key);
  const ms = performance.now() - t0;
  const t1 = performance.now();
  const next = applyPredictions(out);
  stats.applyMs = performance.now() - t1;
  hist.push(next); hist.shift(); tickNo++;
  draw(next);
  record(ms);
  busy = false;
  if (!next.e.some(v => v >= 1 && v <= 20)) gameOver();
}

async function runModel(key) {
  const x = new Float32Array((K * 50 + 5) * CELLS);
  for (let f = 0; f < K; f++) {
    const { t, e } = hist[f], base = f * 50 * CELLS;
    for (let i = 0; i < CELLS; i++) {
      x[base + t[i] * CELLS + i] = 1;
      x[base + (NT + e[i]) * CELLS + i] = 1;
    }
  }
  x.fill(1, (K * 50 + key) * CELLS, (K * 50 + key + 1) * CELLS);
  const last = hist[K - 1];
  const lt = new Float32Array(NT * CELLS), le = new Float32Array(NE * CELLS);
  for (let i = 0; i < CELLS; i++) { lt[last.t[i] * CELLS + i] = 1; le[last.e[i] * CELLS + i] = 1; }
  const r = await session.run({
    x: new ort.Tensor("float32", x, [1, K * 50 + 5, N, N]),
    last_t: new ort.Tensor("float32", lt, [1, NT, N, N]),
    last_e: new ort.Tensor("float32", le, [1, NE, N, N]),
  });
  return { terrain: r.terrain.data, entity: r.entity.data, fate: r.fate.data };
}

// scores laid out [class][cell]; helpers per cell
function argmaxAt(a, C, i) { let b = 0, bv = -Infinity; for (let c = 0; c < C; c++) { const v = a[c * CELLS + i]; if (v > bv) { bv = v; b = c; } } return b; }
function sampleAt(a, C, i, adjust) {
  let mx = -Infinity; const p = new Float64Array(C);
  for (let c = 0; c < C; c++) { p[c] = a[c * CELLS + i] + (adjust ? adjust(c) : 0); if (p[c] > mx) mx = p[c]; }
  let s = 0; for (let c = 0; c < C; c++) { p[c] = Math.exp(p[c] - mx); s += p[c]; }
  let r = Math.random() * s;
  for (let c = 0; c < C; c++) { r -= p[c]; if (r <= 0) return c; }
  return C - 1;
}

function nextToLava(t, i) {
  const y = (i / N) | 0, x = i % N;
  return (y > 0 && t[i - N] === LAVA) || (y < N - 1 && t[i + N] === LAVA) || (x > 0 && t[i - 1] === LAVA) || (x < N - 1 && t[i + 1] === LAVA);
}

// the view moved by (dy, dx) if new[y][x] ~= old[y+dy][x+dx]  (tlfp/moves.py find_scroll)
function findScroll(a, b) {
  let best = [0, 0], bestErr = Infinity;
  for (const [dy, dx] of MOVES) {
    let diff = 0, n = 0;
    for (let y = Math.max(-dy, 0); y < N + Math.min(-dy, 0); y++)
      for (let x = Math.max(-dx, 0); x < N + Math.min(-dx, 0); x++) { n++; if (b[y * N + x] !== a[(y + dy) * N + x + dx]) diff++; }
    const err = diff / n;
    if (err < bestErr - 1e-9) { bestErr = err; best = [dy, dx]; }
  }
  return best;
}

function applyPredictions({ terrain, entity, fate }) {
  const last = hist[K - 1];
  // terrain: most likely class, but lava and its neighbours roll dice (spreading and burning out
  // are per-cell dice rolls in the real game)
  const t = new Uint8Array(CELLS);
  for (let i = 0; i < CELLS; i++)
    t[i] = (last.t[i] === LAVA || nextToLava(last.t, i)) ? sampleAt(terrain, NT, i) : argmaxAt(terrain, NT, i);
  const [sy, sx] = findScroll(last.t, t);
  world[0] += sy; world[1] += sx;

  // entities: each one is moved by its predicted fate, so none can duplicate or vanish by accident
  const e = new Uint8Array(CELLS);
  const dice = $("mode").value === "dice";
  for (let i = 0; i < CELLS; i++) {
    let id = last.e[i];
    if (!id) continue;
    const y = (i / N) | 0, x = i % N;
    const isPlayer = id <= 20, isDino = id >= 21 && id <= 40;
    const f = isDino && dice ? sampleAt(fate, NF, i) : argmaxAt(fate, NF, i);
    if (f === GONE) continue;
    let dy = 0, dx = 0;
    if (f === DIE || id === 41) { if (f === DIE) id = 41; }
    else {
      [dy, dx] = MOVES[f];
      if (f > 0) { const base = isPlayer ? 1 : 21; id = base + (((id - base) / 4) | 0) * 4 + (f - 1); }  // face the way it went
    }
    let ty = y + dy - sy, tx = x + dx - sx;
    if (isPlayer && y === 22 && x === 22) { ty = 22; tx = 22; }  // the view follows the player
    if (ty < 0 || ty >= N || tx < 0 || tx >= N || e[ty * N + tx]) {
      ty = y - sy; tx = x - sx;  // blocked or off-screen: stay
      if (ty < 0 || ty >= N || tx < 0 || tx >= N || e[ty * N + tx]) continue;
    }
    e[ty * N + tx] = id;
  }

  // new dinos only in the row/column the scroll just revealed; training weighted dino cells by
  // DINO_W, which multiplied the net's dino odds by that much, so divide it back out
  const adj = c => (c >= 21 && c <= 40 ? -Math.log(DINO_W) : 0);
  const reveal = i => { if (!e[i]) { const c = sampleAt(entity, NE, i, adj); if (c >= 21 && c <= 40) e[i] = c; } };
  if (sy) { const y = sy > 0 ? N - 1 : 0; for (let x = 0; x < N; x++) reveal(y * N + x); }
  if (sx) { const x = sx > 0 ? N - 1 : 0; for (let y = 0; y < N; y++) reveal(y * N + x); }
  return { t, e };
}

// ---------- drawing ----------
const hash = (a, b) => { let h = (a * 374761393 + b * 668265263) | 0; h = (h ^ (h >>> 13)) * 1274126177; return (h ^ (h >>> 16)) >>> 0; };
function draw({ t, e }) {
  const cell = canvas.width / N;
  g.fillStyle = "#111"; g.fillRect(0, 0, canvas.width, canvas.height);
  g.textAlign = "center"; g.textBaseline = "middle";
  g.font = `${cell * 0.95}px sans-serif`;
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const i = y * N + x, look = TERRAIN_LOOK[t[i]];
    // stable glyph per world position; water, ocean and lava shimmer like in the game
    const h = hash(y + world[0], x + world[1]) + (look.flow ? hash(tickNo >> 2, i) : 0);
    g.fillStyle = look.fgs[h % look.fgs.length];
    g.fillText(look.chs[(h >>> 3) % look.chs.length], x * cell + cell / 2, y * cell + cell / 2);
  }
  g.font = `bold ${cell * 0.95}px sans-serif`;
  for (let i = 0; i < CELLS; i++) {
    const id = e[i];
    if (!id) continue;
    const cx = (i % N) * cell + cell / 2, cy = ((i / N) | 0) * cell + cell / 2;
    g.fillStyle = "#111"; g.fillRect(cx - cell / 2, cy - cell / 2, cell, cell);
    if (id === 41) { g.fillStyle = "lightgrey"; g.fillText("%", cx, cy); continue; }
    const level = (((id <= 20 ? id - 1 : id - 21) / 4) | 0);
    g.fillStyle = id <= 20 ? PLAYER_COLOR : DINO_COLOR[level];
    g.fillText(DINO_GLYPH[level], cx, cy);
  }
}

// ---------- meter ----------
function record(ms) {
  const now = performance.now();
  stats.ticks.push(now); stats.modelMs.push(ms);
  while (stats.ticks.length && now - stats.ticks[0] > 2000) { stats.ticks.shift(); stats.modelMs.shift(); }
  const fps = stats.ticks.length > 1 ? (stats.ticks.length - 1) / ((now - stats.ticks[0]) / 1000) : 0;
  const avg = stats.modelMs.reduce((a, b) => a + b, 0) / stats.modelMs.length;
  $("meter").innerHTML = `${fps.toFixed(1)} FPS <span style="color:#888">(game: ${(1000 / TICK_MS).toFixed(1)})</span><br>net ${avg.toFixed(1)} ms/frame (${BACKEND === "webgpu" ? "GPU" : "CPU"})`;
  const p = hist[K - 1].e.find(v => v >= 1 && v <= 20);
  $("status").textContent = `tick ${tickNo}` + (p ? ` · level ${(((p - 1) / 4) | 0) + 1}` : "");
}

function gameOver() {
  stop(); dead = true;
  $("otitle").textContent = "You were eaten (in the net's imagination)";
  $("otext").textContent = `Survived ${tickNo} ticks. Restart or pick another start.`;
  $("obtns").hidden = true;
  $("overlay").hidden = false;
}
