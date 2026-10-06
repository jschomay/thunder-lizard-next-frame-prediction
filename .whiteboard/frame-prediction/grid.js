// Draws an encoded 45x45 state grid (terrain digits + base-36 entity string) on a canvas.
const TERRAIN_NAMES = ["OCEAN", "WATER", "DIRT", "GRASS", "SHRUB", "JUNGLE", "LAVA", "BURNT"];
const TERRAIN_COLORS = ["#16206e", "#4a55e6", "#7a5a40", "#2aa52a", "#1a7a2a", "#0b4d0b", "#e3341f", "#3a2a22"];
const B36 = "0123456789abcdefghijklmnopqrstuvwxyz!@#$%^";
const DIR_ARROWS = ["▲", "▶", "▼", "◀"];

function decodeEntity(v) {
  if (v === 0) return null;
  if (v === 41) return { kind: "skeleton" };
  const player = v <= 20;
  const i = (player ? v - 1 : v - 21);
  return { kind: player ? "player" : "dino", level: Math.floor(i / 4) + 1, dir: i % 4 };
}

function drawGrid(canvas, terrain, entity, opts = {}) {
  const N = 45, cell = opts.cell || 8;
  canvas.width = N * cell; canvas.height = N * cell;
  const ctx = canvas.getContext("2d");
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.font = `bold ${cell + 2}px system-ui`;
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const i = y * N + x;
    ctx.fillStyle = TERRAIN_COLORS[+terrain[i]];
    ctx.fillRect(x * cell, y * cell, cell, cell);
  }
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const e = decodeEntity(B36.indexOf(entity[y * N + x]));
    if (!e) continue;
    const cx = x * cell + cell / 2, cy = y * cell + cell / 2;
    if (e.kind === "skeleton") { ctx.fillStyle = "#ddd"; ctx.fillText("%", cx, cy); continue; }
    ctx.fillStyle = e.kind === "player" ? "#ffe14d" : ["#f0f0f0", "#ffb0b0", "#ff7a7a", "#ff4040", "#c000ff"][e.level - 1];
    ctx.beginPath(); ctx.arc(cx, cy, cell * 0.85, 0, 7); ctx.fill();
    ctx.fillStyle = "#000"; ctx.font = `bold ${cell}px system-ui`;
    ctx.fillText(DIR_ARROWS[e.dir], cx, cy + 0.5);
    ctx.font = `bold ${cell + 2}px system-ui`;
  }
  if (opts.highlight) {  // cells that differ from a previous frame
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 1;
    for (const i of opts.highlight) ctx.strokeRect((i % N) * cell + 0.5, Math.floor(i / N) * cell + 0.5, cell - 1, cell - 1);
  }
}

function changedCells(a, b) {
  const out = [];
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) out.push(i);
  return out;
}

function legendHTML() {
  return TERRAIN_NAMES.map((n, i) => `<span class="sw"><i style="background:${TERRAIN_COLORS[i]}"></i>${i} ${n}</span>`).join("") +
    `<span class="sw"><i style="background:#ffe14d;border-radius:50%"></i>player (arrow = facing)</span>` +
    `<span class="sw"><i style="background:#ff4040;border-radius:50%"></i>dino (redder = higher level)</span>` +
    `<span class="sw"><b style="width:12px;display:inline-block;text-align:center">%</b>skeleton</span>`;
}
