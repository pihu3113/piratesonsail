/* ═══════════════════════════════════════════════════════════════════════
   Treasure Hunt: The Pirate's Maze — Game Logic
   Canvas rendering · Collision · Audio · Speech · Stats tracking
   ═══════════════════════════════════════════════════════════════════════ */

"use strict";

// ── Canvas & context ──────────────────────────────────────────────────
const canvas = document.getElementById("gameCanvas");
const ctx    = canvas.getContext("2d");

// ── Game state ────────────────────────────────────────────────────────
const G = {
  level:        parseInt(document.getElementById("hud").dataset.level  || "1"),
  playerName:   document.getElementById("hud").dataset.name            || "Captain",
  maze:         [],
  rocks:        [],    // [{r,c,alive}]
  playerR:      1,
  playerC:      1,
  goalR:        0,
  goalC:        0,
  rows:         0,
  cols:         0,
  cellSize:     40,
  lives:        parseInt(document.getElementById("hud").dataset.lives  || "3"),
  moves:        0,
  mistakes:     0,     // wall hits + rock hits
  optimalMoves: 0,
  startTime:    null,
  elapsed:      0,
  done:         false,
  invincible:   false,   // brief immunity after rock hit
  animFrame:    null,
};

// ══════════════════════════════════════════════════════════════════════
//  AUDIO ENGINE  (Web Audio API — no external files needed)
// ══════════════════════════════════════════════════════════════════════
let audioCtx    = null;
let masterGain  = null;   // master volume knob
let ambientNode = null;   // ocean background
let ambientGain = null;

/** Boot the AudioContext (must be called from a user-gesture handler). */
function ensureAudio() {
  if (!audioCtx) {
    audioCtx   = new (window.AudioContext || window.webkitAudioContext)();
    masterGain = audioCtx.createGain();
    masterGain.gain.value = 0.85;
    masterGain.connect(audioCtx.destination);
  }
  if (audioCtx.state === "suspended") audioCtx.resume();
}

// ── Low-level helpers ─────────────────────────────────────────────────

/** Create a one-shot oscillator, returns its gain node already connected. */
function _osc(freq, type, vol, startT, stopT) {
  const osc  = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.type             = type;
  osc.frequency.value  = freq;
  gain.gain.value      = vol;
  osc.connect(gain);
  gain.connect(masterGain);
  osc.start(startT);
  osc.stop(stopT);
  return gain;
}

/** ADSR envelope on a gain node. */
function _adsr(gainNode, vol, attackT, decayT, sustainVol, releaseT, now) {
  const g = gainNode.gain;
  g.setValueAtTime(0, now);
  g.linearRampToValueAtTime(vol,        now + attackT);
  g.linearRampToValueAtTime(sustainVol, now + attackT + decayT);
  g.linearRampToValueAtTime(0.0001,     now + attackT + decayT + releaseT);
}

/** White-noise buffer source. dur = seconds. Returns the BufferSource. */
function _noise(dur) {
  const sr  = audioCtx.sampleRate;
  const buf = audioCtx.createBuffer(1, Math.ceil(sr * dur), sr);
  const d   = buf.getChannelData(0);
  for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
  const src = audioCtx.createBufferSource();
  src.buffer = buf;
  return src;
}

// ── Individual sound effects ──────────────────────────────────────────

/**
 * 1. MOVE / WATER SPLASH
 *    Short bandpass-filtered noise burst → sounds like a water drip/splash.
 */
function playMove() {
  ensureAudio();
  const now  = audioCtx.currentTime;
  const src  = _noise(0.12);
  const bp   = audioCtx.createBiquadFilter();
  bp.type    = "bandpass";
  bp.frequency.value = 900;
  bp.Q.value         = 1.8;
  const gain = audioCtx.createGain();
  _adsr(gain, 0.28, 0.005, 0.04, 0.08, 0.07, now);
  src.connect(bp);
  bp.connect(gain);
  gain.connect(masterGain);
  src.start(now);
  src.stop(now + 0.13);
}

/**
 * 2. WALL HIT — dull wooden thud
 *    Low sine + noise burst with fast decay.
 */
function playWallHit() {
  ensureAudio();
  const now = audioCtx.currentTime;

  // Thud — pitched sine
  const g1 = _osc(90, "sine", 0, now, now + 0.22);
  _adsr(g1, 0.35, 0.003, 0.04, 0.1, 0.17, now);

  // Wood texture — noise
  const src = _noise(0.12);
  const lp  = audioCtx.createBiquadFilter();
  lp.type   = "lowpass";
  lp.frequency.value = 400;
  const g2  = audioCtx.createGain();
  _adsr(g2, 0.18, 0.002, 0.03, 0.0, 0.08, now);
  src.connect(lp); lp.connect(g2); g2.connect(masterGain);
  src.start(now); src.stop(now + 0.13);
}

/**
 * 3. ROCK HIT — sharp crack + low boom
 */
function playRockHit() {
  ensureAudio();
  const now = audioCtx.currentTime;

  // Crack (high noise burst)
  const crack = _noise(0.18);
  const hp    = audioCtx.createBiquadFilter();
  hp.type     = "highpass";
  hp.frequency.value = 1800;
  const gCrack = audioCtx.createGain();
  _adsr(gCrack, 0.5, 0.001, 0.02, 0.05, 0.14, now);
  crack.connect(hp); hp.connect(gCrack); gCrack.connect(masterGain);
  crack.start(now); crack.stop(now + 0.19);

  // Low boom
  const g2 = _osc(70, "sine", 0, now, now + 0.28);
  _adsr(g2, 0.45, 0.005, 0.05, 0.08, 0.18, now);

  // Distortion rumble (sawtooth)
  const g3 = _osc(140, "sawtooth", 0, now + 0.02, now + 0.2);
  _adsr(g3, 0.2, 0.004, 0.04, 0.0, 0.12, now + 0.02);
}

/**
 * 4. LIFE LOST — descending minor arpeggio (dramatic!)
 */
function playLifeLost() {
  ensureAudio();
  // A minor descent: A4, F4, D4, A3
  const notes = [440, 349, 294, 220];
  notes.forEach((freq, i) => {
    const t   = audioCtx.currentTime + i * 0.2;
    const g   = _osc(freq, "square", 0, t, t + 0.32);
    _adsr(g, 0.28, 0.01, 0.05, 0.12, 0.18, t);
    // Add sine layer for richness
    const g2  = _osc(freq, "sine", 0, t, t + 0.32);
    _adsr(g2, 0.15, 0.01, 0.05, 0.1, 0.18, t);
  });
}

/**
 * 5. WIN / LEVEL COMPLETE — triumphant ascending fanfare
 */
function playWin() {
  ensureAudio();
  // C major: C5, E5, G5, C6
  const notes = [523, 659, 784, 1047];
  notes.forEach((freq, i) => {
    const t = audioCtx.currentTime + i * 0.15;
    const g = _osc(freq, "sine", 0, t, t + 0.4);
    _adsr(g, 0.32, 0.01, 0.04, 0.18, 0.22, t);
    // Harmonics
    const g2 = _osc(freq * 2, "sine", 0, t, t + 0.4);
    _adsr(g2, 0.08, 0.01, 0.04, 0.04, 0.22, t);
  });
  // Final chord swell
  [523, 659, 784].forEach(freq => {
    const t = audioCtx.currentTime + 0.7;
    const g = _osc(freq, "sine", 0, t, t + 0.6);
    _adsr(g, 0.18, 0.04, 0.1, 0.1, 0.36, t);
  });
}

/**
 * 6. GAME OVER — sorrowful descending melody
 */
function playGameOver() {
  ensureAudio();
  // A minor descending: A4, G4, F4, E4, D4
  const notes = [440, 392, 349, 330, 294];
  notes.forEach((freq, i) => {
    const t = audioCtx.currentTime + i * 0.25;
    const g = _osc(freq, "sine", 0, t, t + 0.45);
    _adsr(g, 0.3, 0.01, 0.06, 0.15, 0.24, t);
    const g2 = _osc(freq / 2, "sine", 0, t, t + 0.45);
    _adsr(g2, 0.12, 0.01, 0.06, 0.06, 0.24, t);
  });
}

// ── 7. OCEAN AMBIENT ──────────────────────────────────────────────────
/**
 * Continuous ocean background:
 *   - Two low sine oscillators slightly detuned → beating "waves"
 *   - Low-pass filtered white noise  → "shhhh" of surf
 *   - Slow LFO on noise gain         → rhythmic swell
 */
function startOceanAmbient() {
  ensureAudio();
  if (ambientNode) return;

  const now = audioCtx.currentTime;

  // Master ambient gain (fades in gently)
  ambientGain = audioCtx.createGain();
  ambientGain.gain.setValueAtTime(0, now);
  ambientGain.gain.linearRampToValueAtTime(0.55, now + 3.0);
  ambientGain.connect(masterGain);

  // ── Wave tones (two detuned sines = beating effect) ──
  [58, 61.5].forEach(freq => {
    const osc  = audioCtx.createOscillator();
    osc.type   = "sine";
    osc.frequency.value = freq;
    const g    = audioCtx.createGain();
    g.gain.value = 0.18;
    osc.connect(g);
    g.connect(ambientGain);
    osc.start(now);
  });

  // ── Surf noise (looping, low-pass filtered) ──
  const sr     = audioCtx.sampleRate;
  const bufLen = sr * 4;          // 4-second noise loop
  const buf    = audioCtx.createBuffer(1, bufLen, sr);
  const d      = buf.getChannelData(0);
  for (let i = 0; i < bufLen; i++) d[i] = Math.random() * 2 - 1;

  const src    = audioCtx.createBufferSource();
  src.buffer   = buf;
  src.loop     = true;

  const lp     = audioCtx.createBiquadFilter();
  lp.type      = "lowpass";
  lp.frequency.value = 320;
  lp.Q.value   = 0.7;

  const noiseGain = audioCtx.createGain();
  noiseGain.gain.value = 0.22;
  src.connect(lp);
  lp.connect(noiseGain);
  noiseGain.connect(ambientGain);
  src.start(now);
  ambientNode = src;   // keep reference so we don't start twice

  // ── LFO: slow volume swell → rhythmic wave feel ──
  const lfo       = audioCtx.createOscillator();
  lfo.type        = "sine";
  lfo.frequency.value = 0.18;   // ~one swell every 5.5 s
  const lfoGain   = audioCtx.createGain();
  lfoGain.gain.value = 0.12;
  lfo.connect(lfoGain);
  lfoGain.connect(noiseGain.gain);
  lfo.start(now);
}

// ── 8. SPEECH SYNTHESIS ───────────────────────────────────────────────
/**
 * Speak with a slightly low, slow pirate voice using the best
 * available browser voice.
 */
function speak(text) {
  if (!window.speechSynthesis) return;

  const sayIt = () => {
    const u   = new SpeechSynthesisUtterance(text);
    u.rate    = 0.82;
    u.pitch   = 0.78;
    u.volume  = 1.0;

    // Pick an English male voice if available
    const voices = speechSynthesis.getVoices();
    const male   = voices.find(v =>
      v.lang.startsWith("en") && /male|guy|david|mark|alex|fred/i.test(v.name)
    );
    const eng    = voices.find(v => v.lang.startsWith("en"));
    if (male) u.voice = male;
    else if (eng) u.voice = eng;

    speechSynthesis.cancel();
    speechSynthesis.speak(u);
  };

  // Voices may not be loaded yet on first call
  if (speechSynthesis.getVoices().length > 0) {
    sayIt();
  } else {
    speechSynthesis.addEventListener("voiceschanged", sayIt, { once: true });
  }
}

// ── Cell size calculation ─────────────────────────────────────────────
function calcCellSize(rows, cols) {
  const maxW = Math.min(window.innerWidth  * 0.88, 680);
  const maxH = Math.min(window.innerHeight * 0.60, 600);
  return Math.max(18, Math.floor(Math.min(maxW / cols, maxH / rows)));
}

// ── Drawing helpers ───────────────────────────────────────────────────

/** Cross-browser rounded rectangle (fallback if roundRect unavailable) */
function fillRoundRect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.arcTo(x + w, y,     x + w, y + r,     r);
  ctx.lineTo(x + w, y + h - r);
  ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
  ctx.lineTo(x + r, y + h);
  ctx.arcTo(x,     y + h, x,     y + h - r, r);
  ctx.lineTo(x,     y + r);
  ctx.arcTo(x,     y,     x + r, y,         r);
  ctx.closePath();
  ctx.fill();
}

/** Draw a centered label inside a cell */
function cellLabel(text, cx, cy, size, color) {
  ctx.fillStyle    = color;
  ctx.font         = `bold ${size}px sans-serif`;
  ctx.textAlign    = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, cx, cy);
}

// ── Drawing ───────────────────────────────────────────────────────────
const COLORS = {
  // Walls are rocky stone
  rock1:     "#6b7280",   // main rock face (gray)
  rock2:     "#4b5563",   // darker crevice
  rock3:     "#9ca3af",   // lighter highlight
  rock4:     "#374151",   // deep shadow edge
  // Open path = ocean
  path:      "#1e40af",
  pathLight: "#2563eb",
  // Entities
  player:    "#dc2626",
  goal:      "#16a34a",
  octopus:   "#7c3aed",
};

function drawCell(r, c) {
  const x = c * G.cellSize, y = r * G.cellSize, s = G.cellSize;

  if (G.maze[r][c] === 0) {
    // ── Wall rendered as a rock / stone block ────────────────────────
    // Base rock color
    ctx.fillStyle = COLORS.rock1;
    ctx.fillRect(x, y, s, s);

    // Dark shadow edges (bottom-right) — gives 3-D depth
    ctx.fillStyle = COLORS.rock4;
    ctx.fillRect(x,         y + s - 3, s,     3);   // bottom strip
    ctx.fillRect(x + s - 3, y,         3,     s);   // right strip

    // Light highlight edge (top-left)
    ctx.fillStyle = COLORS.rock3;
    ctx.fillRect(x, y, s, 2);       // top strip
    ctx.fillRect(x, y, 2, s);       // left strip

    // Inner rock face — slightly darker
    ctx.fillStyle = COLORS.rock2;
    ctx.fillRect(x + 3, y + 3, s - 6, s - 6);

    // Random crack lines for texture (seeded by position so they don't flicker)
    const seed = (r * 31 + c * 17) % 7;
    ctx.strokeStyle = COLORS.rock4;
    ctx.lineWidth   = 1;
    if (seed < 3 && s >= 20) {
      ctx.beginPath();
      ctx.moveTo(x + 5,     y + Math.floor(s * 0.3));
      ctx.lineTo(x + s - 6, y + Math.floor(s * 0.55));
      ctx.stroke();
    }
    if (seed >= 2 && seed < 5 && s >= 20) {
      ctx.beginPath();
      ctx.moveTo(x + Math.floor(s * 0.4), y + 4);
      ctx.lineTo(x + Math.floor(s * 0.6), y + s - 5);
      ctx.stroke();
    }

  } else {
    // ── Open path — bright blue ocean ────────────────────────────────
    ctx.fillStyle = COLORS.path;
    ctx.fillRect(x, y, s, s);
    // Subtle shimmer stripe
    if (s >= 24 && (r * 3 + c * 7) % 9 === 0) {
      ctx.fillStyle = "rgba(96,165,250,0.22)";
      ctx.fillRect(x + 3, y + Math.floor(s * 0.35), Math.floor(s * 0.45), 2);
    }
  }
}

function drawMaze() {
  for (let r = 0; r < G.rows; r++)
    for (let c = 0; c < G.cols; c++)
      drawCell(r, c);
}

function drawRocks() {
  // Rocks are now the maze walls themselves — no in-path obstacles
}

function drawGoal() {
  const s   = G.cellSize;
  const pad = Math.max(3, Math.floor(s * 0.12));
  const r2  = Math.max(3, Math.floor(s * 0.2));
  const x   = G.goalC * s, y = G.goalR * s;

  // Bright GREEN block
  ctx.fillStyle = COLORS.goal;
  fillRoundRect(x + pad, y + pad, s - pad * 2, s - pad * 2, r2);

  // White border glow
  ctx.strokeStyle = "rgba(255,255,255,0.55)";
  ctx.lineWidth   = 2;
  ctx.stroke();

  // "G" label
  if (s >= 22) cellLabel("G", x + s / 2, y + s / 2, Math.floor(s * 0.42), "#fff");

  // Purple octopus block on level 3 (adjacent open cell to goal)
  if (G.level === 3) {
    const candidates = [
      [G.goalR - 2, G.goalC],
      [G.goalR,     G.goalC - 2],
      [G.goalR - 1, G.goalC - 1],
    ];
    for (const [rr, cc] of candidates) {
      if (rr >= 0 && cc >= 0 && rr < G.rows && cc < G.cols
          && G.maze[rr][cc] === 1
          && !(rr === G.playerR && cc === G.playerC)) {
        const ox = cc * s, oy = rr * s;
        ctx.fillStyle = COLORS.octopus;
        fillRoundRect(ox + pad, oy + pad, s - pad * 2, s - pad * 2, r2);
        if (s >= 22) cellLabel("👾", ox + s / 2, oy + s / 2, Math.floor(s * 0.55), "#fff");
        break;
      }
    }
  }
}

// ── Player: solid RED block ───────────────────────────────────────────
let blinkTick = 0;
function drawPlayer() {
  const s   = G.cellSize;
  const pad = Math.max(3, Math.floor(s * 0.1));
  const r2  = Math.max(3, Math.floor(s * 0.2));
  const x   = G.playerC * s, y = G.playerR * s;

  // Flicker when invincible
  blinkTick++;
  if (G.invincible && blinkTick % 6 < 3) return;

  // Red block
  ctx.fillStyle = COLORS.player;
  fillRoundRect(x + pad, y + pad, s - pad * 2, s - pad * 2, r2);

  // White border
  ctx.strokeStyle = "rgba(255,255,255,0.7)";
  ctx.lineWidth   = 2;
  ctx.stroke();

  // "P" label
  if (s >= 22) cellLabel("P", x + s / 2, y + s / 2, Math.floor(s * 0.42), "#fff");
}

function render() {
  drawMaze();
  drawGoal();
  drawRocks();
  drawPlayer();
}

// ── HUD update ────────────────────────────────────────────────────────
const heartsEl  = document.getElementById("hearts");
const timerEl   = document.getElementById("timer");
const movesEl   = document.getElementById("moves-count");

function updateHUD() {
  // Hearts
  heartsEl.textContent = "❤️".repeat(G.lives) + "🖤".repeat(Math.max(0, 3 - G.lives));
  // Timer
  timerEl.textContent  = formatTime(G.elapsed);
  // Moves
  movesEl.textContent  = G.moves;
}

function formatTime(sec) {
  const m = Math.floor(sec / 60).toString().padStart(2, "0");
  const s = Math.floor(sec % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}

// ── Notification toast ────────────────────────────────────────────────
const notifEl = document.getElementById("notif");
let notifTimer = null;

function showNotif(msg, ms = 1800) {
  notifEl.textContent = msg;
  notifEl.classList.add("show");
  clearTimeout(notifTimer);
  notifTimer = setTimeout(() => notifEl.classList.remove("show"), ms);
}

// ── Shared: take damage (wall or boundary hit) ────────────────────────
function takeDamage(msg) {
  if (G.invincible) return;

  G.lives--;
  G.mistakes++;
  playWallHit();
  playLifeLost();
  showNotif(msg);
  updateHUD();

  if (G.lives <= 0) {
    playGameOver();
    setTimeout(triggerGameOver, 600);
    return;
  }

  // Brief invincibility so one crash doesn't drain all lives instantly
  G.invincible = true;
  setTimeout(() => { G.invincible = false; }, 900);
}

// ── Movement & collision ──────────────────────────────────────────────
function tryMove(dr, dc) {
  if (G.done) return;

  const nr = G.playerR + dr;
  const nc = G.playerC + dc;

  // Out-of-bounds crash
  if (nr < 0 || nc < 0 || nr >= G.rows || nc >= G.cols) {
    takeDamage("💥 Crashed into the edge! −1 life");
    return;
  }

  // Rock-wall crash — lose a life, stay in place
  if (G.maze[nr][nc] === 0) {
    takeDamage("🪨 Crashed into a rock wall! −1 life");
    return;
  }

  // Valid open-water move
  G.playerR = nr;
  G.playerC = nc;
  G.moves++;
  playMove();

  // Goal check
  if (nr === G.goalR && nc === G.goalC) {
    playWin();
    G.done = true;
    speak("Yo Ho Captain! You found the treasure!");
    setTimeout(triggerLevelComplete, 1200);
  }

  updateHUD();
}

// ── Input ─────────────────────────────────────────────────────────────
document.addEventListener("keydown", e => {
  const map = {
    ArrowUp:    [-1,  0],
    ArrowDown:  [ 1,  0],
    ArrowLeft:  [ 0, -1],
    ArrowRight: [ 0,  1],
    w: [-1, 0], s: [1, 0], a: [0, -1], d: [0, 1],
  };
  if (map[e.key]) {
    e.preventDefault();
    tryMove(...map[e.key]);
  }
});

// Touch / swipe
let touchStartX = 0, touchStartY = 0;
canvas.addEventListener("touchstart", e => {
  touchStartX = e.touches[0].clientX;
  touchStartY = e.touches[0].clientY;
}, { passive: true });
canvas.addEventListener("touchend", e => {
  const dx = e.changedTouches[0].clientX - touchStartX;
  const dy = e.changedTouches[0].clientY - touchStartY;
  if (Math.abs(dx) < 10 && Math.abs(dy) < 10) return;
  if (Math.abs(dx) > Math.abs(dy)) tryMove(0, dx > 0 ? 1 : -1);
  else                              tryMove(dy > 0 ? 1 : -1, 0);
}, { passive: true });

// D-pad buttons (mobile)
["btn-up","btn-down","btn-left","btn-right"].forEach(id => {
  const el = document.getElementById(id);
  if (!el) return;
  const map = { "btn-up":[-1,0], "btn-down":[1,0], "btn-left":[0,-1], "btn-right":[0,1] };
  el.addEventListener("click", () => tryMove(...map[id]));
});

// ── Game loop ─────────────────────────────────────────────────────────
let lastTs = null;
function loop(ts) {
  if (!G.done) {
    if (lastTs !== null) G.elapsed += (ts - lastTs) / 1000;
    lastTs = ts;
    updateHUD();
  }
  render();
  G.animFrame = requestAnimationFrame(loop);
}

// ── Level complete / Game over ────────────────────────────────────────
async function triggerLevelComplete() {
  cancelAnimationFrame(G.animFrame);
  const timeTaken = Math.round(G.elapsed);

  try {
    await fetch("/api/level_complete", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        time_taken:    timeTaken,
        mistakes:      G.mistakes,
        moves:         G.moves,
        optimal_moves: G.optimalMoves,
        lives:         G.lives,
      }),
    });
  } catch (_) {}

  // Check if all 3 levels done
  if (G.level >= 3) {
    window.location.href = "/win";
  } else {
    window.location.href = "/level_complete";
  }
}

function triggerGameOver() {
  cancelAnimationFrame(G.animFrame);
  document.getElementById("overlay-gameover").classList.add("active");
  speak("Oh no! The ocean has claimed you, Captain!");
}

document.getElementById("btn-restart")?.addEventListener("click", () => {
  window.location.href = "/game";
});
document.getElementById("btn-menu")?.addEventListener("click", () => {
  window.location.href = "/";
});

// ── Mute toggle ───────────────────────────────────────────────────────
let muted = false;
document.getElementById("btn-mute")?.addEventListener("click", () => {
  ensureAudio();
  muted = !muted;
  masterGain.gain.setTargetAtTime(muted ? 0 : 0.85, audioCtx.currentTime, 0.1);
  document.getElementById("btn-mute").textContent = muted ? "🔇" : "🔊";
});

// ── Initialise ────────────────────────────────────────────────────────
async function init() {
  // Voice intro
  speak("Loading game… Yo Ho Captain!");

  let data;
  try {
    const res = await fetch("/api/get_maze", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ level: G.level }),
    });
    data = await res.json();
  } catch (err) {
    console.error("Failed to load maze", err);
    return;
  }

  G.maze         = data.grid;
  G.rows         = data.rows;
  G.cols         = data.cols;
  G.playerR      = data.start[0];
  G.playerC      = data.start[1];
  G.goalR        = data.goal[0];
  G.goalC        = data.goal[1];
  G.optimalMoves = data.optimal_moves;
  G.lives        = data.lives ?? G.lives;
  G.rocks        = (data.rocks || []).map(([r, c]) => ({ r, c, alive: true }));

  // Canvas sizing
  G.cellSize = calcCellSize(G.rows, G.cols);
  canvas.width  = G.cols * G.cellSize;
  canvas.height = G.rows * G.cellSize;

  // Ocean ambient starts on the very first user interaction
  const startAmbient = () => { startOceanAmbient(); };
  document.addEventListener("keydown",  startAmbient, { once: true });
  document.addEventListener("pointerdown", startAmbient, { once: true });

  G.startTime = performance.now();
  updateHUD();
  requestAnimationFrame(loop);
}

init();
