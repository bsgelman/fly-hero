// Page wiring: modes, game canvas, charts, live simulation loop.
import { Network, wiring } from './sim.js';
import { Episode, SONGS, SLOT_MS, LEAD_MS, WAIT, hitTime, songMs } from './game.js';
import { BioRL, DeepRL, BIO_DEFAULTS } from './agents.js';
import { Brain } from './brain.js';

const $ = (id) => document.getElementById(id);
const [sub, pointsBuf] = await Promise.all([
  fetch('data/subgraph.json').then(r => r.json()),
  fetch('data/brain_points.bin').then(r => r.arrayBuffer()),
]);
const results = await fetch('data/results.json').then(r => (r.ok ? r.json() : null)).catch(() => null);
const brain = new Brain($('brainGl'), $('brainOverlay'), $('brainHeader'), new Uint16Array(pointsBuf), sub);

const LANE_COL = ['#e76f51', '#e9c46a', '#2a9d8f'], FALL_MS = 1600, HUMAN_WINDOW = 150;
const S = { mode: 'watch', speed: 1, agentKind: 'bio', wiring: 'real', blocked: false, sound: true, curve: [], epNo: 0 };

// ---------- fly ----------
function buildFly() {
  const seed = 1 + Math.floor(Math.random() * 1e6);
  S.net = new Network(sub, wiring(sub, S.wiring, seed), { seed, kcMbonGain: BIO_DEFAULTS.kcMbonGain });
  S.agent = S.agentKind === 'bio' ? new BioRL(S.net, { seed, blocked: S.blocked }) : new DeepRL(S.net, { seed });
  S.curve = [];
  S.epNo = 0;
  newEpisode(true);
  $('agentNote').textContent = S.agentKind === 'bio'
    ? `Agent A: learning inside the connectome (dopamine-gated KC→MBON plasticity)${S.blocked ? ', dopamine BLOCKED' : ''} · ${S.wiring} wiring · simulated`
    : `Agent B: frozen connectome, REINFORCE readout from ${S.net.idx.dn.length} descending neurons · ${S.wiring} wiring · simulated`;
}

function newEpisode(count) {
  if (count) S.epNo++;
  S.ep = new Episode(S.net, S.agent, SONGS.train);
  S.evSeen = 0;
  S.flyPress = [-1e9, -1e9, -1e9];
}

function onFlyEvents(ep, audible) {
  while (S.evSeen < ep.events.length) {
    const ev = ep.events[S.evSeen++];
    if (ev.action !== WAIT) S.flyPress[ev.action] = ev.t;
    if (audible) { click(); if (ev.note >= 0 && ev.action === ev.note) tone(ev.note); }
    const a = S.agent, what = ev.note < 0 ? 'rest' : `lane ${ev.note + 1}`;
    const did = ev.action === WAIT ? 'waited' : `pressed lane ${ev.action + 1}`;
    const verdict = ev.note < 0 ? (ev.action === WAIT ? 'ok' : 'false press') : ev.action === ev.note ? 'HIT' : ev.action === WAIT ? 'miss' : 'wrong lane';
    const learning = a.kind === 'bio'
      ? `RPE δ = ${a.delta >= 0 ? '+' : ''}${a.delta.toFixed(2)} → ${S.blocked ? 'dopamine blocked' : a.delta > 0 ? 'PAM (reward) burst' : a.delta < 0 ? 'PPL1 (punish) burst' : 'no dopamine'}` +
        (!ep.learn ? ' · learning frozen, no weight change' : ev.action === WAIT ? ' · no press, no eligible synapses' : ` · ${S.net.plastic[ev.action].length.toLocaleString()} KC→MBON connections eligible`)
      : ep.learn ? 'REINFORCE update on readout weights' : 'readout frozen';
    $('lastEvent').textContent = `slot ${ev.slot + 1}: ${what} → ${did} → ${verdict} (${ev.reward >= 0 ? '+' : ''}${ev.reward}) · ${learning}`;
  }
}

function advanceWatch(dt) {
  const deadline = performance.now() + 14;
  let steps = S.speed === 'turbo' ? Infinity : Math.round(dt * S.speed);
  while (steps-- > 0) {
    S.ep.step();
    onFlyEvents(S.ep, S.speed === 1);
    if (S.ep.done) {
      S.curve.push(S.ep.rate);
      newEpisode(true);
    }
    if (performance.now() > deadline) break; // ponytail: slow machines just run slower than the requested speed
  }
}

// ---------- human ----------
function startHuman() {
  S.human = { t0: performance.now() + 2000, res: [], hits: 0, extra: 0, press: [-1e9, -1e9, -1e9], done: false };
  S.shadow = new Episode(S.net, S.agent, SONGS.train, { learn: false });
  S.evSeen = 0;
  S.flyPress = [-1e9, -1e9, -1e9];
  $('humanMsg').textContent = 'get ready…';
}

function advanceHuman(now) {
  const h = S.human;
  if (!h || h.done) return;
  const t = now - h.t0;
  while (S.shadow.t < t && !S.shadow.done) { S.shadow.step(); onFlyEvents(S.shadow, false); }
  const notes = SONGS.train.notes;
  for (let s = 0; s < notes.length; s++) if (notes[s] >= 0 && !h.res[s] && t > hitTime(s) + HUMAN_WINDOW) h.res[s] = 'miss';
  if (t > songMs(SONGS.train) + 300) {
    h.done = true;
    const n = notes.filter(x => x >= 0).length, you = (100 * h.hits) / n, fly = 100 * S.shadow.rate;
    const ref = results?.conditions?.['A-real']?.summary;
    $('humanMsg').innerHTML = `You: <b>${you.toFixed(0)}%</b> (${h.hits}/${n}, ${h.extra} extra/wrong presses) · ` +
      `Fly: <b>${fly.toFixed(0)}%</b> (${S.shadow.wrong} wrong, ${S.shadow.falsePresses} false presses; frozen, trained ${S.curve.length} episodes this session)` +
      (ref ? `<br><span class="small">reference: Agent A on real wiring averaged ${(100 * ref.last5).toFixed(0)}% over episodes 26–30 in the headless runs</span>` : '');
  }
}

addEventListener('keydown', (e) => {
  const lane = { j: 0, k: 1, l: 2, 1: 0, 2: 1, 3: 2 }[e.key.toLowerCase()];
  const h = S.human;
  if (S.mode !== 'human' || lane === undefined || !h || h.done) return;
  const t = performance.now() - h.t0, notes = SONGS.train.notes;
  h.press[lane] = t;
  let best = -1;
  for (let s = 0; s < notes.length; s++) {
    if (notes[s] !== lane || h.res[s] || Math.abs(hitTime(s) - t) > HUMAN_WINDOW) continue;
    if (best < 0 || Math.abs(hitTime(s) - t) < Math.abs(hitTime(best) - t)) best = s;
  }
  if (best >= 0) { h.res[best] = 'hit'; h.hits++; tone(lane); } else h.extra++;
});

// ---------- drawing ----------
function drawGame(t, resultOf, humanPress) {
  const c = $('game'), g = c.getContext('2d'), W = c.width, H = c.height, lw = W / 3, hitY = H - 70;
  g.fillStyle = '#131418';
  g.fillRect(0, 0, W, H);
  for (let l = 0; l < 3; l++) { g.fillStyle = l === 1 ? '#191a1f' : '#16171b'; g.fillRect(l * lw, 0, lw, H); }
  g.fillStyle = '#777';
  g.fillRect(0, hitY - 1, W, 2);
  const notes = SONGS.train.notes;
  for (let s = 0; s < notes.length; s++) {
    const n = notes[s], d = hitTime(s) - t;
    if (n < 0 || d > FALL_MS || d < -400) continue;
    const res = resultOf(s), x = n * lw + lw / 2;
    if (res === 'hit') {
      const a = Math.max(0, 1 + d / 400);
      g.strokeStyle = `rgba(255,255,255,${a})`;
      g.lineWidth = 3;
      g.beginPath(); g.arc(x, hitY, 20 + (1 - a) * 25, 0, 2 * Math.PI); g.stroke();
      continue;
    }
    const y = hitY - (d / FALL_MS) * hitY;
    g.fillStyle = res ? '#444' : LANE_COL[n];
    g.fillRect(x - 38, y - 9, 76, 18);
  }
  for (let l = 0; l < 3; l++) {
    const lit = humanPress && t - humanPress[l] < 120;
    g.fillStyle = lit ? LANE_COL[l] : '#222';
    g.fillRect(l * lw + 10, H - 50, lw - 20, 36);
    g.fillStyle = lit ? '#111' : '#888';
    g.font = '14px ui-monospace, monospace';
    g.fillText('JKL'[l], l * lw + lw / 2 - 4, H - 27);
    if (t - S.flyPress[l] < 160) { // the fly's press
      g.fillStyle = '#fff';
      g.beginPath(); g.moveTo(l * lw + lw / 2, hitY + 6); g.lineTo(l * lw + lw / 2 - 9, hitY + 18); g.lineTo(l * lw + lw / 2 + 9, hitY + 18); g.fill();
    }
  }
}

function drawCurve() {
  const c = $('curve'), g = c.getContext('2d'), W = c.width, H = c.height, L = 30, B = 18;
  g.fillStyle = '#0e0f12'; g.fillRect(0, 0, W, H);
  g.strokeStyle = '#333'; g.strokeRect(L, 4, W - L - 4, H - B - 4);
  g.fillStyle = '#777'; g.font = '10px ui-monospace, monospace';
  g.fillText('100%', 0, 12); g.fillText('0%', 10, H - B); g.fillText('hit % per episode (this fly, simulated)', L + 6, H - 4);
  const y = (v) => 4 + (1 - v) * (H - B - 8);
  g.setLineDash([4, 4]); g.strokeStyle = '#555';
  g.beginPath(); g.moveTo(L, y(0.8)); g.lineTo(W - 4, y(0.8)); g.stroke(); g.setLineDash([]);
  const n = Math.max(30, S.curve.length), x = (e) => L + (e / Math.max(1, n - 1)) * (W - L - 8);
  g.strokeStyle = '#ffb347'; g.lineWidth = 2; g.beginPath();
  S.curve.forEach((v, e) => (e ? g.lineTo(x(e), y(v)) : g.moveTo(x(e), y(v))));
  g.stroke(); g.lineWidth = 1;
}

const CMP = {
  'A-real': ['#ffb347', []], 'A-shuffled': ['#ff6f3c', [6, 3]], 'A-random': ['#a0523b', [2, 3]], 'A-dopamine-blocked': ['#999', [4, 4]],
  'B-real': ['#5fb3ff', []], 'B-shuffled': ['#3f7fd0', [6, 3]], 'B-random': ['#35608f', [2, 3]],
};

function drawCompare() {
  const c = $('cmp'), g = c.getContext('2d'), W = c.width, H = c.height, L = 34, B = 22, T = 6;
  g.fillStyle = '#0e0f12'; g.fillRect(0, 0, W, H);
  if (!results) { g.fillStyle = '#aaa'; g.fillText('no data/results.json yet: run node tools/run_experiments.mjs', 10, 30); return; }
  const E = results.meta.episodes, x = (e) => L + (e / E) * (W - L - 6), y = (v) => T + (1 - v) * (H - T - B);
  g.strokeStyle = '#333'; g.strokeRect(L, T, W - L - 6, H - T - B);
  g.fillStyle = '#777'; g.font = '10px ui-monospace, monospace';
  for (const v of [0, 0.5, 1]) g.fillText(`${v * 100}%`, 0, y(v) + 4);
  g.fillText('episode (0 = frozen, before learning)', L + 4, H - 6);
  let rows = '';
  for (const [name, [color, dash]] of Object.entries(CMP)) {
    const cond = results.conditions[name];
    if (!cond) continue;
    const series = cond.runs.map(r => [r.before, ...r.curve]);
    const mean = series[0].map((_, e) => series.reduce((s, r) => s + r[e], 0) / series.length);
    const sd = mean.map((m, e) => Math.sqrt(series.reduce((s, r) => s + (r[e] - m) ** 2, 0) / series.length));
    g.fillStyle = color + '22';
    g.beginPath();
    mean.forEach((m, e) => g.lineTo(x(e), y(Math.min(1, m + sd[e]))));
    for (let e = mean.length - 1; e >= 0; e--) g.lineTo(x(e), y(Math.max(0, mean[e] - sd[e])));
    g.fill();
    g.strokeStyle = color; g.setLineDash(dash); g.lineWidth = 2; g.beginPath();
    mean.forEach((m, e) => (e ? g.lineTo(x(e), y(m)) : g.moveTo(x(e), y(m))));
    g.stroke(); g.setLineDash([]); g.lineWidth = 1;
  }
  for (const [name, cond] of Object.entries(results.conditions)) {
    const s = cond.summary, p = (v) => (100 * v).toFixed(0);
    const sw = CMP[name] ? `<i class="sw" style="background:${CMP[name][0]}"></i>` : '<i class="sw"></i>';
    rows += `<tr><td>${sw}${name}</td><td>${p(s.before)}</td><td>${p(s.ep1)}</td><td>${p(s.last5)}±${p(s.last5sd)}</td><td>${s.reached80}/${s.to80.length}</td><td>${p(s.test)}±${p(s.testsd)}</td></tr>`;
  }
  $('cmpTable').innerHTML = `<table><tr><th>condition</th><th>before</th><th>ep 1</th><th>last 5</th><th>seeds ≥80%</th><th>unseen song</th></tr>${rows}</table>
    <p class="small">hit % of notes, mean ± sd over seeds · "seeds ≥80%" = seeds whose 3-episode moving average reached 80% · B-real-150ep: Deep-RL given 150 episodes (3 seeds)</p>`;
}

function updateStory(ep) {
  const local = ep.t - LEAD_MS, slot = Math.floor(local / SLOT_MS), ph = local - slot * SLOT_MS;
  const note = local >= 0 ? (SONGS.train.notes[slot] ?? -1) : -1;
  const on = note < 0 ? '' : ph < 100 ? 'eyes' : ph < 200 ? 'memory' : ph < 240 ? 'output' : ph < 300 ? 'result' : ph < 340 ? 'dopamine' : 'update';
  for (const el of $('story').children) el.classList.toggle('on', el.dataset.s === on);
}

// ---------- audio ----------
let ac = null;
function beep(freq, dur, gain, type = 'triangle') {
  if (!S.sound) return;
  ac ??= new AudioContext();
  const o = ac.createOscillator(), gn = ac.createGain();
  o.type = type; o.frequency.value = freq;
  gn.gain.setValueAtTime(gain, ac.currentTime);
  gn.gain.exponentialRampToValueAtTime(0.001, ac.currentTime + dur);
  o.connect(gn).connect(ac.destination);
  o.start(); o.stop(ac.currentTime + dur);
}
const tone = (lane) => beep([392, 494, 587][lane], 0.18, 0.08);
const click = () => beep(1400, 0.02, 0.02, 'square');
addEventListener('pointerdown', () => ac?.state === 'suspended' && ac.resume());

// ---------- controls ----------
function setMode(mode) {
  S.mode = mode;
  document.querySelectorAll('[data-mode]').forEach(b => b.classList.toggle('on', b.dataset.mode === mode));
  $('play').hidden = mode === 'compare';
  $('compare').hidden = mode !== 'compare';
  $('humanBox').hidden = mode !== 'human';
  $('curveBox').hidden = mode === 'human';
  if (mode === 'human') { S.human = null; $('humanMsg').textContent = ''; }
  else if (S.human || S.shadow) { S.human = S.shadow = null; newEpisode(false); } // the human run reset the network state
  if (mode === 'compare') drawCompare();
}
document.querySelectorAll('[data-mode]').forEach(b => (b.onclick = () => setMode(b.dataset.mode)));
document.querySelectorAll('[data-speed]').forEach(b => (b.onclick = () => {
  S.speed = b.dataset.speed === 'turbo' ? 'turbo' : +b.dataset.speed;
  document.querySelectorAll('[data-speed]').forEach(x => x.classList.toggle('on', x === b));
}));
$('agentSel').onchange = (e) => { S.agentKind = e.target.value; $('blockedChk').disabled = S.agentKind !== 'bio'; buildFly(); };
$('wiringSel').onchange = (e) => { S.wiring = e.target.value; buildFly(); };
$('blockedChk').onchange = (e) => { S.blocked = e.target.checked; buildFly(); };
$('resetBtn').onclick = buildFly;
$('soundChk').onchange = (e) => { S.sound = e.target.checked; };
$('humanStart').onclick = startHuman;

// ---------- main loop ----------
buildFly();
let last = performance.now();
function frame(now) {
  const dt = Math.min(100, now - last);
  last = now;
  let ep = S.ep, t, resultOf, press = null;
  if (S.mode === 'human') {
    advanceHuman(now);
    const h = S.human;
    ep = S.shadow || S.ep;
    t = h ? now - h.t0 : -FALL_MS;
    resultOf = (s) => h?.res[s];
    press = h?.press;
    const hits = h ? h.hits : 0, n = SONGS.train.notes.filter(x => x >= 0).length;
    $('hud').textContent = `You vs fly · hits ${hits} / ${n} · fly hits ${S.shadow ? S.shadow.hits : 0}`;
    $('progress').firstChild.style.width = `${Math.max(0, Math.min(100, (100 * t) / songMs(SONGS.train)))}%`;
  } else {
    advanceWatch(dt);
    ep = S.ep;
    t = ep.t;
    resultOf = (s) => { const ev = ep.events[s]; return ev && (ev.action === ev.note ? 'hit' : ev.action === WAIT ? 'miss' : 'wrong'); };
    $('hud').textContent = `Episode ${S.epNo} · "${SONGS.train.name}" · hits ${ep.hits} · wrong lane ${ep.wrong} · misses ${ep.misses} · false presses ${ep.falsePresses}`;
    $('progress').firstChild.style.width = `${(100 * ep.t) / songMs(SONGS.train)}%`;
    drawCurve();
  }
  if (S.mode !== 'compare') drawGame(t, resultOf, press);
  brain.frame(ep.net);
  updateStory(ep);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
