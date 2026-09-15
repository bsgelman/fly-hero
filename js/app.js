// Page: modes, game canvas, charts, live simulation loop.
import { Network, wiring } from './sim.js';
import { Episode, SONGS, WAIT, hitTime, songMs } from './game.js';
import { BioRL, DeepRL, BIO_DEFAULTS } from './agents.js';
import { Brain } from './brain.js';

const $ = (id) => document.getElementById(id);
const load = (url, as) => fetch(url).then(r => { if (!r.ok) throw new Error(`${url}: ${r.status}`); return r[as](); });
let sub, pointsBuf;
try {
  [sub, pointsBuf] = await Promise.all([load('data/subgraph.json', 'json'), load('data/brain_points.bin', 'arrayBuffer')]);
} catch (err) {
  $('hud').textContent = 'The brain data did not load. Serve this folder with python -m http.server, then reload the page.';
  throw err;
}
const results = await fetch('data/results.json').then(r => (r.ok ? r.json() : null)).catch(() => null);
const points = new Uint16Array(pointsBuf);

// Colour tokens live in CSS so the canvases follow light and dark mode.
let C;
function readTheme() {
  const cs = getComputedStyle(document.documentElement);
  C = Object.fromEntries(['paper', 'ink', 'graphite', 'rule', 'faint', 'gfp', 'magenta', 'lane1', 'lane2', 'lane3', 'onlane'].map(k => [k, cs.getPropertyValue('--' + k).trim()]));
}
readTheme();
let brain = new Brain($('brain'), $('brainHeader'), points, sub, C);
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
  readTheme();
  brain = new Brain($('brain'), $('brainHeader'), points, sub, C);
  if (S.mode === 'compare') drawCompare();
});

const FALL_MS = 1600, WINDOW_MS = 150, NOTES = SONGS.train.notes.filter(n => n >= 0).length;
const count = (k, one, many) => `${k} ${k === 1 ? one : many}`;
const S = { mode: 'watch', speed: 1, curve: [], epNo: 0 };

// ---------- fly ----------
function buildFly() {
  const [kind, variant] = $('cond').value.split(':');
  const seed = 1 + Math.floor(Math.random() * 1e6);
  S.net = new Network(sub, wiring(sub, variant === 'blocked' ? 'real' : variant, seed), { seed, kcMbonGain: BIO_DEFAULTS.kcMbonGain });
  S.agent = kind === 'bio' ? new BioRL(S.net, { seed, blocked: variant === 'blocked' }) : new DeepRL(S.net, { seed });
  S.curve = [];
  S.epNo = 0;
  newEpisode(true);
}

function newEpisode(count) {
  if (count) S.epNo++;
  S.ep = new Episode(S.net, S.agent, SONGS.train);
  S.seen = 0;
  S.flyPress = [-1e9, -1e9, -1e9];
}

function onFlyEvents(ep) {
  while (S.seen < ep.events.length) {
    const ev = ep.events[S.seen++], a = S.agent;
    if (ev.action !== WAIT) S.flyPress[ev.action] = ev.t;
    const dopamine = a.kind !== 'bio' ? '' : a.o.blocked ? ', dopamine blocked' : a.delta > 0 ? ', reward dopamine' : a.delta < 0 ? ', punishment dopamine' : '';
    const verdict = ev.note < 0 ? (ev.action === WAIT ? 'correct wait' : 'stray press') : ev.action === ev.note ? 'hit' : ev.action === WAIT ? 'miss' : 'wrong lane';
    const el = $('lastEvent'), result = document.createElement('span');
    el.textContent = `${ev.note < 0 ? 'Empty beat' : 'Note in lane ' + (ev.note + 1)}, ${ev.action === WAIT ? 'waited' : 'pressed lane ' + (ev.action + 1)}: `;
    result.textContent = `${verdict} (${ev.reward > 0 ? '+' : ''}${ev.reward})${dopamine}`;
    if (a.kind === 'bio' && !a.o.blocked && a.delta) result.className = a.delta > 0 ? 'flash-gfp' : 'flash-magenta'; // same colour as the dopamine neurons
    el.append(result);
  }
}

function advanceWatch(dt) {
  const deadline = performance.now() + 14;
  let steps = Infinity;
  if (S.speed !== 'max') { // carry leftover milliseconds so 0.5× runs at exactly half speed
    S.carry = (S.carry || 0) + dt * S.speed;
    steps = Math.floor(S.carry);
    S.carry -= steps;
  }
  while (steps-- > 0 && performance.now() < deadline) { // ponytail: slow machines just run slower than requested
    S.ep.step();
    onFlyEvents(S.ep);
    if (S.ep.done) {
      S.curve.push(S.ep.rate);
      newEpisode(true);
    }
  }
}

// ---------- you vs fly ----------
function startHuman() {
  S.human = { t0: performance.now() + 2000, res: [], hits: 0, extra: 0, press: [-1e9, -1e9, -1e9], done: false };
  S.shadow = new Episode(S.net, S.agent, SONGS.train, { learn: false });
  S.seen = 0;
  S.flyPress = [-1e9, -1e9, -1e9];
  $('humanMsg').textContent = 'Get ready…';
}

function advanceHuman(now) {
  const h = S.human;
  if (!h || h.done) return;
  const t = now - h.t0, notes = SONGS.train.notes;
  while (S.shadow.t < t && !S.shadow.done) {
    S.shadow.step();
    onFlyEvents(S.shadow);
  }
  for (let s = 0; s < notes.length; s++) if (notes[s] >= 0 && !h.res[s] && t > hitTime(s) + WINDOW_MS) h.res[s] = 'miss';
  if (t > songMs(SONGS.train) + 300) {
    h.done = true;
    $('humanMsg').textContent = `You: ${Math.round((100 * h.hits) / NOTES)}% (${count(h.extra, 'stray press', 'stray presses')}). ` +
      `Fly: ${Math.round(100 * S.shadow.rate)}% (${count(S.shadow.wrong + S.shadow.falsePresses, 'stray press', 'stray presses')}, trained for ${count(S.curve.length, 'episode', 'episodes')}).`;
  }
}

addEventListener('keydown', (e) => {
  const lane = { j: 0, k: 1, l: 2 }[e.key.toLowerCase()], h = S.human;
  if (S.mode !== 'human' || lane === undefined || !h || h.done || e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
  e.preventDefault(); // keep J/K/L from also changing a focused <select>
  const t = performance.now() - h.t0, notes = SONGS.train.notes;
  h.press[lane] = t;
  let best = -1;
  for (let s = 0; s < notes.length; s++) {
    if (notes[s] !== lane || h.res[s] || Math.abs(hitTime(s) - t) > WINDOW_MS) continue;
    if (best < 0 || Math.abs(hitTime(s) - t) < Math.abs(hitTime(best) - t)) best = s;
  }
  if (best >= 0) {
    h.res[best] = 'hit';
    h.hits++;
  } else h.extra++;
});

// ---------- drawing ----------
function drawGame(t, resultOf, humanPress) {
  const c = $('game'), g = c.getContext('2d'), W = c.width, H = c.height, lw = W / 3, hitY = H - 64;
  const lane = [C.lane1, C.lane2, C.lane3];
  g.clearRect(0, 0, W, H);
  g.fillStyle = C.rule;
  for (const x of [0, lw, 2 * lw, W - 1]) g.fillRect(Math.round(x), 0, 1, H);
  g.fillStyle = C.ink;
  g.fillRect(0, hitY, W, 2);
  const notes = SONGS.train.notes;
  for (let s = 0; s < notes.length; s++) {
    const n = notes[s], d = hitTime(s) - t, res = resultOf(s);
    if (n < 0 || d > FALL_MS || d < -400) continue;
    if (res === 'hit') { // a ring in the lane colour grows and fades after a hit
      const a = Math.max(0, 1 + d / 400);
      g.globalAlpha = a;
      g.strokeStyle = lane[n];
      g.lineWidth = 3;
      g.beginPath();
      g.arc(n * lw + lw / 2, hitY, 22 + (1 - a) * 28, 0, 2 * Math.PI);
      g.stroke();
      g.globalAlpha = 1;
    } else if (d > -300) {
      g.fillStyle = res ? C.faint : lane[n];
      g.fillRect(n * lw + 20, hitY - (d / FALL_MS) * hitY - 10, lw - 40, 20);
    }
  }
  g.font = '600 20px "Atkinson Hyperlegible Next", system-ui, sans-serif';
  g.textAlign = 'center';
  g.lineWidth = 2;
  for (let l = 0; l < 3; l++) {
    const x = l * lw + lw / 2, lit = humanPress ? t - humanPress[l] < 120 : t - S.flyPress[l] < 150;
    g.fillStyle = lit ? lane[l] : C.paper;
    g.fillRect(l * lw + 16, H - 50, lw - 32, 40);
    g.strokeStyle = lane[l];
    g.strokeRect(l * lw + 16, H - 50, lw - 32, 40);
    g.fillStyle = lit ? C.onlane : C.ink;
    g.fillText('JKL'[l], x, H - 23);
    if (humanPress && t - S.flyPress[l] < 150) { // in you-vs-fly, mark the fly's press above the line
      g.fillStyle = C.graphite;
      g.fillText('fly', x, hitY - 12);
    }
  }
  g.textAlign = 'start';
  g.lineWidth = 1;
}

function drawLines(canvas, series, colors, maxX, dashes = []) {
  const g = canvas.getContext('2d'), W = canvas.width, H = canvas.height, L = 30;
  g.clearRect(0, 0, W, H);
  g.fillStyle = C.graphite;
  g.font = '11px "Atkinson Hyperlegible Next", system-ui, sans-serif';
  g.fillText('100%', 0, 12);
  g.fillText('0%', 12, H - 4);
  g.lineWidth = 1;
  g.setLineDash([]);
  g.strokeStyle = C.rule;
  g.strokeRect(L, 4, W - L - 2, H - 8);
  series.forEach((ys, k) => {
    g.strokeStyle = colors[k];
    g.lineWidth = 1.5;
    g.setLineDash(dashes[k] || []);
    g.beginPath();
    ys.forEach((v, e) => g.lineTo(L + (e / maxX) * (W - L - 2), 4 + (1 - v) * (H - 8)));
    g.stroke();
  });
}

// Line shade encodes the wiring, dashes encode the learner; magenta is the dopamine-blocked control.
const SERIES = { 'A-real': ['ink', []], 'A-shuffled': ['graphite', []], 'A-random': ['faint', []], 'A-dopamine-blocked': ['magenta', []], 'B-real': ['ink', [5, 3]], 'B-shuffled': ['graphite', [5, 3]], 'B-random': ['faint', [5, 3]] };

function drawCompare() {
  if (!results) {
    $('cmpTable').textContent = 'No data/results.json yet. Run: node tools/run_experiments.mjs';
    return;
  }
  const names = Object.keys(SERIES).filter(n => results.conditions[n]);
  const means = names.map(n => {
    const runs = results.conditions[n].runs.map(r => [r.before, ...r.curve]);
    return runs[0].map((_, e) => runs.reduce((s, r) => s + r[e], 0) / runs.length);
  });
  drawLines($('cmp'), means, names.map(n => C[SERIES[n][0]]), results.meta.episodes, names.map(n => SERIES[n][1]));
  const pct = (v) => `${Math.round(100 * v)}%`;
  $('cmpTable').innerHTML = '<table><caption class="small">Mean hit rate over seeds. Solid lines are Bio-RL and dashed lines are Deep-RL. Dark, grey and light lines are real, shuffled and random wiring; magenta is Bio-RL with dopamine blocked.</caption><tr><th scope="col">Fly</th><th scope="col">Before</th><th scope="col">Last 5</th><th scope="col">Unseen song</th></tr>' +
    Object.entries(results.conditions).map(([n, c]) =>
      `<tr><td><span class="swatch" aria-hidden="true" style="border-top-color:${SERIES[n] ? C[SERIES[n][0]] : 'transparent'};border-top-style:${n.startsWith('B-') ? 'dashed' : 'solid'}"></span>${n}</td><td class="num">${pct(c.summary.before)}</td><td class="num">${pct(c.summary.last5)}</td><td class="num">${pct(c.summary.test)}</td></tr>`).join('') +
    '</table>';
}

// ---------- controls ----------
function setMode(mode) {
  if (S.shadow) newEpisode(false); // a you-vs-fly round reset the network state
  S.mode = mode;
  S.human = S.shadow = null;
  document.querySelectorAll('[data-mode]').forEach(b => b.setAttribute('aria-pressed', b.dataset.mode === mode));
  $('play').hidden = mode === 'compare';
  $('compare').hidden = mode !== 'compare';
  $('humanBox').hidden = mode !== 'human';
  $('curveBox').hidden = mode === 'human';
  $('humanMsg').textContent = $('lastEvent').textContent = '';
  if (mode === 'compare') drawCompare();
}
document.querySelectorAll('[data-mode]').forEach(b => (b.onclick = () => setMode(b.dataset.mode)));
$('cond').onchange = buildFly;
$('reset').onclick = buildFly;
// read the menu at startup too: browsers restore a select's previous choice on reload
const readSpeed = () => { const v = $('speed').value; S.speed = v === 'max' ? 'max' : +v; };
$('speed').onchange = readSpeed;
readSpeed();
$('humanStart').onclick = startHuman;

// ---------- main loop ----------
buildFly();
let last = performance.now();
function frame(now) {
  const dt = Math.min(100, now - last);
  last = now;
  let ep, t, resultOf;
  if (S.mode === 'human') {
    advanceHuman(now);
    const h = S.human;
    ep = S.shadow || S.ep;
    t = h ? now - h.t0 : -FALL_MS;
    resultOf = (s) => h?.res[s];
    $('hud').textContent = h ? `You ${h.hits}, fly ${S.shadow.hits}, out of ${NOTES} notes` : `${NOTES} notes`;
    $('prog').value = Math.max(0, t / songMs(SONGS.train));
  } else {
    advanceWatch(dt);
    ep = S.ep;
    t = ep.t;
    resultOf = (s) => { const ev = ep.events[s]; return ev && (ev.action === ev.note ? 'hit' : 'miss'); };
    $('hud').textContent = `Episode ${S.epNo}: ${ep.hits} of ${ep.notes} notes hit`;
    $('prog').value = ep.t / songMs(SONGS.train);
    drawLines($('curve'), [S.curve], [C.ink], Math.max(30, S.curve.length - 1));
  }
  if (S.mode !== 'compare') drawGame(t, resultOf, S.mode === 'human' ? S.human?.press : null);
  brain.frame(ep.net);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
