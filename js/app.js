// Page: modes, game canvas, charts, live simulation loop.
import { Network, wiring } from './sim.js';
import { Episode, SONGS, WAIT, hitTime, songMs } from './game.js';
import { BioRL, DeepRL, BIO_DEFAULTS } from './agents.js';
import { Brain } from './brain.js';

const $ = (id) => document.getElementById(id);
const [sub, pointsBuf] = await Promise.all([
  fetch('data/subgraph.json').then(r => r.json()),
  fetch('data/brain_points.bin').then(r => r.arrayBuffer()),
]);
const results = await fetch('data/results.json').then(r => (r.ok ? r.json() : null)).catch(() => null);
const brain = new Brain($('brain'), $('brainHeader'), new Uint16Array(pointsBuf), sub);

const FALL_MS = 1600, WINDOW_MS = 150, NOTES = SONGS.train.notes.filter(n => n >= 0).length;
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
    $('lastEvent').textContent = `saw ${ev.note < 0 ? 'nothing' : 'lane ' + (ev.note + 1)}, pressed ${ev.action === WAIT ? 'nothing' : 'lane ' + (ev.action + 1)}, reward ${ev.reward}${dopamine}`;
  }
}

function advanceWatch(dt) {
  const deadline = performance.now() + 14;
  let steps = S.speed === 'max' ? Infinity : Math.round(dt * S.speed);
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
  S.human = { t0: performance.now() + 2000, res: [], hits: 0, extra: 0, done: false };
  S.shadow = new Episode(S.net, S.agent, SONGS.train, { learn: false });
  S.seen = 0;
  S.flyPress = [-1e9, -1e9, -1e9];
  $('humanMsg').textContent = '';
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
    $('humanMsg').textContent = `You: ${Math.round((100 * h.hits) / NOTES)}% (${h.extra} stray presses). ` +
      `Fly: ${Math.round(100 * S.shadow.rate)}% (${S.shadow.wrong + S.shadow.falsePresses} stray presses, trained ${S.curve.length} episodes).`;
  }
}

addEventListener('keydown', (e) => {
  const lane = { j: 0, k: 1, l: 2 }[e.key.toLowerCase()], h = S.human;
  if (S.mode !== 'human' || lane === undefined || !h || h.done) return;
  const t = performance.now() - h.t0, notes = SONGS.train.notes;
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
function drawGame(t, resultOf) {
  const c = $('game'), g = c.getContext('2d'), W = c.width, H = c.height, lw = W / 3, hitY = H - 40;
  g.clearRect(0, 0, W, H);
  g.fillStyle = '#EAEAEA';
  g.fillRect(lw, 0, 1, H);
  g.fillRect(2 * lw, 0, 1, H);
  g.fillStyle = '#111';
  g.fillRect(0, hitY, W, 2);
  const notes = SONGS.train.notes;
  for (let s = 0; s < notes.length; s++) {
    const n = notes[s], d = hitTime(s) - t, res = resultOf(s);
    if (n < 0 || res === 'hit' || d > FALL_MS || d < -300) continue;
    g.fillStyle = res ? '#D5D4D0' : '#2F3437';
    g.fillRect(n * lw + 15, hitY - (d / FALL_MS) * hitY - 6, lw - 30, 12);
  }
  g.font = '12px "SF Mono", Menlo, Consolas, monospace';
  g.fillStyle = '#787774';
  for (let l = 0; l < 3; l++) {
    g.fillText('JKL'[l], l * lw + lw / 2 - 4, H - 4);
    if (t - S.flyPress[l] < 150) g.fillText('fly', l * lw + lw / 2 - 8, hitY + 16);
  }
}

function drawLines(canvas, series, colors, maxX) {
  const g = canvas.getContext('2d'), W = canvas.width, H = canvas.height, L = 30;
  g.clearRect(0, 0, W, H);
  g.fillStyle = '#787774';
  g.font = '10px "SF Mono", Menlo, Consolas, monospace';
  g.fillText('100%', 0, 12);
  g.fillText('0%', 12, H - 4);
  g.lineWidth = 1;
  g.strokeStyle = '#EAEAEA';
  g.strokeRect(L, 4, W - L - 2, H - 8);
  series.forEach((ys, k) => {
    g.strokeStyle = colors[k];
    g.lineWidth = 1.5;
    g.beginPath();
    ys.forEach((v, e) => g.lineTo(L + (e / maxX) * (W - L - 2), 4 + (1 - v) * (H - 8)));
    g.stroke();
  });
}

const COLORS = { 'A-real': '#111', 'A-shuffled': '#787774', 'A-random': '#C9C8C4', 'A-dopamine-blocked': '#9F2F2D', 'B-real': '#1F6C9F', 'B-shuffled': '#6BA4C9', 'B-random': '#B7D4E8' };

function drawCompare() {
  if (!results) {
    $('cmpTable').textContent = 'No data/results.json yet. Run: node tools/run_experiments.mjs';
    return;
  }
  const names = Object.keys(COLORS).filter(n => results.conditions[n]);
  const means = names.map(n => {
    const runs = results.conditions[n].runs.map(r => [r.before, ...r.curve]);
    return runs[0].map((_, e) => runs.reduce((s, r) => s + r[e], 0) / runs.length);
  });
  drawLines($('cmp'), means, names.map(n => COLORS[n]), results.meta.episodes);
  const pct = (v) => `${Math.round(100 * v)}%`;
  $('cmpTable').innerHTML = '<table><tr><th>fly</th><th>before</th><th>last 5</th><th>unseen song</th></tr>' +
    Object.entries(results.conditions).map(([n, c]) =>
      `<tr><td><span class="swatch" style="background:${COLORS[n] || 'transparent'}"></span>${n}</td><td class="num">${pct(c.summary.before)}</td><td class="num">${pct(c.summary.last5)}</td><td class="num">${pct(c.summary.test)}</td></tr>`).join('') +
    '</table>';
}

// ---------- controls ----------
function setMode(mode) {
  if (S.shadow) newEpisode(false); // a you-vs-fly round reset the network state
  S.mode = mode;
  S.human = S.shadow = null;
  document.querySelectorAll('[data-mode]').forEach(b => (b.disabled = b.dataset.mode === mode));
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
$('speed').onchange = (e) => { S.speed = e.target.value === 'max' ? 'max' : +e.target.value; };
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
    $('hud').textContent = h ? `you ${h.hits}, fly ${S.shadow.hits}, of ${NOTES} notes` : `${NOTES} notes`;
    $('prog').value = Math.max(0, t / songMs(SONGS.train));
  } else {
    advanceWatch(dt);
    ep = S.ep;
    t = ep.t;
    resultOf = (s) => { const ev = ep.events[s]; return ev && (ev.action === ev.note ? 'hit' : 'miss'); };
    $('hud').textContent = `episode ${S.epNo} · ${ep.hits} of ${ep.notes} notes hit`;
    $('prog').value = ep.t / songMs(SONGS.train);
    drawLines($('curve'), [S.curve], ['#111'], Math.max(30, S.curve.length - 1));
  }
  if (S.mode !== 'compare') drawGame(t, resultOf);
  brain.frame(ep.net);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
