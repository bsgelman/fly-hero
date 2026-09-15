// Assert-based checks. Run: node tools/test.mjs
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { Network, wiring, P } from '../js/sim.js';
import { reward, makeSong, SONGS, WAIT, Episode } from '../js/game.js';
import { BioRL, BIO_DEFAULTS } from '../js/agents.js';

const sub = JSON.parse(readFileSync(new URL('../data/subgraph.json', import.meta.url)));
const check = (name, fn) => { fn(); console.log('ok -', name); };

check('driven neuron spikes, isolated neuron silent, inhibition lowers v', () => {
  const tiny = { neurons: { role: ['vpn', 'kc', 'kc'], tag: [0, -1, -1] }, edges: { pre: [0], post: [1], w: [-50] } };
  const net = new Network(tiny, wiring(tiny));
  net.setDrive([0], 150);
  let minV1 = 0;
  for (let t = 0; t < 1000; t++) { net.step(); minV1 = Math.min(minV1, net.v[1]); }
  assert.ok(net.count[0] > 50, `driven spikes ${net.count[0]}`);
  assert.equal(net.count[2], 0);
  assert.ok(minV1 < P.v0 - 1, `inhibited v ${minV1}`);
});

function blockDegrees(e, role) {
  const m = new Map();
  const add = (k) => m.set(k, (m.get(k) || 0) + 1);
  for (let i = 0; i < e.pre.length; i++) {
    const b = role[e.pre[i]] + '>' + role[e.post[i]];
    add(b + ' out ' + e.pre[i]); add(b + ' in ' + e.post[i]); add(b + ' w ' + e.w[i] + ' ' + e.pre[i]);
  }
  return m;
}

check('shuffled wiring preserves per-block degrees and weights, changes endpoints', () => {
  const real = wiring(sub, 'real'), sh = wiring(sub, 'shuffled', 3);
  assert.deepEqual(blockDegrees(sh, sub.neurons.role), blockDegrees(real, sub.neurons.role));
  let moved = 0;
  for (let i = 0; i < real.post.length; i++) moved += real.post[i] !== sh.post[i];
  assert.ok(moved > real.post.length * 0.5, `moved ${moved}`);
});

check('rewards and songs', () => {
  assert.equal(reward(0, 0), 1);
  assert.equal(reward(0, 1), -1);
  assert.equal(reward(0, WAIT), -1);
  assert.equal(reward(-1, 1), -0.5);
  assert.equal(reward(-1, WAIT), 0);
  assert.deepEqual(makeSong(5, 'a').notes, makeSong(5, 'b').notes);
  const rests = SONGS.train.notes.filter(n => n < 0).length / SONGS.train.notes.length;
  assert.ok(rests > 0.1 && rests < 0.4, `rests ${rests}`);
  assert.notDeepEqual(SONGS.train.notes, SONGS.test.notes);
});

check('Agent A learns on real wiring (smoke test, seed 7)', () => {
  const net = new Network(sub, wiring(sub, 'real', 7), { seed: 7, kcMbonGain: BIO_DEFAULTS.kcMbonGain });
  const agent = new BioRL(net, { seed: 7 });
  const before = new Episode(net, agent, SONGS.train, { learn: false }).run().rate;
  const m0 = Float32Array.from(net.m);
  const rates = [];
  for (let e = 0; e < 10; e++) rates.push(new Episode(net, agent, SONGS.train).run().rate);
  const late = (rates[7] + rates[8] + rates[9]) / 3;
  assert.ok(late - before >= 0.25, `before ${before} late ${late}`);
  assert.ok(net.m.some((x, k) => x !== m0[k]), 'plastic weights changed');
});

check('random wiring keeps edge count and weight multiset, no self-loops', () => {
  const real = wiring(sub, 'real'), rnd = wiring(sub, 'random', 3);
  assert.equal(rnd.pre.length, real.pre.length);
  assert.deepEqual([...rnd.w].sort((a, b) => a - b), [...real.w].sort((a, b) => a - b));
  for (let i = 0; i < rnd.pre.length; i++) assert.notEqual(rnd.pre[i], rnd.post[i]);
});
