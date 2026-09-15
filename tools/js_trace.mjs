// Runs the browser port of the model and prints what it did, so python -m flyhero.test can compare it with the Python model.
// Usage: node tools/js_trace.mjs <bio|deep> <real|shuffled|random> <seed> <songs>
import { readFileSync } from 'node:fs';
import { Network, wiring } from '../js/sim.js';
import { Episode, SONGS } from '../js/game.js';
import { BioRL, DeepRL, BIO_DEFAULTS } from '../js/agents.js';

const [kind, variant, seedArg, songsArg] = process.argv.slice(2);
const seed = +seedArg;
const sub = JSON.parse(readFileSync(new URL('../data/subgraph.json', import.meta.url)));
const net = new Network(sub, wiring(sub, variant, seed), { seed, kcMbonGain: BIO_DEFAULTS.kcMbonGain });
const agent = kind === 'bio' ? new BioRL(net, { seed }) : new DeepRL(net, { seed });
const songs = [];
for (let s = 0; s < +songsArg; s++) {
  const ep = new Episode(net, agent, SONGS.train);
  let spikes = 0, checksum = 0;
  while (!ep.done) {
    ep.step();
    for (let n = 0; n < net.nSpikes; n++) { spikes++; checksum += ((net.t - 1) % 1000) * 7 + net.spikes[n]; }
  }
  songs.push({ actions: ep.events.map(e => e.action), spikes, checksum });
}
const mSum = net.m.reduce((s, x) => s + x, 0);
const wSum = agent.W ? agent.W.reduce((s, row) => s + row.reduce((a, x) => a + x, 0), 0) : 0;
console.log(JSON.stringify({ songs, mSum, wSum }));
