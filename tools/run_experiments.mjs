// Runs every agent x wiring condition headless with the same modules the page uses.
// Writes data/results.json and prints a summary table. Run: node tools/run_experiments.mjs
import { readFileSync, writeFileSync } from 'node:fs';
import { Network, wiring } from '../js/sim.js';
import { Episode, SONGS } from '../js/game.js';
import { BioRL, DeepRL, BIO_DEFAULTS, DEEP_DEFAULTS } from '../js/agents.js';

const sub = JSON.parse(readFileSync(new URL('../data/subgraph.json', import.meta.url)));
const SEEDS = [1, 2, 3, 4, 5], EPISODES = 30;
const CONDITIONS = {
  'A-real': { agent: 'bio', wiring: 'real' },
  'A-shuffled': { agent: 'bio', wiring: 'shuffled' },
  'A-random': { agent: 'bio', wiring: 'random' },
  'A-dopamine-blocked': { agent: 'bio', wiring: 'real', blocked: true },
  'B-real': { agent: 'deep', wiring: 'real' },
  'B-shuffled': { agent: 'deep', wiring: 'shuffled' },
  'B-random': { agent: 'deep', wiring: 'random' },
  // fairness check: give the Deep-RL readout 5x the training budget
  'B-real-150ep': { agent: 'deep', wiring: 'real', episodes: 150, seeds: [1, 2, 3] },
};

function runOne(c, seed, episodes) {
  const net = new Network(sub, wiring(sub, c.wiring, seed), { seed, kcMbonGain: BIO_DEFAULTS.kcMbonGain });
  const agent = c.agent === 'bio' ? new BioRL(net, { seed, blocked: !!c.blocked }) : new DeepRL(net, { seed });
  const before = new Episode(net, agent, SONGS.train, { learn: false }).run().rate; // untrained, frozen
  const curve = [], falsePresses = [];
  for (let e = 0; e < episodes; e++) {
    const ep = new Episode(net, agent, SONGS.train).run();
    curve.push(+ep.rate.toFixed(4));
    falsePresses.push(ep.falsePresses);
  }
  const test = +new Episode(net, agent, SONGS.test, { learn: false }).run().rate.toFixed(4);
  return { before: +before.toFixed(4), curve, falsePresses, test };
}

const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
const sd = (a) => Math.sqrt(mean(a.map(x => (x - mean(a)) ** 2)));
const pct = (x) => (100 * x).toFixed(1);
function episodesTo80(curve) {
  for (let e = 2; e < curve.length; e++) if ((curve[e] + curve[e - 1] + curve[e - 2]) / 3 >= 0.8) return e + 1;
  return null;
}

const results = {
  meta: {
    date: new Date().toISOString(), episodes: EPISODES, seeds: SEEDS, songs: [SONGS.train.name, SONGS.test.name],
    notesPerEpisode: SONGS.train.notes.filter(n => n >= 0).length, bio: BIO_DEFAULTS, deep: DEEP_DEFAULTS,
  },
  conditions: {},
};
const rows = [];
for (const [name, c] of Object.entries(CONDITIONS)) {
  const t0 = performance.now();
  const runs = (c.seeds || SEEDS).map(seed => runOne(c, seed, c.episodes || EPISODES));
  const last5 = runs.map(r => mean(r.curve.slice(-5)));
  const to80 = runs.map(r => episodesTo80(r.curve));
  const summary = {
    before: mean(runs.map(r => r.before)), ep1: mean(runs.map(r => r.curve[0])),
    last5: mean(last5), last5sd: sd(last5), test: mean(runs.map(r => r.test)), testsd: sd(runs.map(r => r.test)),
    to80, reached80: to80.filter(x => x !== null).length, falsePressesLast: mean(runs.map(r => r.falsePresses.at(-1))),
  };
  results.conditions[name] = { ...c, runs, summary };
  rows.push(`${name.padEnd(20)} before ${pct(summary.before).padStart(5)}  ep1 ${pct(summary.ep1).padStart(5)}  last5 ${pct(summary.last5).padStart(5)} ±${pct(summary.last5sd).padStart(4)}  ` +
    `test ${pct(summary.test).padStart(5)} ±${pct(summary.testsd).padStart(4)}  to80 ${JSON.stringify(to80)}  falsePresses(last) ${summary.falsePressesLast.toFixed(1)}`);
  console.log(rows.at(-1), `(${((performance.now() - t0) / 1000).toFixed(0)}s)`);
}
writeFileSync(new URL('../data/results.json', import.meta.url), JSON.stringify(results));
console.log('\nwrote data/results.json');
