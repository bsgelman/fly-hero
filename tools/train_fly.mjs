// Trains the main fly (Bio-RL, real wiring, seed 1) for 30 songs and saves its learned KC->MBON weights,
// so "You vs fly" starts against a practised fly instead of an untrained one.
// Run: node tools/train_fly.mjs   (writes data/trained_fly.bin)
import { readFileSync, writeFileSync } from 'node:fs';
import { Network, wiring } from '../js/sim.js';
import { Episode, SONGS } from '../js/game.js';
import { BioRL, BIO_DEFAULTS } from '../js/agents.js';

const SONGS_OF_PRACTICE = 30;
const sub = JSON.parse(readFileSync(new URL('../data/subgraph.json', import.meta.url)));
const net = new Network(sub, wiring(sub, 'real', 1), { seed: 1, kcMbonGain: BIO_DEFAULTS.kcMbonGain });
const agent = new BioRL(net, { seed: 1 });

const before = new Episode(net, agent, SONGS.train, { learn: false }).run().rate;
for (let e = 0; e < SONGS_OF_PRACTICE; e++) new Episode(net, agent, SONGS.train).run();
const after = new Episode(net, agent, SONGS.train, { learn: false }).run().rate;

// Only KC->MBON connections learn; save them in the network's fixed plastic-edge order.
const edges = net.plastic.flatMap(ids => [...ids]);
const learned = Float32Array.from(edges, k => net.m[k]);
writeFileSync(new URL('../data/trained_fly.bin', import.meta.url), Buffer.from(learned.buffer));
console.log(`saved ${edges.length} learned connections (${learned.byteLength} bytes)`);
console.log(`frozen score on the practice song: before ${(100 * before).toFixed(0)}%, after ${SONGS_OF_PRACTICE} songs ${(100 * after).toFixed(0)}%`);
