// Agent A (Bio-RL): dopamine-gated, reward-prediction-error plasticity on KC->MBON synapses inside the connectome.
// Agent B (Deep-RL): frozen connectome, REINFORCE on a linear softmax readout of descending-neuron spikes.
import { rng } from './sim.js';
import { WAIT } from './game.js';

// kcMbonGain 1: in the Janelia/Google male CNS the output neurons respond to visual input with no boost (FlyWire needed 8).
export const BIO_DEFAULTS = {
  kcMbonGain: 1, noiseHz: 400, noiseKick: 1.2, theta: 0.5, cRef: 2, eta: 0.15, mMax: 4, dopaHz: 150, sMax: 12,
};

export class BioRL {
  constructor(net, opts = {}) {
    this.o = { ...BIO_DEFAULTS, ...opts };
    this.kind = 'bio';
    this.rand = rng((opts.seed ?? 1) * 31 + 7);
    net.setNoise(net.idx.mbon, this.o.noiseHz, this.o.noiseKick);
    this.plasticPre = net.plasticPre;
  }

  decide(net) {
    const rates = net.groupMbon.map(ids => ids.reduce((s, i) => s + net.count[i], 0) / ids.length);
    const best = Math.max(...rates);
    const winners = [0, 1, 2].filter(k => rates[k] === best);
    const action = best >= this.o.theta ? winners[Math.floor(this.rand() * winners.length)] : WAIT;
    this.rates = rates;
    this.action = action;
    this.V = action === WAIT ? 0 : Math.tanh(best / this.o.cRef);
    if (action !== WAIT) this.elig = this.plasticPre[action].map(p => net.trace[p]); // eligibility snapshot
    return action;
  }

  reward(net, r) {
    this.delta = r - this.V; // reward-prediction error
    const hz = this.o.blocked ? 0 : this.o.dopaHz;
    net.setDrive(net.idx.pam, hz * Math.max(this.delta, 0) / 2);
    net.setDrive(net.idx.ppl1, hz * Math.max(-this.delta, 0) / 2);
  }

  afterDopamine(net, learn) {
    net.setDrive(net.idx.pam, 0);
    net.setDrive(net.idx.ppl1, 0);
    const sum = (ids) => ids.reduce((s, i) => s + net.count[i], 0);
    this.D = sum(net.idx.pam) / (net.idx.pam.length * this.o.sMax) - sum(net.idx.ppl1) / (net.idx.ppl1.length * this.o.sMax);
    if (!learn || this.action === WAIT) return; // no press -> no eligibility -> no weight change
    const edges = net.plastic[this.action], m = net.m, { eta, mMax } = this.o;
    for (let n = 0; n < edges.length; n++) {
      const k = edges[n], nm = m[k] + eta * this.D * this.elig[n];
      m[k] = nm < 0 ? 0 : nm > mMax ? mMax : nm;
    }
  }
}

// lr picked from a seed-1 sweep {0.03, 0.01, 0.003, 0.001} on the male CNS: highest mean of songs 26 to 30 (see docs/lab-notebook.md)
export const DEEP_DEFAULTS = { lr: 0.03, baselineRate: 0.05 }; // network built with BIO_DEFAULTS.kcMbonGain for both agents

export class DeepRL {
  constructor(net, opts = {}) {
    this.o = { ...DEEP_DEFAULTS, ...opts };
    this.kind = 'deep';
    this.rand = rng((opts.seed ?? 1) * 31 + 11);
    this.F = net.idx.dn.length;
    this.W = [0, 1, 2, 3].map(() => new Float32Array(this.F));
    this.b = new Float32Array(4);
    this.baseline = 0;
    this.mu = new Float32Array(this.F);
    this.var = new Float32Array(this.F).fill(1);
    this.seen = 0;
  }

  decide(net) {
    // Standardize each input with running statistics. Raw spike counts from hundreds of active neurons swamp the
    // softmax before it can learn, even though a readout told the answer decodes the lane from them perfectly.
    const raw = Float32Array.from(net.idx.dn, i => Math.log1p(net.count[i]));
    const k = 1 / Math.min(++this.seen, 200);
    for (let j = 0; j < this.F; j++) {
      const d = raw[j] - this.mu[j];
      this.mu[j] += k * d;
      this.var[j] += k * (d * (raw[j] - this.mu[j]) - this.var[j]);
    }
    const f = this.f = raw.map((v, j) => (v - this.mu[j]) / Math.sqrt(this.var[j] + 1e-3));
    const logits = this.W.map((w, a) => w.reduce((s, x, j) => s + x * f[j], this.b[a]));
    const mx = Math.max(...logits), ex = logits.map(z => Math.exp(z - mx)), Z = ex.reduce((s, x) => s + x, 0);
    this.p = ex.map(x => x / Z);
    let u = this.rand(), a = 0;
    while (a < 3 && u > this.p[a]) u -= this.p[a++];
    return (this.action = a);
  }

  reward(net, r) { this.r = r; }

  afterDopamine(net, learn) {
    if (!learn) return;
    const adv = this.r - this.baseline;
    this.baseline += this.o.baselineRate * (this.r - this.baseline);
    for (let a = 0; a < 4; a++) {
      const gcoef = this.o.lr * adv * ((a === this.action ? 1 : 0) - this.p[a]);
      this.b[a] += gcoef;
      const w = this.W[a];
      for (let j = 0; j < this.F; j++) w[j] += gcoef * this.f[j];
    }
  }
}
