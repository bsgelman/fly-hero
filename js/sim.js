// Leaky integrate-and-fire network on a FlyWire subgraph.
// Parameters from Shiu et al. 2024 (github.com/philshiu/Drosophila_brain_model), dt = 1 ms.

export const P = {
  v0: -52, vth: -45, tauM: 20, tauS: 5, refrMs: 2, delayMs: 2,
  wSyn: 0.275, poiKick: 0.275 * 250,
};

export function rng(seed) { // mulberry32
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffleInPlace(a, rand) {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const tmp = a[i]; a[i] = a[j]; a[j] = tmp;
  }
}

// 'real' | 'shuffled' (permute post endpoints within each pre-role -> post-role block) | 'random' (uniform endpoints)
export function wiring(sub, variant = 'real', seed = 1) {
  const pre = Int32Array.from(sub.edges.pre);
  const post = Int32Array.from(sub.edges.post);
  const w = Float32Array.from(sub.edges.w);
  const rand = rng(seed * 7919 + 13);
  const role = sub.neurons.role;
  if (variant === 'shuffled') {
    const blocks = new Map();
    for (let e = 0; e < pre.length; e++) {
      if (role[pre[e]] === 'vnc' || role[post[e]] === 'vnc') continue; // the nerve cord is never scrambled
      const k = role[pre[e]] + '>' + role[post[e]];
      if (!blocks.has(k)) blocks.set(k, []);
      blocks.get(k).push(e);
    }
    for (const ids of blocks.values()) {
      const posts = ids.map(e => post[e]);
      shuffleInPlace(posts, rand);
      ids.forEach((e, i) => { post[e] = posts[i]; });
    }
  } else if (variant === 'random') {
    // Only the brain circuit is randomised (its neurons come first); the nerve cord keeps its real wiring.
    const N = role.filter(r => r !== 'vnc').length;
    const brainEdges = [];
    for (let e = 0; e < pre.length; e++) {
      if (role[pre[e]] === 'vnc' || role[post[e]] === 'vnc') continue;
      brainEdges.push(e);
      pre[e] = Math.floor(rand() * N);
      do post[e] = Math.floor(rand() * N); while (post[e] === pre[e]);
    }
    const bw = brainEdges.map(e => w[e]);
    shuffleInPlace(bw, rand);
    brainEdges.forEach((e, i) => { w[e] = bw[i]; });
  } else if (variant !== 'real') throw new Error('unknown wiring ' + variant);
  return { pre, post, w };
}

export class Network {
  constructor(sub, edges, { seed = 1, kcMbonGain = 1 } = {}) {
    const N = this.N = sub.neurons.role.length;
    this.sub = sub;
    this.rand = rng(seed);
    const role = sub.neurons.role, tag = sub.neurons.tag;
    this.idx = {};
    for (const r of ['vpn', 'kc', 'apl', 'mbon', 'pam', 'ppl1', 'mid', 'dn', 'vnc'])
      this.idx[r] = Int32Array.from(role.flatMap((x, i) => (x === r ? [i] : [])));
    this.laneVpn = [0, 1, 2].map(l => this.idx.vpn.filter(i => tag[i] === l));
    this.groupMbon = [0, 1, 2].map(k => this.idx.mbon.filter(i => tag[i] === k));
    this.groupOf = new Int8Array(N).fill(-1);
    this.groupMbon.forEach((ids, k) => ids.forEach(i => { this.groupOf[i] = k; }));

    // CSR by presynaptic neuron
    const E = edges.pre.length;
    const rowStart = this.rowStart = new Int32Array(N + 1);
    for (let e = 0; e < E; e++) rowStart[edges.pre[e] + 1]++;
    for (let i = 0; i < N; i++) rowStart[i + 1] += rowStart[i];
    const fill = rowStart.slice(0, N);
    this.col = new Int32Array(E);
    this.w = new Float32Array(E);
    this.m = new Float32Array(E).fill(1);
    const plastic = [[], [], []];
    for (let e = 0; e < E; e++) {
      const i = edges.pre[e], j = edges.post[e], k = fill[i]++;
      this.col[k] = j;
      const isKcMbon = role[i] === 'kc' && role[j] === 'mbon';
      this.w[k] = edges.w[e] * P.wSyn * (isKcMbon ? kcMbonGain : 1);
      if (isKcMbon) plastic[this.groupOf[j]].push(k);
    }
    this.plastic = plastic.map(a => Int32Array.from(a));
    this.plasticPre = this.plastic.map(ids => ids.map(k => this.preOfEdge(k)));

    this.v = new Float32Array(N);
    this.g = new Float32Array(N);
    this.refr = new Uint8Array(N);
    this.trace = new Float32Array(N);      // spike trace, tau = traceTau
    this.count = new Uint16Array(N);       // spikes since resetCounts()
    this.lastSpike = new Float64Array(N);
    this.ring = [0, 1, 2].map(() => new Float32Array(N));
    this.drive = new Float32Array(N);      // Hz, Shiu-style suprathreshold Poisson kicks on v
    this.noise = new Float32Array(N);      // Hz, subthreshold kicks on g
    this.noiseKick = new Float32Array(N);
    this.spikes = new Int32Array(N);
    this.nSpikes = 0;
    this.traceTau = 200;
    this.resetState();
  }

  preOfEdge(k) { // binary search rowStart
    let lo = 0, hi = this.N;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (this.rowStart[mid] <= k) lo = mid; else hi = mid; }
    return lo;
  }

  resetState() {
    this.t = 0;
    this.v.fill(P.v0); this.g.fill(0); this.refr.fill(0); this.trace.fill(0);
    this.count.fill(0); this.lastSpike.fill(-1e9);
    this.ring.forEach(r => r.fill(0));
    this.drive.fill(0);
  }

  resetCounts() { this.count.fill(0); }

  setDrive(indices, hz) { for (const i of indices) this.drive[i] = hz; }

  setNoise(indices, hz, kickMv) { for (const i of indices) { this.noise[i] = hz; this.noiseKick[i] = kickMv; } }

  step() {
    const { N, v, g, refr, trace, drive, noise, noiseKick, rand, spikes } = this;
    const inbox = this.ring[this.t % 3];
    const dm = 1 / P.tauM, dg = Math.exp(-1 / P.tauS), dtr = Math.exp(-1 / this.traceTau);
    let n = 0;
    for (let i = 0; i < N; i++) {
      g[i] += inbox[i]; inbox[i] = 0;
      trace[i] *= dtr;
      if (refr[i] > 0) { refr[i]--; continue; }
      v[i] += (P.v0 - v[i] + g[i]) * dm;
      g[i] *= dg;
      if (noise[i] > 0 && rand() < noise[i] * 1e-3) g[i] += noiseKick[i];
      if (drive[i] > 0 && rand() < drive[i] * 1e-3) v[i] += P.poiKick;
      if (v[i] > P.vth) {
        v[i] = P.v0; g[i] = 0; refr[i] = P.refrMs;
        trace[i] += 1; this.count[i]++; this.lastSpike[i] = this.t;
        spikes[n++] = i;
      }
    }
    const out = this.ring[(this.t + P.delayMs) % 3];
    const { rowStart, col, w, m } = this;
    for (let s = 0; s < n; s++) {
      const i = spikes[s];
      for (let k = rowStart[i]; k < rowStart[i + 1]; k++) out[col[k]] += w[k] * m[k];
    }
    this.nSpikes = n;
    this.t++;
  }
}
