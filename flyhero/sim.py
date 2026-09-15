"""Leaky integrate-and-fire network on the male CNS subgraph, 1 ms steps. Parameters from Shiu et al. 2024.

The page runs js/sim.js, a line-for-line port of this file. Both use float64 and the same random number stream,
so they produce identical spikes; python -m flyhero.test checks this.
"""
import numpy as np

V0, VTH, TAU_M, REFR_MS, DELAY_MS, W_SYN = -52.0, -45.0, 20.0, 2, 2, 0.275
POI_KICK = W_SYN * 250
DM = 1 / TAU_M
DG, DTR = 0.8187307530779818, 0.9950124791926823  # exp(-1/5) and exp(-1/200), written out so Python and JS match bit for bit
ROLES = ['vpn', 'kc', 'apl', 'mbon', 'pam', 'ppl1', 'mid', 'dn', 'vnc']
M32 = 0xFFFFFFFF


class Rng:
    """mulberry32: the same stream as rng() in js/sim.js."""

    def __init__(self, seed):
        self.s = seed & M32

    def __call__(self):
        self.s = (self.s + 0x6D2B79F5) & M32
        t = self.s
        t = ((t ^ (t >> 15)) * (t | 1)) & M32
        t ^= (t + ((t ^ (t >> 7)) * (t | 61))) & M32
        return ((t ^ (t >> 14)) & M32) / 4294967296

    def draw(self, n):
        """The next n numbers of the stream at once (mulberry32's state just counts up)."""
        s = (self.s + 0x6D2B79F5 * np.arange(1, n + 1, dtype=np.uint64)) & M32
        if n:
            self.s = int(s[-1])
        t = ((s ^ (s >> 15)) * (s | 1)) & M32
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & M32)) & M32
        return ((t ^ (t >> 14)) & M32) / 4294967296


def shuffle(a, rand):
    """Fisher-Yates in place, consuming random numbers in the same order as js/sim.js."""
    n = len(a)
    if n < 2:
        return
    i = np.arange(n - 1, 0, -1)
    j = np.floor(rand.draw(n - 1) * (i + 1)).astype(np.int64)
    x = a.tolist()
    for i_, j_ in zip(i.tolist(), j.tolist()):
        x[i_], x[j_] = x[j_], x[i_]
    a[:] = x


def wiring(sub, variant='real', seed=1):
    """'real', 'shuffled' (post endpoints permuted within each role -> role block) or 'random' (uniform endpoints)."""
    pre = np.array(sub['edges']['pre'], dtype=np.int64)
    post = np.array(sub['edges']['post'], dtype=np.int64)
    w = np.array(sub['edges']['w'], dtype=np.float64)
    rand = Rng(seed * 7919 + 13)
    role = np.array(sub['neurons']['role'])
    brain = np.flatnonzero((role[pre] != 'vnc') & (role[post] != 'vnc'))  # the nerve cord is never scrambled
    if variant == 'shuffled':
        code = np.unique(role, return_inverse=True)[1]
        keys = code[pre[brain]] * len(ROLES) + code[post[brain]]
        _, first = np.unique(keys, return_index=True)
        for k in keys[np.sort(first)]:  # blocks in order of first appearance, like the JS Map
            ids = brain[keys == k]
            posts = post[ids]
            shuffle(posts, rand)
            post[ids] = posts
    elif variant == 'random':
        # Only the brain circuit is randomised (its neurons come first); the nerve cord keeps its real wiring.
        n = int((role != 'vnc').sum())
        p, q = pre.tolist(), post.tolist()
        for e in brain.tolist():
            p[e] = int(rand() * n)
            q[e] = int(rand() * n)
            while q[e] == p[e]:
                q[e] = int(rand() * n)
        pre[:], post[:] = p, q
        bw = w[brain]
        shuffle(bw, rand)
        w[brain] = bw
    elif variant != 'real':
        raise ValueError(f'unknown wiring {variant}')
    return pre, post, w


class Network:
    def __init__(self, sub, edges, seed=1, kc_mbon_gain=1):
        role, tag = np.array(sub['neurons']['role']), np.array(sub['neurons']['tag'])
        self.N = N = len(role)
        self.rand = Rng(seed)
        self.idx = {r: np.flatnonzero(role == r) for r in ROLES}
        self.lane_vpn = [self.idx['vpn'][tag[self.idx['vpn']] == lane] for lane in range(3)]
        self.group_mbon = [self.idx['mbon'][tag[self.idx['mbon']] == k] for k in range(3)]

        # CSR by presynaptic neuron. A stable sort keeps each row in edge order, the order js/sim.js fills it in.
        pre, post, w = edges
        order = np.argsort(pre, kind='stable')
        self.row_start = np.concatenate(([0], np.cumsum(np.bincount(pre, minlength=N))))
        self.col = post[order]
        kc_mbon = (role[pre] == 'kc') & (role[post] == 'mbon')
        self.w = w[order] * W_SYN * np.where(kc_mbon[order], kc_mbon_gain, 1)
        self.m = np.ones(len(pre))
        slot = np.empty(len(pre), dtype=np.int64)
        slot[order] = np.arange(len(pre))
        # plastic KC->MBON connections per action group, listed in edge order like the JS
        self.plastic = [slot[kc_mbon & (tag[post] == k)] for k in range(3)]
        self.plastic_pre = [pre[order][ks] for ks in self.plastic]

        self.v, self.g, self.trace = np.empty(N), np.empty(N), np.empty(N)  # trace: spike trace, tau = 200 ms
        self.refr = np.empty(N, dtype=np.int64)
        self.count = np.empty(N, dtype=np.uint16)  # spikes since reset_counts()
        self.last_spike = np.empty(N)
        self.ring = np.empty((3, N))
        self.drive = np.empty(N)  # Hz, Shiu-style suprathreshold Poisson kicks on v
        self.noise, self.noise_kick = np.zeros(N), np.zeros(N)  # Hz and mV, subthreshold kicks on g
        self.spikes = np.empty(0, dtype=np.int64)
        self.reset_state()

    def reset_state(self):
        self.t = 0
        self.v.fill(V0)
        for a in (self.g, self.refr, self.trace, self.count, self.ring, self.drive):
            a.fill(0)
        self.last_spike.fill(-1e9)

    def reset_counts(self):
        self.count.fill(0)

    def set_drive(self, indices, hz):
        self.drive[indices] = hz

    def set_noise(self, indices, hz, kick_mv):
        self.noise[indices] = hz
        self.noise_kick[indices] = kick_mv

    def step(self):
        v, g, refr, trace = self.v, self.g, self.refr, self.trace
        inbox = self.ring[self.t % 3]
        g += inbox
        inbox.fill(0)
        trace *= DTR
        resting = refr > 0
        np.subtract(refr, 1, out=refr, where=resting)
        live = ~resting
        v[:] = np.where(live, v + (V0 - v + g) * DM, v)  # whole-array arithmetic is faster than indexing the live neurons
        g[:] = np.where(live, g * DG, g)
        # Random numbers go to neurons in index order: a noise draw, then a drive draw, for each neuron that needs one.
        noisy, driven = np.flatnonzero(live & (self.noise > 0)), np.flatnonzero(live & (self.drive > 0))
        if len(noisy) or len(driven):
            order = np.concatenate((noisy * 2, driven * 2 + 1)).argsort()
            u = np.empty(len(order))
            u[order] = self.rand.draw(len(order))
            hit = noisy[u[:len(noisy)] < self.noise[noisy] * 1e-3]
            g[hit] += self.noise_kick[hit]
            hit = driven[u[len(noisy):] < self.drive[driven] * 1e-3]
            v[hit] += POI_KICK
        spk = self.spikes = np.flatnonzero(live & (v > VTH))
        v[spk] = V0
        g[spk] = 0
        refr[spk] = REFR_MS
        trace[spk] += 1
        self.count[spk] += 1
        self.last_spike[spk] = self.t
        if len(spk):
            starts, lens = self.row_start[spk], self.row_start[spk + 1] - self.row_start[spk]
            k = np.repeat(starts - np.cumsum(lens) + lens, lens) + np.arange(lens.sum())
            # bincount adds in edge order, the same order and rounding as the JS loop
            self.ring[(self.t + DELAY_MS) % 3] += np.bincount(self.col[k], weights=self.w[k] * self.m[k], minlength=self.N)
        self.t += 1
