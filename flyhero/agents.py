"""The two learners. js/agents.js is the browser port.

Bio-RL: dopamine-gated, reward-prediction-error plasticity on KC->MBON synapses inside the connectome.
Deep-RL: frozen connectome, REINFORCE on a linear softmax readout of descending-neuron spikes.
"""
import math

import numpy as np

from .game import WAIT
from .sim import Rng

# kc_mbon_gain 1: in the Janelia/Google male CNS the output neurons respond to visual input with no boost (FlyWire needed 8).
BIO_DEFAULTS = dict(kc_mbon_gain=1, noise_hz=400, noise_kick=1.2, theta=0.5, c_ref=2, eta=0.15, m_max=4, dopa_hz=150, s_max=12)


class BioRL:
    kind = 'bio'

    def __init__(self, net, seed=1, blocked=False, **opts):
        self.o = {**BIO_DEFAULTS, **opts}
        self.blocked = blocked
        self.rand = Rng(seed * 31 + 7)
        net.set_noise(net.idx['mbon'], self.o['noise_hz'], self.o['noise_kick'])
        self.action, self.V, self.delta = WAIT, 0.0, 0.0

    def decide(self, net):
        rates = [int(net.count[ids].sum()) / len(ids) for ids in net.group_mbon]
        best = max(rates)
        winners = [k for k in range(3) if rates[k] == best]
        action = winners[int(self.rand() * len(winners))] if best >= self.o['theta'] else WAIT
        self.rates, self.action = rates, action
        self.V = 0.0 if action == WAIT else math.tanh(best / self.o['c_ref'])
        if action != WAIT:
            self.elig = net.trace[net.plastic_pre[action]]  # eligibility snapshot (fancy indexing copies)
        return action

    def reward(self, net, r):
        self.delta = r - self.V  # reward-prediction error
        hz = 0 if self.blocked else self.o['dopa_hz']
        net.set_drive(net.idx['pam'], hz * max(self.delta, 0) / 2)
        net.set_drive(net.idx['ppl1'], hz * max(-self.delta, 0) / 2)

    def after_dopamine(self, net, learn):
        pam, ppl1, s_max = net.idx['pam'], net.idx['ppl1'], self.o['s_max']
        net.set_drive(pam, 0)
        net.set_drive(ppl1, 0)
        self.D = int(net.count[pam].sum()) / (len(pam) * s_max) - int(net.count[ppl1].sum()) / (len(ppl1) * s_max)
        if not learn or self.action == WAIT:
            return  # no press -> no eligibility -> no weight change
        k = net.plastic[self.action]
        net.m[k] = np.clip(net.m[k] + self.o['eta'] * self.D * self.elig, 0, self.o['m_max'])


# lr picked from a seed-1 sweep {0.03, 0.01, 0.003, 0.001} on the male CNS: highest mean of songs 26 to 30 (see docs/lab-notebook.md)
DEEP_DEFAULTS = dict(lr=0.03, baseline_rate=0.05)  # network built with BIO_DEFAULTS['kc_mbon_gain'] for both agents


class DeepRL:
    kind = 'deep'

    def __init__(self, net, seed=1, **opts):
        self.o = {**DEEP_DEFAULTS, **opts}
        self.rand = Rng(seed * 31 + 11)
        F = len(net.idx['dn'])
        self.W, self.b, self.baseline = np.zeros((4, F)), np.zeros(4), 0.0
        self.mu, self.var, self.seen = np.zeros(F), np.ones(F), 0

    def decide(self, net):
        # Standardize each input with running statistics. Raw spike counts from hundreds of active neurons swamp the
        # softmax before it can learn, even though a readout told the answer decodes the lane from them perfectly.
        raw = np.log1p(net.count[net.idx['dn']].astype(np.float64))
        self.seen += 1
        k = 1 / min(self.seen, 200)
        d = raw - self.mu
        self.mu += k * d
        self.var += k * (d * (raw - self.mu) - self.var)
        f = self.f = (raw - self.mu) / np.sqrt(self.var + 1e-3)
        # summed left to right like the JS; np.dot adds in a different order and rounds differently
        logits = [float(np.cumsum(np.concatenate(([self.b[a]], self.W[a] * f)))[-1]) for a in range(4)]
        mx = max(logits)
        ex = [math.exp(z - mx) for z in logits]
        Z = 0.0
        for x in ex:
            Z += x
        self.p = [x / Z for x in ex]
        u, a = self.rand(), 0
        while a < 3 and u > self.p[a]:
            u -= self.p[a]
            a += 1
        self.action = a
        return a

    def reward(self, net, r):
        self.r = r

    def after_dopamine(self, net, learn):
        if not learn:
            return
        adv = self.r - self.baseline
        self.baseline += self.o['baseline_rate'] * (self.r - self.baseline)
        for a in range(4):
            gcoef = self.o['lr'] * adv * ((1 if a == self.action else 0) - self.p[a])
            self.b[a] += gcoef
            self.W[a] += gcoef * self.f
