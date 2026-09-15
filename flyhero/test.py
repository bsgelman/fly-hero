"""Assert-based checks, including that the browser port matches this model. Run: python -m flyhero.test"""
import json
import subprocess
from pathlib import Path

import numpy as np

from .agents import BIO_DEFAULTS, BioRL, DeepRL
from .game import SONGS, WAIT, Episode, make_song, reward
from .sim import V0, Network, wiring

ROOT = Path(__file__).resolve().parents[1]
sub = json.loads((ROOT / 'data/subgraph.json').read_text())
role = np.array(sub['neurons']['role'])


def check(name, fn):
    fn()
    print('ok -', name)


def neuron_model():
    tiny = {'neurons': {'role': ['vpn', 'kc', 'kc'], 'tag': [0, -1, -1]}, 'edges': {'pre': [0], 'post': [1], 'w': [-50]}}
    net = Network(tiny, wiring(tiny))
    net.set_drive([0], 150)
    min_v1 = 0
    for _ in range(1000):
        net.step()
        min_v1 = min(min_v1, net.v[1])
    assert net.count[0] > 50, f'driven spikes {net.count[0]}'
    assert net.count[2] == 0
    assert min_v1 < V0 - 1, f'inhibited v {min_v1}'


def shuffled_wiring():
    code = np.unique(role, return_inverse=True)[1]

    def stats(pre, post, w):  # per role -> role block: out-degrees, in-degrees, and each neuron's outgoing weights
        b = code[pre] * 9 + code[post]
        return [np.unique(b * len(role) + pre, return_counts=True), np.unique(b * len(role) + post, return_counts=True),
                np.stack([b, pre, w])[:, np.lexsort((w, pre, b))]]

    real, sh = wiring(sub, 'real'), wiring(sub, 'shuffled', 3)
    for a, b in zip(stats(*real), stats(*sh)):
        assert all(np.array_equal(x, y) for x, y in zip(a, b)) if isinstance(a, tuple) else np.array_equal(a, b)
    moved = int((real[1] != sh[1]).sum())
    assert moved > len(real[1]) * 0.5, f'moved {moved}'


def rewards_and_songs():
    assert reward(0, 0) == 1 and reward(0, 1) == -1 and reward(0, WAIT) == -1
    assert reward(-1, 1) == -0.5 and reward(-1, WAIT) == 0
    assert make_song(5, 'a')['notes'] == make_song(5, 'b')['notes']
    notes = SONGS['train']['notes']
    rests = notes.count(-1) / len(notes)
    assert 0.1 < rests < 0.4, f'rests {rests}'
    assert notes != SONGS['test']['notes']


def bio_learns():
    net = Network(sub, wiring(sub, 'real', 7), seed=7, kc_mbon_gain=BIO_DEFAULTS['kc_mbon_gain'])
    agent = BioRL(net, seed=7)
    before = Episode(net, agent, SONGS['train'], learn=False).run().rate
    m0 = net.m.copy()
    rates = [Episode(net, agent, SONGS['train']).run().rate for _ in range(10)]
    late = sum(rates[7:]) / 3
    assert late - before >= 0.25, f'before {before} late {late}'
    assert (net.m != m0).any(), 'plastic weights changed'


def random_wiring():
    real, rnd = wiring(sub, 'real'), wiring(sub, 'random', 3)
    assert len(rnd[0]) == len(real[0])
    assert np.array_equal(np.sort(rnd[2]), np.sort(real[2]))
    assert (rnd[0] != rnd[1]).all()


def browser_port_matches():
    cases = [('bio', 'real', 7, 2), ('deep', 'real', 7, 2), ('bio', 'shuffled', 3, 1), ('bio', 'random', 3, 1)]
    js = [subprocess.Popen(['node', 'tools/js_trace.mjs', *map(str, c)], cwd=ROOT, stdout=subprocess.PIPE, text=True) for c in cases]
    for (kind, variant, seed, n_songs), proc in zip(cases, js):
        net = Network(sub, wiring(sub, variant, seed), seed=seed, kc_mbon_gain=BIO_DEFAULTS['kc_mbon_gain'])
        agent = BioRL(net, seed=seed) if kind == 'bio' else DeepRL(net, seed=seed)
        songs = []
        for _ in range(n_songs):
            ep = Episode(net, agent, SONGS['train'])
            spikes = checksum = 0
            while not ep.done:
                ep.step()
                spikes += len(net.spikes)
                checksum += ((net.t - 1) % 1000) * 7 * len(net.spikes) + int(net.spikes.sum())
            songs.append({'actions': [e['action'] for e in ep.events], 'spikes': spikes, 'checksum': checksum})
        out = json.loads(proc.communicate()[0])
        assert out['songs'] == songs, f'{kind} {variant}: spikes or key presses differ'
        # the learned values match to rounding: exp, tanh and log1p differ in the last digit between Node and NumPy
        assert abs(out['mSum'] - net.m.sum()) < 1e-9 * net.m.sum(), (out['mSum'], net.m.sum())
        w_sum = agent.W.sum() if kind == 'deep' else 0
        assert abs(out['wSum'] - w_sum) < 1e-9 * (1 + abs(w_sum)), (out['wSum'], w_sum)


check('driven neuron spikes, isolated neuron silent, inhibition lowers v', neuron_model)
check('shuffled wiring preserves per-block degrees and weights, changes endpoints', shuffled_wiring)
check('rewards and songs', rewards_and_songs)
check('random wiring keeps edge count and weight multiset, no self-loops', random_wiring)
check('browser port (js/) gives identical spikes and key presses', browser_port_matches)
check('Bio-RL learns on real wiring (smoke test, seed 7)', bio_learns)
