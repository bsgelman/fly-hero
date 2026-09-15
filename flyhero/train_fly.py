"""Trains the main fly (Bio-RL, real wiring, seed 1) for 30 songs and saves its learned KC->MBON weights,
so "You vs fly" starts against a practised fly instead of an untrained one.
Run: python -m flyhero.train_fly   (writes data/trained_fly.bin)
"""
import json
from pathlib import Path

import numpy as np

from .agents import BIO_DEFAULTS, BioRL
from .game import SONGS, Episode
from .sim import Network, wiring

ROOT = Path(__file__).resolve().parents[1]
SONGS_OF_PRACTICE = 30

if __name__ == '__main__':
    sub = json.loads((ROOT / 'data/subgraph.json').read_text())
    net = Network(sub, wiring(sub, 'real', 1), seed=1, kc_mbon_gain=BIO_DEFAULTS['kc_mbon_gain'])
    agent = BioRL(net, seed=1)
    before = Episode(net, agent, SONGS['train'], learn=False).run().rate
    for _ in range(SONGS_OF_PRACTICE):
        Episode(net, agent, SONGS['train']).run()
    after = Episode(net, agent, SONGS['train'], learn=False).run().rate

    # Only KC->MBON connections learn; save them as little-endian float64 in the network's fixed plastic-edge order.
    learned = np.concatenate([net.m[k] for k in net.plastic]).astype('<f8')
    learned.tofile(ROOT / 'data/trained_fly.bin')
    print(f'saved {len(learned)} learned connections ({learned.nbytes} bytes)')
    print(f'frozen score on the practice song: before {100 * before:.0f}%, after {SONGS_OF_PRACTICE} songs {100 * after:.0f}%')
