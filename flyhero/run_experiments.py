"""Runs every learner and wiring condition in parallel and writes data/results.json. Run: python -m flyhero.run_experiments"""
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from .agents import BIO_DEFAULTS, DEEP_DEFAULTS, BioRL, DeepRL
from .game import SONGS, Episode
from .sim import Network, wiring

ROOT = Path(__file__).resolve().parents[1]
SEEDS, EPISODES = [1, 2, 3, 4, 5], 30
CONDITIONS = {
    'A-real': {'agent': 'bio', 'wiring': 'real'},
    'A-shuffled': {'agent': 'bio', 'wiring': 'shuffled'},
    'A-random': {'agent': 'bio', 'wiring': 'random'},
    'A-dopamine-blocked': {'agent': 'bio', 'wiring': 'real', 'blocked': True},
    'B-real': {'agent': 'deep', 'wiring': 'real'},
    'B-shuffled': {'agent': 'deep', 'wiring': 'shuffled'},
    'B-random': {'agent': 'deep', 'wiring': 'random'},
    # fairness check: give the Deep-RL readout 5x the training budget
    'B-real-150ep': {'agent': 'deep', 'wiring': 'real', 'episodes': 150, 'seeds': [1, 2, 3]},
}


def run_one(name, seed):
    c = CONDITIONS[name]
    sub = json.loads((ROOT / 'data/subgraph.json').read_text())
    net = Network(sub, wiring(sub, c['wiring'], seed), seed=seed, kc_mbon_gain=BIO_DEFAULTS['kc_mbon_gain'])
    agent = BioRL(net, seed=seed, blocked=c.get('blocked', False)) if c['agent'] == 'bio' else DeepRL(net, seed=seed)
    before = Episode(net, agent, SONGS['train'], learn=False).run().rate  # untrained, frozen
    curve, false_presses = [], []
    for _ in range(c.get('episodes', EPISODES)):
        ep = Episode(net, agent, SONGS['train']).run()
        curve.append(round(ep.rate, 4))
        false_presses.append(ep.false_presses)
    test = Episode(net, agent, SONGS['test'], learn=False).run().rate
    return {'before': round(before, 4), 'curve': curve, 'falsePresses': false_presses, 'test': round(test, 4)}


def mean(a):
    return sum(a) / len(a)


def sd(a):
    return (mean([(x - mean(a)) ** 2 for x in a])) ** 0.5


def episodes_to_80(curve):
    return next((e + 1 for e in range(2, len(curve)) if (curve[e] + curve[e - 1] + curve[e - 2]) / 3 >= 0.8), None)


def pct(x):
    return f'{100 * x:.1f}'


if __name__ == '__main__':
    t0 = time.time()
    jobs = sorted(((n, s) for n, c in CONDITIONS.items() for s in c.get('seeds', SEEDS)), key=lambda j: 'episodes' not in CONDITIONS[j[0]])
    runs = {}
    with ProcessPoolExecutor() as pool:  # one process per run; the long 150-song runs start first
        futures = {pool.submit(run_one, *j): j for j in jobs}
        for f in as_completed(futures):
            runs[futures[f]] = f.result()
            print(f'{len(runs)}/{len(jobs)} runs done ({time.time() - t0:.0f}s)', flush=True)

    train = SONGS['train']['notes']
    results = {
        'meta': {'date': datetime.now(timezone.utc).isoformat(), 'episodes': EPISODES, 'seeds': SEEDS,
                 'songs': [SONGS['train']['name'], SONGS['test']['name']], 'notesPerEpisode': len(train) - train.count(-1),
                 'bio': BIO_DEFAULTS, 'deep': DEEP_DEFAULTS},
        'conditions': {},
    }
    for name, c in CONDITIONS.items():
        rs = [runs[name, s] for s in c.get('seeds', SEEDS)]
        last5 = [mean(r['curve'][-5:]) for r in rs]
        to80 = [episodes_to_80(r['curve']) for r in rs]
        summary = {
            'before': mean([r['before'] for r in rs]), 'ep1': mean([r['curve'][0] for r in rs]),
            'last5': mean(last5), 'last5sd': sd(last5), 'test': mean([r['test'] for r in rs]), 'testsd': sd([r['test'] for r in rs]),
            'to80': to80, 'reached80': sum(x is not None for x in to80), 'falsePressesLast': mean([r['falsePresses'][-1] for r in rs]),
        }
        results['conditions'][name] = {**c, 'runs': rs, 'summary': summary}
        print(f"{name:<20} before {pct(summary['before']):>5}  ep1 {pct(summary['ep1']):>5}  last5 {pct(summary['last5']):>5} ±{pct(summary['last5sd']):>4}  "
              f"test {pct(summary['test']):>5} ±{pct(summary['testsd']):>4}  to80 {to80}  falsePresses(last) {summary['falsePressesLast']:.1f}")
    (ROOT / 'data/results.json').write_text(json.dumps(results, separators=(',', ':')))
    print('\nwrote data/results.json')
