# Fly Hero

A spiking model of a real fruit fly brain learns a three-lane rhythm game by trial and error. The learning happens inside the simulated brain: dopamine strengthens the synapses that led to a hit and weakens the ones that led to a mistake.

**Live demo: [bsgelman.github.io/fly-hero](https://bsgelman.github.io/fly-hero/)**

**33% of notes hit before practice, 99.3% after 30 songs, and 100% on a song it never heard.** With its dopamine neurons blocked, it stays at 33%.

![Fly Hero: the game on the left, the simulated brain and nerve cord on the right](docs/images/watch.png)

The simulation has 6,900 neurons (3,275 in the brain and 3,625 in the nerve cord) and 6,419,481 synapses from the Janelia and Google male fruit fly connectome. Every neuron that fires flashes on the brain view as it happens, all the way down the nerve cord.

## How it works

```
Janelia/Google male CNS connectome (public data)
   │
   ▼  extract_subgraph.py   connectome  →  6,900 neurons, 6,419,481 synapses
   ▼  flyhero/sim.py        synapses    →  spiking neurons, 1 ms steps (NumPy)
   ▼  flyhero/game.py       notes       →  the eye neurons for that lane fire
   ▼  flyhero/agents.py     spikes      →  key press, reward, dopamine, synapse update
   ▼  js/                   same model  →  runs live in the browser: game, brain view and charts
```

| File | Lines | Job |
|---|--:|---|
| `tools/extract_subgraph.py` | 126 | Picks the eye, memory, dopamine, output and nerve cord neurons from the male CNS and writes the network |
| `flyhero/sim.py` | 170 | Leaky integrate-and-fire neurons on the real synapse list, plus the shuffled and random wiring controls |
| `flyhero/game.py` | 87 | Songs, rewards and a millisecond-level game loop |
| `flyhero/agents.py` | 105 | Bio-RL (dopamine learning inside the brain) and Deep-RL (a readout trained with REINFORCE) |
| `flyhero/run_experiments.py` | 88 | Runs every learner and control in parallel and writes `data/results.json` |
| `flyhero/train_fly.py` | 30 | Trains the fly you play against |
| `flyhero/test.py` | 106 | Checks the neuron model, wiring controls, rewards and learning, and that the browser version matches |
| `js/sim.js`, `js/game.js`, `js/agents.js` | 327 | The same model ported line for line, so it runs live in the browser |
| `js/app.js` | 285 | The page: watch the fly learn, you vs fly, compare learners |
| `js/brain.js` | 53 | Brain and nerve cord drawing |

The model is Python with NumPy as its only dependency. Browsers can't run it live, so the page runs a JavaScript port with no frameworks or build step. Both use float64 and the same random number generator, and `flyhero/test.py` checks that they produce identical spikes and key presses from the same seed.

## Results

Each learner practised for 30 songs (the same song 30 times), repeated 5 times from scratch. Scores are the share of notes hit.

| Learner | Before practice | End of practice | New song |
|---|--:|--:|--:|
| **Bio-RL, real wiring** | 33.1% | **99.3%** | **100.0%** |
| Bio-RL, shuffled wiring | 21.9% | 21.9% | 27.2% |
| Bio-RL, random network | 15.0% | 25.9% | 26.2% |
| Bio-RL, dopamine blocked | 31.9% | 33.3% | 35.9% |
| Deep-RL, real wiring | 23.1% | 99.0% | 97.4% |
| Deep-RL, shuffled wiring | 23.1% | 82.8% | 75.9% |
| Deep-RL, random network | 23.1% | 100.0% | 100.0% |
| Deep-RL, real wiring, 150 songs | 21.9% | 98.5% | 95.7% |

End of practice is the average of songs 26 to 30 (146 to 150 for the last row). New song is a different song, played once with learning switched off.

- **Dopamine does the learning.** With it blocked, the brain-style learner stays at 33%.
- **The brain-style learner needs the real wiring.** On shuffled wiring it stays at 22%, and on a random network at 26%. Shuffling makes the network fire out of control, so the fly presses on almost every beat; a random network barely passes the eyes' signal to the output neurons.
- **A trained readout doesn't.** Deep-RL learns from almost any spiking activity: 99.0% on the real wiring, 82.8% on shuffled wiring and 100% on a random network.
- **On the real wiring the two learners are tied.** Bio-RL passed 80% by song 3 or 4 on every run, and Deep-RL by song 4 or 5. Five times the practice (150 songs) didn't change Deep-RL (98.5%).

Full numbers, spreads and the reasoning behind each control are in [docs/lab-notebook.md](docs/lab-notebook.md).

## Design notes

**It reads notes instead of memorising the song.** The fly only sees which lane a note is in, never where it is in the song. That is why it scores the same on a song it never practised.

**Dopamine is made of simulated spikes.** After each press, reward (PAM) or punishment (PPL1) dopamine neurons fire in proportion to how surprising the result was, and their spike counts set how much the synapses change. Blocking dopamine silences those neurons, and learning stops.

**Mistakes can't wipe out what it learned.** Simple dopamine rules often fail because early mistakes far outnumber hits. Here a change shrinks to zero once a pathway already predicts its reward, only synapses that just fired can change, waiting changes nothing, and synapse strength is capped.

**One fly for everything.** The wiring, the sign of every synapse, the drawing and each neuron's position all come from the Janelia and Google male CNS connectome, so every flash is at that neuron's own cell body.

**The nerve cord fires too.** 3,625 nerve cord neurons receive the brain's descending neurons, so activity travels down the neck when the fly acts. The cord only listens: nothing flows back into the brain, and the wiring controls leave it untouched. Replaying runs with and without it gave identical key presses.

**Settings were fixed before the results.** Every learner and control runs through the same code with the same parameters, set once before the experiments.

## Limitations

Each note is a single decision, so this is reward prediction error learning, not learning over long sequences. In real flies, dopamine mostly weakens these synapses, while here reward strengthens them. Dopamine reaches the whole memory centre at once rather than individual compartments. Only the right half of the brain is simulated, and the nerve cord only receives signals from the brain.

## Running it

```bash
git clone https://github.com/bsgelman/fly-hero.git
cd fly-hero
python tools/serve.py
```

Open http://localhost:8766.

- **Watch the fly learn:** pick a learner and wiring, and watch its hit rate climb song by song.
- **You vs fly:** play the same song with J, K and L against a fly that has practised it 30 times.
- **Compare learners:** every learner and control on one chart, with a plain-language guide.

```bash
pip install -r requirements.txt
python -m flyhero.test              # checks, about 4 min (the browser check also needs Node)
python -m flyhero.run_experiments   # every learner and control, about 55 min on 22 cores
python -m flyhero.train_fly         # retrains the You vs fly opponent
```

To rebuild the network from the raw connectome, download three files from the [Janelia male CNS downloads](https://male-cns.janelia.org/download/) into `data/raw/` and run `python tools/extract_subgraph.py` (needs pandas and pyarrow):

- `body-annotations-male-cns-v1.0-minconf-0.5.feather`, saved as `mcns_annotations.feather`
- `connectome-weights-male-cns-v1.0-minconf-0.5.feather`, saved as `mcns_weights.feather` (about 1 GB)
- `body-neurotransmitters-male-cns-v1.0.feather`, saved as `mcns_neurotransmitters.feather`

## Based on published research

- **Neuron model:** Shiu et al., [Nature 2024](https://doi.org/10.1038/s41586-024-07763-9).
- **Brain wiring, cell types and drawing:** the male fruit fly connectome (male CNS v1.0), released open source by Janelia Research Campus, Google Research, the University of Cambridge and the MRC Laboratory of Molecular Biology. Berg et al., [Cell 2026](https://doi.org/10.1016/j.cell.2026.08.015). [Data](https://male-cns.janelia.org/) (CC-BY) and [Google Research announcement](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/).
- **Dopamine learning rule:** inspired by Bennett, Philippides and Nowotny, [Nature Communications 2021](https://doi.org/10.1038/s41467-021-22592-4).

## License

Code is under the MIT License. The network and brain positions are derived from the Janelia and Google male CNS data (CC-BY), credited above. The typeface is Atkinson Hyperlegible Next under the SIL Open Font License (`fonts/OFL.txt`).
