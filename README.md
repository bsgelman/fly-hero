# Fly Hero

A spiking model of a real fruit fly brain learns a three-lane rhythm game by trial and error. The learning happens inside the simulated brain: dopamine strengthens the synapses that led to a hit and weakens the ones that led to a mistake.

**40% of notes hit before practice, 96% after 30 songs, and 96% on a song it never heard.** With its dopamine neurons blocked, it stays at 41%.

![Fly Hero: the game on the left, the simulated brain and nerve cord on the right](docs/images/watch.png)

The brain has 3,371 simulated neurons and 650,482 synapses taken from the FlyWire connectome of an adult fruit fly. Every neuron that fires flashes on the brain view as it happens.

## How it works

```
FlyWire connectome (public data)
   │
   ▼  extract_subgraph.py   connectome  →  3,371 neurons, 650,482 synapses
   ▼  sim.js                synapses    →  spiking neurons, 1 ms steps
   ▼  game.js               notes       →  the eye neurons for that lane fire
   ▼  agents.js             spikes      →  key press, reward, dopamine, synapse update
   ▼  app.js, brain.js      state       →  game, brain view and charts in the browser
```

| File | Lines | Job |
|---|--:|---|
| `tools/extract_subgraph.py` | 116 | Picks the eye, memory, dopamine and output neurons from FlyWire and writes the network |
| `js/sim.js` | 155 | Leaky integrate-and-fire neurons on the real synapse list, plus the shuffled and random wiring controls |
| `js/game.js` | 62 | Songs, rewards and a millisecond-level game loop |
| `js/agents.js` | 89 | Bio-RL (dopamine learning inside the brain) and Deep-RL (a trained readout) |
| `js/app.js` | 294 | The page: watch the fly learn, you vs fly, compare learners |
| `js/brain.js` | 37 | Brain and nerve cord drawing |
| `tools/run_experiments.mjs` | 68 | Runs every learner and control and writes `data/results.json` |
| `tools/train_fly.mjs` | 23 | Trains the fly you play against |
| `tools/test.mjs` | 69 | Checks for the neuron model, wiring controls, rewards and learning |

No frameworks, no build step and no npm packages. The same modules run in Node for the experiments and in the browser for the page.

## Results

Each learner practised for 30 songs (the same song 30 times), repeated 5 times from scratch. Scores are the share of notes hit.

| Learner | Before practice | End of practice | New song |
|---|--:|--:|--:|
| **Bio-RL, real wiring** | 40% | **96%** | **96%** |
| Bio-RL, shuffled wiring | 38% | 100% | 100% |
| Bio-RL, random network | 6% | 7% | 9% |
| Bio-RL, dopamine blocked | 39% | 41% | 36% |
| Deep-RL, real wiring | 23% | 56% | 57% |
| Deep-RL, shuffled wiring | 23% | 74% | 67% |
| Deep-RL, random network | 23% | 48% | 44% |
| Deep-RL, real wiring, 150 songs | 22% | 75% | 68% |

End of practice is the average of songs 26 to 30 (146 to 150 for the last row). New song is a different song, played once with learning switched off.

- **Dopamine does the learning.** Blocking it keeps the fly at 41%.
- **Brain structure matters.** A fully random network never learns (7%).
- **The exact connections don't.** Shuffled wiring keeps the brain's structure but rewires individual synapses at random, and it did as well as the real wiring.
- **Learning inside the brain beat a trained readout.** On real and shuffled wiring, Bio-RL passed 80% by song 3 or 4 on every run. Deep-RL keeps the brain frozen and trains a separate readout, and it reached 75% even with five times the practice.

Full numbers, spreads and the reasoning behind each control are in [docs/lab-notebook.md](docs/lab-notebook.md).

## Design notes

**It reads notes instead of memorising the song.** The fly only sees which lane a note is in, never where it is in the song. That is why it scores the same on a song it never practised.

**Dopamine is made of simulated spikes.** After each press, reward (PAM) or punishment (PPL1) dopamine neurons fire in proportion to how surprising the result was, and their spike counts set how much the synapses change. Blocking dopamine silences those neurons, and learning stops.

**Mistakes can't wipe out what it learned.** Simple dopamine rules often fail because early mistakes far outnumber hits. Here a change shrinks to zero once a pathway already predicts its reward, only synapses that just fired can change, waiting changes nothing, and synapse strength is capped.

**The brain view shows the whole nervous system.** FlyWire maps only the brain, so the drawing uses cell positions from the Janelia male CNS, which includes the nerve cord, and places each simulated neuron at a cell of the same type.

**Settings were fixed before the results.** Every learner and control runs through the same code with the same parameters, set once before the experiments.

## Limitations

Each note is a single decision, so this is reward prediction error learning, not learning over long sequences. In real flies, dopamine mostly weakens these synapses, while here reward strengthens them. Dopamine reaches the whole memory centre at once rather than individual compartments. The output neurons needed an 8× boost before visual input could make them fire. Only the right half of the brain is simulated, and flash positions come from a different fly, so they are approximate.

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
node tools/test.mjs              # checks, about 10 s
node tools/run_experiments.mjs   # every learner and control, about 12 min
node tools/train_fly.mjs         # retrains the You vs fly opponent
```

To rebuild the network from the raw connectome, download three files into `data/raw/` and run `python tools/extract_subgraph.py` (needs pandas and pyarrow):

- `Connectivity_783.parquet` from [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model)
- `supplemental_files/Supplemental_file1_neuron_annotations.tsv` from [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations), saved as `annotations.tsv`
- `body-annotations-male-cns-v1.0-minconf-0.5.feather` from the [Janelia male CNS downloads](https://male-cns.janelia.org/download/), saved as `mcns_annotations.feather`

## Based on published research

- **Brain wiring and cell types:** the FlyWire connectome. Dorkenwald et al., [Nature 2024](https://doi.org/10.1038/s41586-024-07558-y); Schlegel et al., [Nature 2024](https://doi.org/10.1038/s41586-024-07686-5).
- **Neuron model:** Shiu et al., [Nature 2024](https://doi.org/10.1038/s41586-024-07763-9).
- **Brain and nerve cord drawing:** the male fruit fly connectome from Janelia Research Campus and Google. Berg et al., [bioRxiv 2025](https://doi.org/10.1101/2025.10.09.680999).
- **Dopamine learning rule:** inspired by Bennett, Philippides and Nowotny, [Nature Communications 2021](https://doi.org/10.1038/s41467-021-22592-4).

## License

Code is under the MIT License. The network and brain positions are derived from FlyWire and Janelia male CNS data, both CC-BY and credited above. The typeface is Atkinson Hyperlegible Next under the SIL Open Font License (`fonts/OFL.txt`).
