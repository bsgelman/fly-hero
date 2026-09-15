# Fly Hero

A spiking model built from a real fruit-fly connectome (FlyWire v783) learns a 3-lane rhythm game by trial and error. On the left is the game. On the right is the brain: all 139,248 neuron positions drawn as a point cloud, with the 3,371 simulated neurons flashing when they spike.

Everything in the brain panel is a **simulation** (a leaky integrate-and-fire model), not recorded fly activity.

## Run it

```bash
python -m http.server 8000
```

Open http://localhost:8000. No build step, no dependencies: plain HTML, canvas and ES modules.

- **Watch the fly learn.** Episodes play back to back and the curve shows hit % per episode. One menu picks the fly: Agent A or B on real, shuffled or random wiring, or Agent A with dopamine blocked. Speeds are 1×, 4× and max.
- **You vs fly.** Play the same song with J K L (±150 ms window). The fly plays alongside you with learning frozen.
- **Compare learners.** Curves and a table from `data/results.json`.

Checks and experiments (Node 24, no packages):

```bash
node tools/test.mjs
node tools/run_experiments.mjs
```

The tests take about 10 s. The experiments take about 12 min and rewrite `data/results.json`.

To rebuild the data (Python with pandas and pyarrow), download two files into `data/raw/`:

- `Connectivity_783.parquet` from https://github.com/philshiu/Drosophila_brain_model
- `supplemental_files/Supplemental_file1_neuron_annotations.tsv` from https://github.com/flyconnectome/flywire_annotations, saved as `annotations.tsv`

Then run:

```bash
python tools/extract_subgraph.py
```

## What is simulated

This is the right-hemisphere mushroom-body loop, selected by fixed rules in `tools/extract_subgraph.py`:

| Role | Rule | Neurons |
|---|---|---|
| eyes | visual projection neurons with ≥10 synapses onto right Kenyon cells, dealt round-robin into 3 lanes | 58 |
| memory | all right Kenyon cells (KCs) | 2,597 |
| inhibition | the right APL neuron | 1 |
| output | MBONs with ≥500 synapses from right KCs, dealt round-robin into 3 action groups | 35 |
| reward / punishment | right PAM / PPL1 dopamine neurons | 153 / 8 |
| downstream | neurons with ≥10 synapses from an MBON and ≥10 onto a descending neuron | 206 |
| descending | descending neurons with ≥10 synapses from an MBON or downstream neuron | 313 |

Every connectome edge between these neurons is kept: **3,371 neurons, 233,686 connections, 650,482 synapses**.

The neuron model is Shiu et al. 2024's LIF: v_rest −52 mV, threshold −45 mV, τ_m 20 ms, τ_syn 5 ms, 0.275 mV per synapse (signed by neurotransmitter), and 150 Hz Poisson input for driven neurons. This project runs it at **dt = 1 ms**, coarser than the original, with delay and refractory period rounded to 2 ms.

The brain panel is a front view of `pos_x`/`pos_y` from the FlyWire annotations, drawn as grey dots on a plain 2D canvas. Only simulated neurons ever flash: black for spikes, green for reward dopamine (PAM), red for punishment dopamine (PPL1). FlyWire v783 covers the brain only, so no nerve cord is drawn.

## The RL problem, stated honestly

| | |
|---|---|
| Song | 48 slots, 400 ms apart. Each slot is a note in lane 1–3 or a rest (16 of 48 are rests). |
| State | What the fly's eyes see as a slot approaches: the visual neurons for that lane get 150 Hz input from 200 ms before the hit line. |
| Actions | Press lane 1, 2 or 3, or wait. The press registers 40 ms after the hit line, inside the ±150 ms window humans get. |
| Reward | +1 hit, −1 wrong lane, −1 miss, −0.5 press on a rest, 0 wait on a rest |
| Episode | One play of the song |

Actions don't change what comes next, so this is a **contextual bandit**, not a multi-step problem. Agent A's rule is reward-prediction-error learning with eligibility traces. It is *not* temporal-difference learning across time steps, and nothing here claims it is. Because the state is "which lane", the fly can be tested on a song it never trained on.

### Agent A: Bio-RL (learning inside the connectome)

1. **Choosing.** Lane input drives visual neurons, then Kenyon cells, then MBONs. The action group with the most MBON spikes per neuron in the decision window wins, if it reaches 0.5 spikes per neuron; otherwise the fly waits. Exploration comes only from spiking noise (subthreshold Poisson input on MBONs).
2. **Prediction and error.** Prediction V = tanh(winner spikes / 2). Error δ = reward − V.
3. **Dopamine is simulated.** PAM neurons are driven in proportion to max(δ, 0) and PPL1 neurons to max(−δ, 0), for 100 ms. Their spike counts give the dopamine signal D.
4. **Update.** Only KC→MBON connections onto the chosen group change: m ← clip(m + 0.15 · D · KC trace, 0, 4), where the KC spike trace has τ = 200 ms.
5. **No press, no change.** A wait has no eligible synapses.

This is built to avoid FlyPong's failure, where punishments wore down the useful synapses:
- The error shrinks to zero once a pathway predicts its reward.
- Only the chosen pathway and recently active KCs are eligible.
- The APL keeps KC codes sparse (about 20 of 2,597 KCs fire per lane).
- Weights are bounded.
- Misses change nothing.

### Agent B: Deep-RL (frozen connectome + trained readout)

Plasticity is off and the dopamine neurons get no input. A linear softmax policy reads log(1 + spike count) from the 313 descending neurons and is trained with REINFORCE and a running reward baseline, learning rate 0.002. This is the recipe most "fly brain plays X" demos use; PPO would add nothing on a one-step problem.

### Controls

| Control | What changes |
|---|---|
| shuffled | Within each role-to-role block (e.g. KC→MBON), postsynaptic partners are permuted. Every neuron keeps its per-block in- and out-degree and its synapse counts, and the architecture stays intact. |
| random | A generic network: same neurons, same number of connections, same weights, but random endpoints. |
| dopamine blocked | Agent A with the PAM and PPL1 drive set to zero. |

All the numbers above were set once, before the full experiment runs. They are recorded in `js/agents.js` and in the lab notebook below.

## Results

Setup: 5 seeds × 30 training episodes on "Banana Drift" (32 notes, 16 rests), then one frozen play of the unseen song "Wing Hum". The raw console output is in `notes/experiments.log`; full curves are in `data/results.json`.

- **Before** is a frozen play before any learning.
- **Last 5** is the mean hit % over episodes 26–30.
- **Seeds ≥80%** counts seeds whose 3-episode moving average reached 80%.

All values are mean ± sd across seeds, as % of notes hit.

| Condition | Before | Episode 1 | Last 5 | Seeds ≥80% (episode reached) | Unseen song | False presses on rests (last episode) |
|---|---|---|---|---|---|---|
| **A-real** (Bio-RL) | 40.0 | 66.9 | **96.0 ± 0.3** | 5/5 (3, 3, 4, 4, 3) | **96.4 ± 4.5** | 0 / 16 |
| A-shuffled | 38.1 | 76.3 | 100.0 ± 0.0 | 5/5 (3, 3, 3, 3, 3) | 100.0 ± 0.0 | 0 / 16 |
| A-random | 5.6 | 7.5 | 6.6 ± 13.3 | 0/5 | 9.2 ± 18.5 | 0 / 16 |
| A-dopamine-blocked | 39.4 | 39.4 | 41.0 ± 1.2 | 0/5 | 35.9 ± 2.8 | 0 / 16 |
| B-real (Deep-RL) | 23.1 | 26.9 | 56.3 ± 10.2 | 0/5 | 56.9 ± 12.1 | 11.2 / 16 |
| B-shuffled | 23.1 | 24.4 | 73.6 ± 2.3 | 2/5 (28, 30) | 67.2 ± 6.8 | 12.8 / 16 |
| B-random | 23.1 | 23.1 | 47.5 ± 15.5 | 0/5 | 43.6 ± 17.1 | 13.0 / 16 |
| B-real, 150 episodes (3 seeds) | 21.9 | 30.2 | 74.8 ± 0.6 | 0/3 | 68.4 ± 1.2 | 8.0 / 16 |

For the 150-episode row, "Last 5" means episodes 146–150.

### What the controls support

**Headline (supported).** The Bio-RL fly goes from 40% to 96% of notes hit in 30 plays. Every seed passes 80% by play 3–4. It scores 96% on a song it never trained on. Blocking its dopamine neurons keeps it at 41%.

The learning happens inside the simulated connectome, through dopamine-gated KC→MBON plasticity, with no trained readout.

**Real wiring does not beat shuffled wiring.** Randomly re-pairing partners within each cell-class block did as well or better, for both agents: A-shuffled 100% vs A-real 96%, and B-shuffled 74% vs B-real 56%. Nothing here shows that the specific FlyWire wiring matters for this task.

This fits Caron et al. 2013, who found KC inputs are close to random. One possible reason shuffled did slightly *better* (untested): the shuffle spreads visual input across more KCs and MBONs, which would remove the "group 3 barely hears the eyes" imbalance in the real wiring.

**The architecture does matter.** A generic random network with the same neurons, connection count and weights fails for Agent A (7%). There, visual input doesn't reach the output neurons in a usable way.

So the honest claim: *eyes → sparse Kenyon-cell expansion → MBONs, with dopamine-gated plasticity* does the work, not the exact synapse list.

**Bio-RL vs Deep-RL.** On the same 30-episode budget, dopamine plasticity inside the brain beat a REINFORCE readout on descending neurons in every wiring condition: 96 vs 56 (real), 100 vs 74 (shuffled), 7 vs 48 (random).

This comparison has caveats:
- **Rests aren't a fair fight.** Agent A's "wait on a rest" is built in (no visual input, no MBON spikes), while Agent B has to learn it and still false-presses on most rests.
- **Agent B is barely tuned.** It's a linear policy with one learning-rate sweep, reading neurons two synapses downstream of the mushroom body. A better-tuned readout could do better.
- **The random-network result reverses.** Agent B beats Agent A there (48% vs 7%). A readout can pull *some* lane signal out of arbitrary spiking; plasticity confined to KC→MBON synapses can't, if the eyes don't reach those synapses.

**More training doesn't close the gap.** With 5× the budget (150 episodes, 3 seeds), Deep-RL on real wiring levels off at 74.8% ± 0.6 and scores 68.4% on the unseen song. That's still well below the 96% Bio-RL reaches in 30 episodes.

**Where the 40% starting point comes from.** Before learning, MBON groups 1 and 2 respond to every lane and group 3 barely responds. The untrained fly therefore mostly presses lanes 1–2, which lands about 40% of notes. That's a structural bias, not skill, and it's why the blocked-dopamine control sits at the same level.

## Known biological liberties

1. **Signed plasticity.** Real dopamine at KC→MBON synapses mostly *depresses* them. Here reward strengthens the chosen pathway.
2. **Broadcast dopamine.** All PAM or all PPL1 neurons are driven together, not compartment by compartment.
3. **Prediction computed in JS.** The prediction V is computed from MBON spikes in JavaScript, standing in for MBON→dopamine feedback (roughly as in Bennett et al. 2021).
4. **Winner-take-all.** The action readout over 3 MBON groups is a modeling choice. The groups are dealt round-robin by KC input, fixed before any results were seen.
5. **KC→MBON gain of 8.** Without it the MBONs never fire from visual input (see the lab notebook).
6. **Visual input.** Lanes are mapped onto real visual projection neurons that synapse onto KCs. That visual input is a small minority of what the mushroom body receives (about 6.5k synapses versus about 427k from central neurons, mostly olfactory).

## Lab notebook

**Data.** I wanted the male CNS connectome, but neuPrint needs an auth token, so I used the public FlyWire v783 files. The annotations have positions for all 139,248 neurons. Right-hemisphere neurons have the larger x values, so x is mirrored for a front view.

**First activity probe** (one lane's visual neurons at 150 Hz for 240 ms):
- About 20 visual neurons and 17–25 KCs fire. The code is sparse, and the APL fires a few spikes.
- **At the plain Shiu weights the MBONs were completely silent**, so nothing could learn.
- A KC→MBON gain of 4 gave about 3 spikes per MBON in groups 1–2 but under 0.4 in group 3. I picked gain 8.
- The round-robin MBON grouping left group 3 nearly deaf to vision, because the visually driven KCs talk to a handful of MBONs. I kept the grouping rather than re-pick it after seeing the data; learning has to strengthen group 3's weak pathway.
- Speed: about 0.45 s per 19.4 s song in Node, about 0.7 s in the browser.

**Agent A calibration.** The first parameter set learned on the first try. Seed 1 went from 66% in episode 1 to 97%, and scored 90% on the unseen song; seed 2 behaved similarly. No further tuning was done.
- Episode 1 is already about 66% because learning happens *during* the first play. That's why the runner also records a frozen "before" episode.
- Agent A never presses on rests. Without visual input, noise never reaches the press threshold, so "wait on a rest" is built in, not learned.

**Agent B learning-rate sweep** (seed 1, 30 episodes):

| Learning rate | Result |
|---|---|
| 0.1 | Collapsed straight to always pressing one lane (41% flat). |
| 0.01 | Flat for about 20 episodes, then climbed to 63%. |
| 0.005 | About 40%, then 53% at episode 30. |
| 0.002 | 28% rising to about 60%, still improving. |

I picked 0.002 because it was the steadiest; a single-seed sweep is noisy. Agent B keeps pressing on 8–12 of the 16 rests, because it samples from a softmax.

**Browser check.** Turbo looked stuck in one screenshot. Measured properly, it runs about 0.6 episodes per second; the screenshot had simply caught it right after a restart.

**UI simplified after feedback.** The first page was a dark, neon-style dashboard: WebGL glow, labeled brain regions, a schematic nerve cord, a story strip and sound. Ben found it too busy and too obviously AI-made. It's now a plain white page with native controls, a black-on-white game, and a grey-dot brain on a 2D canvas. The simulation and experiments are unchanged.

## Sources

- Dorkenwald et al. 2024, *Neuronal wiring diagram of an adult brain*, Nature. FlyWire connectome.
- Schlegel et al. 2024, *Whole-brain annotation and multi-connectome cell typing of Drosophila*, Nature. Matsliah et al. 2024, Nature, and Berg et al. 2025, bioRxiv. Annotations and positions (github.com/flyconnectome/flywire_annotations).
- Shiu et al. 2024, *A Drosophila computational brain model reveals sensorimotor processing*, Nature. LIF model and edge list (github.com/philshiu/Drosophila_brain_model).
- Bennett, Philippides & Nowotny 2021, *Learning with reinforcement prediction errors in a model of the Drosophila mushroom body*, Nature Communications.
- Caron, Ruta, Abbott & Axel 2013, *Random convergence of olfactory inputs in the Drosophila mushroom body*, Nature.
- The other fly-brain demos in the brief (Beat Saber fly, FlyPong, fly-tictactoe, Flappy Fly, Wordle fly) are described from the project brief and were not independently checked here.

No branded game assets are used. "Fly Hero", the song names and the tones are original.
