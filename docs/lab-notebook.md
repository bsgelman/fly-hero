# Fly Hero: technical notes and lab notebook

The full detail behind the [README](../README.md): what is simulated, the learning rules, every control, complete results with spreads, and what I tried along the way.

## The model moves to Python (2026-09-15)

The model, experiments, training and tests are now Python (`flyhero/`, NumPy only). The page still needs the model to run live in the browser, so `js/sim.js`, `js/game.js` and `js/agents.js` are a line-for-line port. Every number below comes from the Python code.

**Same spikes in both languages.** Both versions use float64 throughout and the same random number generator (mulberry32), consume random numbers in the same order, and add synaptic input in the same order, so rounding matches too. The two decay constants, exp(-1/5) and exp(-1/200), are written out as numbers because library exp functions can differ in the last digit. `python -m flyhero.test` runs Bio-RL (real wiring, 2 songs), Deep-RL (real wiring, 2 songs) and Bio-RL on shuffled and random wiring (1 song each) in both languages and checks that every spike and key press is identical. Learned weights agree to within 1e-9; tanh, exp and log1p differ in the last digit between Node and NumPy, which never changed a spike or a press.

**The check found a bug in the old JavaScript.** Bio-RL's eligibility snapshot, the Kenyon cell spike traces used to decide which synapses change, was built with `.map` on an integer array, which returns an integer array. Traces were rounded down to whole numbers, so a synapse whose Kenyon cell fired recently but had a trace below 1 did not change at all. The Python model uses the real values, and the JavaScript port is fixed. The old version also stored weights, traces and the Deep-RL readout as float32, so every condition was rerun.

**Speed.** NumPy runs one song in about 5 to 14 seconds, against about 0.5 seconds for the JavaScript, because each millisecond is a separate set of array operations. The 38 runs go in parallel, one process each: 3,218 seconds (54 minutes) on 22 cores. The You vs fly opponent (Bio-RL, real wiring, seed 1) goes from 28% to 100% on the practice song after 30 songs, and the browser port scores 100% with the saved weights.

**Results from the Python model** (5 repeats of 30 songs; 3 repeats of 150 songs for the last row; mean and spread across repeats):

| Learner | Before practice | Song 1 | End of practice | Runs reaching 80% (song) | New song | Presses on empty beats (last song) |
|---|---|---|---|---|---|---|
| Bio-RL, real wiring | 33.1% | 61.9% | 99.3% ± 0.5 | 5/5 (4, 3, 4, 3, 3) | 100.0% ± 0.0 | 0.0 / 16 |
| Bio-RL, shuffled wiring | 21.9% | 21.9% | 21.9% ± 0.0 | 0/5 | 27.2% ± 1.3 | 15.0 / 16 |
| Bio-RL, random network | 15.0% | 23.1% | 25.9% ± 14.2 | 0/5 | 26.2% ± 13.7 | 0.0 / 16 |
| Bio-RL, dopamine blocked | 31.9% | 36.3% | 33.3% ± 2.6 | 0/5 | 35.9% ± 5.4 | 0.0 / 16 |
| Deep-RL, real wiring | 23.1% | 50.6% | 99.0% ± 0.6 | 5/5 (5, 4, 4, 4, 5) | 97.4% ± 1.6 | 0.6 / 16 |
| Deep-RL, shuffled wiring | 23.1% | 30.6% | 82.8% ± 11.6 | 4/5 (10, 11, 29, 6) | 75.9% ± 15.9 | 9.6 / 16 |
| Deep-RL, random network | 23.1% | 78.1% | 100.0% ± 0.0 | 5/5 (3, 3, 3, 3, 3) | 100.0% ± 0.0 | 0.0 / 16 |
| Deep-RL, real wiring, 150 songs | 21.9% | 47.9% | 98.5% ± 1.6 | 3/3 (5, 4, 4) | 95.7% ± 1.2 | 0.0 / 16 |

**The conclusions hold.** Bio-RL on the real wiring still learns (99.3%, 100% on the new song) and now passes 80% a little sooner (song 3 or 4 on every run, against 3 to 5). It still fails on shuffled (21.9%) and random (25.9%) wiring and with dopamine blocked (33.3%). Deep-RL still learns on every wiring, and the two learners are still tied on the real wiring (99.3% and 99.0%). The rest of this notebook, including the learning rate sweep, was measured with the JavaScript version.

## Switch to the Janelia and Google male CNS (2026-09-15)

The project was meant to use the Janelia and Google male CNS connectome from the start. Its neuPrint service needs a personal login token, which I couldn't get during the unattended first build, so the first version used the public FlyWire (female brain) files instead and borrowed male CNS cell positions only for the drawing. Janelia also publishes the male CNS as plain file downloads with no login, so the whole project now uses it: wiring, synapse signs (from its neurotransmitter predictions), the drawing, and each neuron's own position.

**New network.** Same selection rules, now from the male CNS: 62 eye neurons, 2,045 Kenyon cells, 1 APL, 42 output neurons, 158 reward and 8 punishment dopamine neurons, 461 intermediate and 498 descending neurons. That is 3,275 neurons, 518,428 connections and 1,922,033 synapses; 12 neurons have no recorded cell body and are not drawn. Synapse signs come from the male CNS neurotransmitter predictions, using the same rule as the Shiu model (GABA, glutamate and histamine inhibit, everything else excites).

**No boost needed.** Driving one lane's eye neurons at the plain model weights already makes all three output groups fire (about 1 to 2 spikes per neuron). The old boost of 8 now overdrives the network: Kenyon cell spikes jump from about 80 to about 2,300 and reward dopamine neurons fire with no reward. The boost is now 1.

**Bio-RL, seed 1, boost 1:** 53%, 75%, 94%, then 97 to 100% from song 4; 100% on the new song. With dopamine blocked it stays at 28 to 38% (33% on the new song).

**Deep-RL first failed, and the reason was the readout, not the brain.** With the old setup it sat at about 41% for 30 songs at every learning rate and pressed on almost every empty beat. To check whether the descending neurons carry lane information, I recorded their spike counts for 288 beats with learning off and trained a plain readout that is told the answer: 100% accuracy on held-out beats (the most common answer alone gives 30%), and 98% from the output neurons. So the information is there; hundreds of active inputs with large counts were swamping the softmax. Deep-RL now standardizes each input with running statistics, a standard step for this kind of readout.

**Deep-RL learning rate sweep on the male CNS** (seed 1, 30 songs, standardized inputs):

| Learning rate | Mean of songs 26 to 30 | New song |
|---|---|---|
| 0.03 | 99% | 97% |
| 0.01 | 98% | 95% |
| 0.003 | 96% | 95% |
| 0.001 | 91% | 90% |

I picked 0.03, the highest mean of songs 26 to 30, using the same rule as the original sweep.

**Male CNS results, JavaScript version before the eligibility fix** (5 repeats of 30 songs; 3 repeats of 150 songs for the last row; mean and spread across repeats):

| Learner | Before practice | Song 1 | End of practice | Runs reaching 80% (song) | New song | Presses on empty beats (last song) |
|---|---|---|---|---|---|---|
| Bio-RL, real wiring | 31.9% | 61.9% | 99.6% ± 0.3 | 5/5 (4, 5, 5, 3, 3) | 100.0% ± 0.0 | 0.0 / 16 |
| Bio-RL, shuffled wiring | 21.9% | 21.9% | 21.9% ± 0.0 | 0/5 | 26.7% ± 1.3 | 15.0 / 16 |
| Bio-RL, random network | 15.0% | 21.9% | 26.3% ± 14.3 | 0/5 | 25.6% ± 13.3 | 0.0 / 16 |
| Bio-RL, dopamine blocked | 30.0% | 32.5% | 35.3% ± 2.5 | 0/5 | 37.9% ± 5.5 | 0.0 / 16 |
| Deep-RL, real wiring | 23.1% | 49.4% | 98.9% ± 0.8 | 5/5 (5, 4, 4, 4, 4) | 97.9% ± 2.5 | 0.4 / 16 |
| Deep-RL, shuffled wiring | 23.1% | 29.4% | 80.0% ± 11.6 | 2/5 (19, 6) | 72.8% ± 6.2 | 6.0 / 16 |
| Deep-RL, random network | 23.1% | 78.1% | 100.0% ± 0.0 | 5/5 (3, 3, 3, 3, 3) | 100.0% ± 0.0 | 0.0 / 16 |
| Deep-RL, real wiring, 150 songs | 21.9% | 46.9% | 99.8% ± 0.3 | 3/3 (5, 4, 4) | 97.4% ± 2.1 | 0.3 / 16 |

**Why the brain-style learner fails on scrambled wiring.** I drove one lane's eye neurons at a time and counted spikes. In the real wiring each output group fires about 1 to 2 spikes per neuron and Kenyon cells stay sparse. In shuffled wiring, activity snowballs after the first input: intermediate and descending neurons fire thousands of spikes and output neurons up to 15 per neuron, so the fly presses the same lane on almost every beat (15 of 16 empty beats on the last song) and learning never gets started. In a random network the eyes' signal barely reaches the output neurons (0 to 1.4 spikes per neuron), so there is nothing to strengthen. Descending neurons still carry some lane-specific activity there, which is why the trained readout learns on a random network.

**This reverses the FlyWire result.** On FlyWire, shuffled wiring did as well as the real wiring for Bio-RL (100% vs 96%). On the male CNS, the real wiring's balance is what lets learning inside the brain work.

**Punishment dopamine fires a little on its own.** Over 5 songs with learning on, the 8 PPL1 neurons fired about 4 spikes in the 100 ms after a hit (in 66% of hits), about 73 after a mistake, and about 15 during the decision window before any feedback, all from ordinary network input. Reward dopamine (PAM) fired about 315 spikes after a hit. The stray punishment spikes are small next to the real signals, so learning still works.

**Nerve cord added.** 3,625 nerve cord neurons that get at least 100 synapses from the simulated descending neurons, plus the connections among them: 6,900 neurons, 990,102 connections and 6,419,481 synapses in total. Cord to brain connections are dropped and the wiring controls leave the cord untouched. Replaying Bio-RL on real, shuffled and random wiring, and Deep-RL on real wiring, with and without the cord gave identical key presses. With real wiring, 2,153 of the 3,625 cord neurons fire. The page still plays at 1.00× speed.

**The sections below describe the original FlyWire version.** Results from the male CNS rerun are recorded here once it finishes.

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

The brain panel shows the cell bodies of 139,662 neurons from the Janelia male CNS, a fly whose brain and nerve cord were imaged together. It is seen from above with the head at the top, so the nerve cord runs down the page, as it runs back along a real fly's body. FlyWire covers the brain only, which is why the drawing uses the male CNS.

The simulated neurons come from FlyWire. Each is drawn at the cell body of a male CNS neuron of the same type, on the same side where possible; 26 of the 3,371 have no matching type and aren't drawn. Only simulated neurons ever flash, in the ink colour, with green for reward dopamine (PAM) and magenta for punishment dopamine (PPL1).

## The RL problem, stated honestly

| | |
|---|---|
| Song | 48 slots, 400 ms apart. Each slot is a note in lane 1-3 or a rest (16 of 48 are rests). |
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

This is built to avoid a common problem with simple dopamine learning rules, where punishments wear down the useful synapses:
- The error shrinks to zero once a pathway predicts its reward.
- Only the chosen pathway and recently active KCs are eligible.
- The APL keeps KC codes sparse (about 20 of 2,597 KCs fire per lane).
- Weights are bounded.
- Misses change nothing.

### Agent B: Deep-RL (frozen connectome + trained readout)

Plasticity is off and the dopamine neurons get no input. A linear softmax policy reads log(1 + spike count) from the 313 descending neurons and is trained with REINFORCE and a running reward baseline, learning rate 0.002. PPO would add nothing on a one-step problem.

### Controls

| Control | What changes |
|---|---|
| shuffled | Within each role-to-role block (e.g. KC→MBON), postsynaptic partners are permuted. Every neuron keeps its per-block in- and out-degree and its synapse counts, and the architecture stays intact. |
| random | A generic network: same neurons, same number of connections, same weights, but random endpoints. |
| dopamine blocked | Agent A with the PAM and PPL1 drive set to zero. |

All the numbers above were set once, before the full experiment runs. They are recorded in `flyhero/agents.py` and in the lab notebook below.

## Results

Setup: 5 seeds × 30 training episodes on "Banana Drift" (32 notes, 16 rests), then one frozen play of the unseen song "Wing Hum". The raw console output is in `notes/experiments.log`; full curves are in `data/results.json`.

- **Before** is a frozen play before any learning.
- **Last 5** is the mean hit % over episodes 26-30.
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

For the 150-episode row, "Last 5" means episodes 146-150.

### What the controls support

**Headline (supported).** The Bio-RL fly goes from 40% to 96% of notes hit in 30 plays. Every seed passes 80% by play 3-4. It scores 96% on a song it never trained on. Blocking its dopamine neurons keeps it at 41%.

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

**Where the 40% starting point comes from.** Before learning, MBON groups 1 and 2 respond to every lane and group 3 barely responds. The untrained fly therefore mostly presses lanes 1-2, which lands about 40% of notes. That's a structural bias, not skill, and it's why the blocked-dopamine control sits at the same level.

## Known biological liberties

1. **Signed plasticity.** Real dopamine at KC→MBON synapses mostly *depresses* them. Here reward strengthens the chosen pathway.
2. **Broadcast dopamine.** All PAM or all PPL1 neurons are driven together, not compartment by compartment.
3. **Prediction computed in code.** The prediction V is computed from MBON spikes by the learner's code, standing in for MBON→dopamine feedback (roughly as in Bennett et al. 2021).
4. **Winner-take-all.** The action readout over 3 MBON groups is a modeling choice. The groups are dealt round-robin by KC input, fixed before any results were seen.
5. **KC→MBON gain of 8.** Without it the MBONs never fire from visual input (see the lab notebook).
6. **Visual input.** Lanes are mapped onto real visual projection neurons that synapse onto KCs. That visual input is a small minority of what the mushroom body receives (about 6.5k synapses versus about 427k from central neurons, mostly olfactory).
7. **Drawing positions come from a different fly.** Simulated FlyWire neurons (female brain) are drawn at the cell bodies of same-type neurons in the Janelia male CNS, so flash positions are approximate, and they mark cell bodies rather than the synapse-dense regions where neurons connect.

## Lab notebook

**Data.** I wanted the male CNS connectome, but neuPrint needs an auth token, so I used the public FlyWire v783 files. The annotations have positions for all 139,248 neurons. Right-hemisphere neurons have the larger x values, so x is mirrored for a front view.

**First activity probe** (one lane's visual neurons at 150 Hz for 240 ms):
- About 20 visual neurons and 17-25 KCs fire. The code is sparse, and the APL fires a few spikes.
- **At the plain Shiu weights the MBONs were completely silent**, so nothing could learn.
- A KC→MBON gain of 4 gave about 3 spikes per MBON in groups 1-2 but under 0.4 in group 3. I picked gain 8.
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

I picked 0.002 because it was the steadiest; a single-seed sweep is noisy. Agent B keeps pressing on 8-12 of the 16 rests, because it samples from a softmax.

**Browser check.** Turbo looked stuck in one screenshot. Measured properly, it runs about 0.6 episodes per second; the screenshot had simply caught it right after a restart.

**Nerve cord added.** I couldn't find a brain stem in the drawing. Flies don't have one; the equivalent is the neck connective and the ventral nerve cord, and FlyWire's brain-only data leaves them out. The Janelia male CNS annotations are public, carry a soma position for each neuron, and include FlyWire cell types, so the brain panel now draws that fly's whole CNS from above. 3,345 of the 3,371 simulated neurons are placed at a cell of the same type.

**UI simplified.** The first page was a dark, neon-style dashboard: WebGL glow, labeled brain regions, a schematic nerve cord, a story strip and sound. It was too busy. It's now a plain white page with native controls, a black-on-white game, and a grey-dot brain on a 2D canvas. An accessibility pass followed: AA text contrast, visible keyboard focus, labelled canvases, dashed lines for Deep-RL, and 44px touch targets. A final design pass gave it a microscopy look: ink on white, GFP green and magenta for reward and punishment dopamine (the colour-blind-safe pair microscopists use), one legible typeface, and a figure-legend paragraph under the brain. The last cleanup pass:

- **Font is self-hosted.** The page no longer needs the internet.
- **Dark mode follows the system setting.** The canvases redraw from the same colour tokens.
- **The chart uses shade and dashes instead of extra colours.** Shade shows the wiring and dashes show the learner.
- **A missing-data message** now explains how to serve the folder. The simulation and experiments are unchanged.

## Sources

- Dorkenwald et al. 2024, *Neuronal wiring diagram of an adult brain*, Nature. FlyWire connectome.
- Schlegel et al. 2024, *Whole-brain annotation and multi-connectome cell typing of Drosophila*, Nature. Matsliah et al. 2024, Nature, and Berg et al. 2025, bioRxiv. Annotations and positions (github.com/flyconnectome/flywire_annotations).
- Shiu et al. 2024, *A Drosophila computational brain model reveals sensorimotor processing*, Nature. LIF model and edge list (github.com/philshiu/Drosophila_brain_model).
- Berg et al. 2026, *Sexual dimorphism in the complete Drosophila male central nervous system connectome*, Cell 189, 5504-5526 (doi:10.1016/j.cell.2026.08.015; preprint bioRxiv 2025). Janelia male CNS v1.0 (CC-BY), soma positions for the brain panel.
- Bennett, Philippides & Nowotny 2021, *Learning with reinforcement prediction errors in a model of the Drosophila mushroom body*, Nature Communications.
- Caron, Ruta, Abbott & Axel 2013, *Random convergence of olfactory inputs in the Drosophila mushroom body*, Nature.
