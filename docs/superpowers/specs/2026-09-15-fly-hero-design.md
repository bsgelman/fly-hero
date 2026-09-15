# Fly Hero: design spec

Date: 2026-09-15 · Status: approved for implementation (Ben pre-approved every stage and asked for unattended completion) · Debate: `docs/party-mode-debate.md`

## 1. What it is

A single browser page. On the left is a 3-lane rhythm game; on the right is a simulated fruit-fly brain. A spiking model built from the real FlyWire connectome learns to play the game by trial and error. There are two learners:

- **Agent A, Bio-RL.** Learning happens inside the connectome, through dopamine-gated plasticity on Kenyon cell (KC) → mushroom body output neuron (MBON) synapses.
- **Agent B, Deep-RL.** The connectome is frozen, and REINFORCE trains a linear readout from descending-neuron spikes.

Both agents run against controls. The README lab notebook reports whatever the controls show.

Three modes:
1. **Watch the fly learn.** Episodes run back to back, with a live hit-rate curve and speed options 1× / 4× / turbo.
2. **You vs fly.** A human plays the same song with keys J/K/L and the hit % is compared.
3. **Compare learners.** Learning curves from `data/results.json`, produced by `node tools/run_experiments.mjs` with the exact same JS modules.

## 2. RL framing (stated honestly)

- **Time.** The song is a grid of 48 slots, 400 ms apart. Each slot holds a note in lane 0/1/2 or a rest (25% rests, no chords).
- **State.** What the fly's visual input shows as each slot approaches: lane 0, lane 1, lane 2, or nothing. This is sight-reading, not song position.
- **Actions.** Press lane 0/1/2, or wait.
- **Reward:**

  | Outcome | Reward |
  |---|---|
  | Correct press on a note | +1 |
  | Wrong lane | −1 |
  | Wait on a note (miss) | −1 |
  | Press on a rest | −0.5 |
  | Wait on a rest | 0 |

- **Episode.** One play of the song.
- **What kind of problem this really is.** Actions don't change future states, so this is a contextual bandit. The learning rule is reward-prediction-error learning with eligibility traces. It is not multi-step TD, and the docs must not claim otherwise.
- **Generalization test.** Train on song A, then freeze learning and play unseen song B.

## 3. Data (verified 2026-09-15)

- **Male CNS rejected for now.** It needs a neuPrint auth token, which can't be obtained unattended. It can be swapped in later.
- **FlyWire v783 edges.** `Connectivity_783.parquet` from github.com/philshiu/Drosophila_brain_model: 15.09M connections, with signed synapse counts (`Excitatory` ±1).
- **Positions.** `Supplemental_file1_neuron_annotations.tsv` from github.com/flyconnectome/flywire_annotations has `pos_x/pos_y/pos_z` for 139,248 neurons, in FAFB voxel units. The front view uses (x, y), mirrored in x so the fly's right hemisphere sits on the viewer's left.
- **No nerve cord.** FlyWire covers the brain only, so the nerve cord is drawn as a **labeled schematic**.
- **Extraction.** `tools/extract_subgraph.py` writes `data/subgraph.json` and `data/brain_points.bin`.

## 4. Simulated subgraph (right hemisphere mushroom-body loop)

Selection rules are fixed in the script and deterministic.

| Role | Rule | Count (from extraction) |
|---|---|---|
| vpn (eyes) | right-side visual projection neurons with ≥10 synapses onto right KCs; dealt round-robin into 3 lanes by synapse rank | 58 |
| kc (memory) | all right KCs | 2,597 |
| apl | the right APL (real feedback inhibition) | 1 |
| mbon (output) | MBONs with ≥500 synapses from right KCs; dealt round-robin into 3 action groups | 35 |
| pam (reward DANs) | right PAM | 153 |
| ppl1 (punishment DANs) | right PPL1 | 8 |
| mid | neurons with ≥10 synapses from an MBON and ≥10 onto a DN | 206 |
| dn (descending) | DNs with ≥10 synapses from an MBON or mid | 313 |

All connectome edges between these neurons are kept. Total: **3,371 neurons, 233,686 connections, 650,482 synapses**. The UI shows these numbers read from `meta`.

## 5. Neuron model

This is the Shiu et al. 2024 LIF model:

- **Membrane.** v₀ = v_reset = −52 mV, v_th = −45 mV, τ_m = 20 ms.
- **Synapses.** τ_syn = 5 ms, refractory period 2.2 ms, delay 1.8 ms, w_syn = 0.275 mV per synapse × sign.
- **Poisson input.** 150 Hz, with a kick of 250 × w_syn.

Our deviations:
- **Time step.** dt = 1 ms with exact exponential decay (the original uses a finer brian2 step). Delay rounds to 2 ms, refractory period to 2 ms.
- **KC→MBON gain.** One global gain on KC→MBON synapses, calibrated once so episode 1 presses on most notes. The value is documented.
- **Background noise.** Weak Poisson input on MBONs, which is the only source of exploration. The rate is documented.

## 6. Timing

Times are relative to each slot's hit time t. At 1× speed, simulation time equals game time.

| Window | What happens |
|---|---|
| t − 200 → t + 40 ms | the lane's VPNs get 150 Hz Poisson drive |
| t − 150 → t + 40 ms | decision window: MBON spikes are counted per action group |
| t + 40 ms | the press registers: the group with the most spikes per neuron, if it reaches θ_press; otherwise wait |
| t + 40 → t + 140 ms | reward phase: DANs are driven and plasticity is applied at the end |
| — | the next slot's input starts at t + 200 |

- **Humans** get a ±150 ms window on the same clock, so the fly's +40 ms press is inside the human window.
- **Notes** fall for 1.6 s before reaching the hit line.

## 7. Agent A: Bio-RL rule

| Symbol | Meaning |
|---|---|
| x_k | trace of KC k's spikes (+1 per spike, τ_e = 200 ms) |
| g | chosen action group |
| c_g | chosen group's spikes per neuron in the decision window |
| V | prediction: tanh(c_g / c_ref) |
| r | reward from §2 |
| δ | reward-prediction error: r − V |

1. **Dopamine is simulated.** For 100 ms, PAM neurons get Poisson drive at rate ∝ max(δ, 0) and PPL1 neurons at rate ∝ max(−δ, 0).
2. **D is measured from spikes:** D = PAM spikes / (N_pam · s_max) − PPL1 spikes / (N_ppl1 · s_max).
3. **Update.** For every edge from a KC to an MBON in group g: m ← clip(m + η · D · x_k, 0, m_max). The edge's effective weight is w · gain · m.
4. **No press, no eligibility.** A wait creates no eligibility, so no weights change. The dopamine neurons still fire and are shown.
5. **Dopamine blocked** zeroes the DAN drive, so D = 0.

How this avoids FlyPong's failure: the RPE saturates, eligibility is limited to the chosen pathway and recently active KCs, the APL keeps KC codes sparse, weights are bounded, and misses don't depress anything.

**Known biological liberties (README must list):**
1. The plasticity is signed. In flies, dopamine mainly depresses KC→MBON synapses.
2. Dopamine is broadcast, not compartment-specific.
3. V is read from the chosen group's MBON spikes in JS, standing in for MBON→DAN feedback.
4. The winner-take-all action readout is a modeling choice.

## 8. Agent B: Deep-RL baseline

- **Connectome.** Frozen: m = 1, no dopamine drive.
- **Features.** f = log(1 + DN spike counts) in the decision window.
- **Policy.** softmax(W f + b) over the 4 actions.
- **Learning.** REINFORCE with a running-mean reward baseline, one update per slot. There is no multi-step credit, so PPO is unnecessary. It runs in JS on the same simulator.
- **Learning rate.** 0.002, chosen from a seed-1 sweep over {0.1, 0.01, 0.005, 0.002} (README lab notebook).

## 9. Controls and experiment

Wiring variants are built in JS from the real edge list, with a fixed seed:

- **real**
- **shuffled.** Class-preserving shuffle: within each (pre role → post role) block, postsynaptic endpoints are permuted. This keeps every neuron's out-degree and in-degree within the block, plus the synapse-count distribution.
- **random.** A generic network with the same neuron count, edge count and weight multiset, and uniformly random endpoints, excluding self-loops. Input and output indices stay the same.

Conditions:

| Condition | Agent | Wiring | Dopamine |
|---|---|---|---|
| A-real | A | real | on |
| A-shuffled | A | shuffled | on |
| A-random | A | random | on |
| A-dopamine-blocked | A | real | blocked |
| B-real | B | real | — |
| B-shuffled | B | shuffled | — |
| B-random | B | random | — |

Each condition runs 5 seeds × 30 training episodes on song A, then 1 frozen test episode on song B. Before training, each run also plays one frozen **before** episode, because episode 1 already includes 48 slots of within-episode learning.

Fairness check: **B-real-150ep** gives Deep-RL 5× the training budget (150 episodes, 3 seeds).

**Metrics:**
- Note hit % per episode (hits / notes).
- False presses on rests.
- Episodes to reach 80%, measured on a 3-episode moving average.
- Mean of the last 5 episodes.
- Frozen hit % on the unseen song.

**Headline** (only if the controls support it): "episode-1 X% → Y% by episode N, Z% on an unseen song; dopamine blocked stays at W%". It must also state plainly whether real wiring beat shuffled.

## 10. Brain panel

- **Base cloud.** WebGL `gl.POINTS`: all 139,248 annotated neuron positions (Uint16, 557 KB), dim blue-grey, additive blending.
- **Simulated neurons.** A second draw of the 3,371 simulated neurons, with brightness = exp(−Δt / 200 ms) since each neuron's last spike. White/yellow, except PAM green and PPL1 red. Unsimulated neurons never flash.
- **Region glow.** A 2D canvas overlay per role: an EMA of population rate drives an orange radial glow at the role's centroid.
- **Labels.** eyes (vpn) · memory (KCs) · dopamine (PAM/PPL1) · output (MBON → DN).
- **Nerve cord.** A schematic below the brain, labeled "nerve cord: schematic, not in FlyWire data". DN spikes send a pulse down it; this is visual only and labeled.
- **Header.** "N neurons simulated · M synapses · K spiking now (last 10 ms)", computed from live state.

## 11. Game panel and style

- **Tech.** A single `index.html` with minimal CSS and plain ES modules. No frameworks, no build step, served by any static server.
- **Game canvas.** 3 lanes, a hit line, falling notes, a hit/miss counter, a song progress bar and the episode number.
- **Audio.** WebAudio: a tone per lane on a hit, a soft click on each beat.
- **Charts.** Hand-drawn canvas line charts.

## 12. Files

```
index.html            layout, CSS, mode tabs
js/sim.js             Network (CSR, LIF, step, drive), wiring variants, PRNG
js/agents.js          BioRL, DeepRL: decide(), learn()
js/game.js            songs, slot runner (headless episode), scoring
js/brain.js           WebGL brain panel
js/app.js             UI: modes, game canvas, charts, live loop
tools/extract_subgraph.py
tools/run_experiments.mjs   writes data/results.json
tools/test.mjs              assert-based checks (node tools/test.mjs)
data/subgraph.json, data/brain_points.bin, data/results.json
README.md             how to run + lab notebook
```

## 13. Testing

`node tools/test.mjs` runs assert-based checks:
- A neuron with a constant supra-threshold drive spikes. With no drive, it stays silent.
- Inhibitory weights lower v.
- The shuffle preserves degrees and weight multisets.
- Scoring: a press at +40 ms hits, a wrong lane is −1, a press on a rest is −0.5.
- Agent A on the real wiring (seed 7) improves: the mean of episodes 8–10 is ≥25 points above the frozen before-episode, and plastic weights changed. This is a smoke test, not the headline.

The browser page is checked by loading it and confirming there are no console errors and the frames render.

## 14. Out of scope

Male CNS, compartment-specific dopamine, a real nerve-cord neuron model, PPO, chords or hold notes, mobile layout polish.

## 15. Revision after Ben's feedback (UI only)

Ben found the first UI too busy and too obviously AI-made, and asked for something far more minimal. This replaces §10–§11. The simulation, agents, controls and experiments are unchanged.

- **Page:** plain white page with default fonts and native buttons and selects. One `fly` menu covers the 7 conditions, a `speed` menu offers 1× / 4× / max, and a native `<progress>` shows song progress. No sound, no story strip.
- **Game:** black note bars on white, 3 lane lines, a hit line and J K L labels. "fly" marks the fly's press.
- **Brain:** one 2D canvas, no WebGL. The 139,248 neurons are pale grey dots, drawn once to an offscreen canvas. Simulated neurons flash black; PAM flashes green and PPL1 red. The header keeps the true counts.
- **Removed:** glow, region labels and the schematic nerve cord, replaced by a one-line legend.
- **Compare:** plain mean lines and a short table (before, last 5, unseen song).
