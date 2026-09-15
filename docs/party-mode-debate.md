# Fly Hero: party-mode design debate (2026-09-15)

Run inline (session mode, non-interactive: Ben had stepped away). The room was the standard BMAD cast plus two walk-on experts. Decisions that came out of it are in the spec's decision log. Disagreements that stayed open are kept here as they happened.

Cast: 📊 Mary (analyst) · 🏗️ Winston (architect) · 💻 Amelia (dev) · 📋 John (PM) · 🎨 Sally (UX) · 📚 Paige (tech writer) · 🔬 Nadia (mushroom-body neuroscientist, walk-on) · 🎲 Ravi (RL researcher, walk-on)

---

## Round 1: what the fly sees (state encoding)

📋 **John:** Ben wants a brag. "Where am I in the song" is a lookup table. The fly memorizes 48 slots and we call it learning? Hard no.

🎲 **Ravi:** Agreed, and it's worse than that. Song position means a 48-state tabular problem where every state is seen once per episode. That's *slower* to learn than lane encoding, which has 4 states (lane 1/2/3/empty) seen ~12 times each per episode.

🔬 **Nadia:** And it's biologically backwards. Nothing in the fly brain encodes "bar 17 of the song". Visual projection neurons really do synapse onto Kenyon cells. I checked the FlyWire table: 140 VPNs, about 6.5k synapses onto right-side KCs, mostly aMe12, MTe30 and LTe25 onto KCγ-d and KCαβ-p.

📊 **Mary:** Caveat: 6.5k visual synapses against ~427k from central neurons (olfactory PNs mostly). Vision is a minority input to the MB. Only ~211 of the 2,597 right KCs get any visual input at all.

🔬 **Nadia:** That's the *real* number, and it's enough. Sparse is the point.

🎲 **Ravi:** Lane encoding also gives us a free test: train on song A, freeze, play unseen song B. If it's really a stimulus→action policy, B scores the same. Memorization can't pass that.

📋 **John:** "Sight-reads a song it never heard" is the headline. Done.

🏗️ **Winston:** Not *quite* done. Lane encoding makes the task a contextual bandit, not a temporal problem. Nobody should call it "temporal-difference learning" with a straight face.

🎲 **Ravi:** …fine. It's reward-prediction-error learning with eligibility traces on a one-step problem. Paige, write that down so nobody oversells it.

📚 **Paige:** Written. Verbatim.

## Round 2: which brain, which neurons

💻 **Amelia:** Male CNS needs a neuPrint token. Ben is asleep. I can't script an auth flow. FlyWire v783 edges are public in the Shiu repo (`Connectivity_783.parquet`) and positions are public in `flywire_annotations` (`pos_x/pos_y/pos_z`). Verified, downloaded, parsed.

🎨 **Sally:** But the brief says "nerve cord drawn below". FlyWire is brain-only.

💻 **Amelia:** So we draw the nerve cord as a schematic and *label it schematic*. No fake neuron positions.

🎨 **Sally:** I hate it a little. I'll make the label small but honest.

🏗️ **Winston:** Subgraph proposal, right hemisphere only: 58 VPNs (≥10 syn onto KCs) → 2,597 KCs + the real APL → 35 MBONs (≥500 KC syn) → ≤2 hops (≥10 syn per hop) → 313 descending neurons, plus right PAM and PPL1 DANs. That's ~3.4k neurons and ~235k connections.

💻 **Amelia:** Event-driven LIF in typed arrays handles that in JS. I estimate under a second per episode headless.

🔬 **Nadia:** Keep the APL. It's the real inhibitory neuron that makes KC codes sparse, and sparseness is exactly what saves you from FlyPong's fate.

📊 **Mary:** How do lanes map to VPNs? Whoever picks that mapping by hand can accidentally stack the deck.

🏗️ **Winston:** Deterministic: sort VPNs by KC synapse count, deal them round-robin into 3 lanes. Same for MBONs into 3 action groups. No cherry-picking, documented in the script.

## Round 3: the learning rule (the big fight)

🎲 **Ravi:** FlyPong failed because plain three-factor Hebbian learning never stops. Every punishment keeps depressing, and misses outnumber hits early. RPE fixes the "never stops" part: δ = r − V, and once V ≈ r, δ ≈ 0.

🔬 **Nadia:** Where does V come from? If you compute it in JavaScript from weights, that's a hidden critic.

🎲 **Ravi:** From the MBON spikes of the chosen action group. That's roughly Bennett et al. 2021: MBON output feeds back as the prediction the DANs compare against.

🔬 **Nadia:** Then sign convention. In real flies, dopamine mostly *depresses* KC→MBON synapses. You're going to potentiate the rewarded pathway.

🎲 **Ravi:** Yes, signed plasticity. It's a modeling simplification and we say so in the README.

🔬 **Nadia:** Noted, and I'm not letting it go. *(It stays open. See spec, "Known biological liberties".)*

🏗️ **Winston:** And the other half of FlyPong's failure?

🎲 **Ravi:** Three guards:
1. Eligibility only on synapses onto the **chosen** action group, from KCs active in the last ~200 ms.
2. **Misses create no eligibility**, so a wait-miss flashes PPL1 but changes no weights.
3. Weights are bounded [0, max]. Depression hits a floor, so it can't run away.

📊 **Mary:** Guard 2 sets a trap. If every pathway starts silent, the fly always waits, never presses, and never learns.

🎲 **Ravi:** Exploration comes from spiking noise, and the initial gain is calibrated so episode 1 presses on most notes with roughly random lanes. That calibration goes in the lab notebook, not buried.

💻 **Amelia:** I want the dopamine to be *real simulated spikes*: drive PAM or PPL1 with Poisson input ∝ |δ|, count their spikes, and that count gates the update. Blocking dopamine then literally silences those neurons.

🔬 **Nadia:** Broadcast dopamine, not compartment-specific. Another liberty. Write it down.

📚 **Paige:** Liberty #2, written down.

## Round 4: Deep-RL baseline and the controls

📋 **John:** PPO in Python?

🎲 **Ravi:** Overkill. It's one-step, so PPO's clipping and GAE buy nothing. REINFORCE with a baseline on a linear softmax readout of DN spike counts, in the browser, on the *same* simulator. Fair fight, no second codebase.

🏗️ **Winston:** Controls:
- **Class-preserving shuffle.** Rewire within each (pre-class → post-class) block, keeping degrees and synapse counts.
- **Generic random network.** Same size, same edge count, same weights, random endpoints.
- **Dopamine blocked** (A only).

📊 **Mary:** Prediction on record: the class-preserving shuffle ties the real connectome for Agent A. Caron et al. 2013 showed PN→KC wiring is close to random anyway. The *architecture* (expansion, sparseness, dopamine-gated output) does the work, not the specific wiring.

📋 **John:** That kills my brag.

📊 **Mary:** It kills a *false* brag. The true brag is "learns from dopamine alone, sight-reads an unseen song, and dopamine block drops it to chance".

🎲 **Ravi:** And the generic random network probably fails for *both* agents, because random wiring doesn't route lane information to the outputs. That's a real "architecture matters" result.

💻 **Amelia:** My prediction: Deep-RL's DN readout gets weaker signal than MBON readout, because it's two hops downstream. Could fail outright.

🎲 **Ravi:** Then that's the finding. We report what the controls show.

## Round 5: timing, difficulty, MVP

🎨 **Sally:** A 400 ms note grid. Notes fall for 1.6 s. Humans get ±150 ms.

💻 **Amelia:** The fly's VPNs start firing 200 ms before the hit line. The decision window runs from −150 to +40 ms, and the press registers at +40 ms, inside the human window. Dopamine fires from +40 to +140, and eligibility decays with τ = 200 ms. Sim time equals game time at 1× speed.

🎨 **Sally:** Difficulty: 48 slots, 25% rests, no chords. A human first-timer should land around 70-90%, so a fly at 90%+ is a real comparison.

📋 **John:** Headline metric: note hit % in episode 1 → mean of the last 5 episodes, over 5 seeds, plus hit % on the unseen song with learning frozen. Episodes-to-80% as the efficiency number.

🏗️ **Winston:** Weekend MVP order: extract → sim → Agent A headless learns → game page → brain panel → Agent B + controls → compare chart → README. If A doesn't learn headless, nothing visual matters.

🔬 **Nadia:** And if A doesn't learn at all?

📋 **John:** Then the README says so. *(Everyone looks at Paige.)*

📚 **Paige:** It's in the non-negotiables. I'm not editing it out.
