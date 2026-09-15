// Songs, rewards and the millisecond-level episode runner shared by the page and Node experiments.
import { rng } from './sim.js';

export const SLOT_MS = 400, LEAD_MS = 200, SLOTS = 48;
export const WAIT = 3;
// relative to each slot start (hit line at +200 ms)
export const T_DRIVE_ON = 0, T_COUNT = 50, T_DECIDE = 240, T_LEARN = 340, T_HIT = 200;

export function makeSong(seed, name, slots = SLOTS) {
  const rand = rng(seed);
  const notes = new Int8Array(slots);
  for (let s = 0; s < slots; s++) notes[s] = rand() < 0.25 ? -1 : Math.floor(rand() * 3);
  return { name, notes };
}

export const SONGS = { train: makeSong(101, 'Banana Drift'), test: makeSong(202, 'Wing Hum (unseen)') };

export const hitTime = (slot) => LEAD_MS + slot * SLOT_MS + T_HIT;
export const songMs = (song) => LEAD_MS + song.notes.length * SLOT_MS;

export function reward(note, action) {
  if (note < 0) return action === WAIT ? 0 : -0.5;
  return action === note ? 1 : -1;
}

// One play of a song, advanced 1 ms at a time so the page can render in between.
export class Episode {
  constructor(net, agent, song, { learn = true } = {}) {
    Object.assign(this, { net, agent, song, learn });
    this.t = 0;
    this.hits = 0; this.notes = 0; this.wrong = 0; this.misses = 0; this.falsePresses = 0;
    this.events = []; // {slot, note, action, reward, t}
    net.resetState();
  }

  get done() { return this.t >= songMs(this.song); }
  get rate() { return this.notes ? this.hits / this.notes : 0; }

  step() {
    const { net, agent, song } = this;
    const local = this.t - LEAD_MS;
    if (local >= 0) {
      const slot = Math.floor(local / SLOT_MS), ph = local - slot * SLOT_MS, note = song.notes[slot];
      if (ph === T_DRIVE_ON && note >= 0) net.setDrive(net.laneVpn[note], 150);
      else if (ph === T_COUNT) net.resetCounts();
      else if (ph === T_DECIDE) {
        if (note >= 0) net.setDrive(net.laneVpn[note], 0);
        const action = agent.decide(net);
        const r = reward(note, action);
        if (note >= 0) { this.notes++; if (action === note) this.hits++; else if (action === WAIT) this.misses++; else this.wrong++; }
        else if (action !== WAIT) this.falsePresses++;
        this.events.push({ slot, note, action, reward: r, t: this.t });
        agent.reward(net, r);
        net.resetCounts();
      } else if (ph === T_LEARN) agent.afterDopamine(net, this.learn);
    }
    net.step();
    this.t++;
  }

  run() { while (!this.done) this.step(); return this; }
}
