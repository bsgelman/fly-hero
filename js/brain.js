// Brain panel: every FlyWire v783 neuron as a grey dot (front view); simulated neurons flash when they spike.
import { T_DECIDE, T_LEARN } from './game.js';

const rgb = (hex) => [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16)).join(',');

export class Brain {
  constructor(canvas, header, points, sub, colors) {
    Object.assign(this, { canvas, header, sub });
    const W = canvas.width, H = (canvas.height = Math.round(W * sub.meta.aspect));
    this.ctx = canvas.getContext('2d');
    // the 139k-point cloud never changes, so draw it once
    const bg = (this.bg = document.createElement('canvas'));
    bg.width = W;
    bg.height = H;
    const g = bg.getContext('2d');
    g.fillStyle = colors.paper;
    g.fillRect(0, 0, W, H);
    g.fillStyle = `rgba(${rgb(colors.ink)},0.09)`;
    for (let k = 0; k < points.length; k += 2) g.fillRect((points[k] / 65535) * W, (points[k + 1] / 65535) * H, 1, 1);
    const n = sub.neurons;
    this.xy = n.x.map((x, i) => [x * W, n.y[i] * H]);
    this.ink = rgb(colors.ink);
    this.dopa = { pam: rgb(colors.gfp), ppl1: rgb(colors.magenta) };
    this.role = n.role;
    this.rgb = n.role.map(() => this.ink);
    this.seenT = 0;
  }

  frame(net, ep) {
    const g = this.ctx, t = net.t, ls = net.lastSpike;
    g.drawImage(this.bg, 0, 0);
    // Dopamine neurons show green or magenta only for spikes during the feedback after a press, and only for the
    // population being signalled. Their background spikes at other times flash in ink like every other neuron.
    const ev = ep.events.at(-1), d = ep.agent.delta;
    const signalled = ev && ep.agent.kind === 'bio' && !ep.agent.o.blocked && d ? (d > 0 ? 'pam' : 'ppl1') : null;
    if (t < this.seenT) this.seenT = 0; // new song
    let spiking = 0;
    for (let i = 0; i < this.xy.length; i++) {
      const age = t - ls[i];
      if (ls[i] >= this.seenT && this.dopa[this.role[i]]) {
        const inFeedback = signalled === this.role[i] && ls[i] >= ev.t && ls[i] < ev.t + T_LEARN - T_DECIDE;
        this.rgb[i] = inFeedback ? this.dopa[this.role[i]] : this.ink;
      }
      if (age < 10) spiking++;
      if (age > 600 || this.xy[i][0] < 0) continue; // negative x: no matching cell type to draw at
      g.fillStyle = `rgba(${this.rgb[i]},${Math.exp(-age / 200)})`;
      g.fillRect(this.xy[i][0] - 1.5, this.xy[i][1] - 1.5, 3, 3);
    }
    this.seenT = t;
    const m = this.sub.meta;
    this.header.textContent = `${m.n_neurons.toLocaleString()} neurons simulated, ${m.n_synapses.toLocaleString()} synapses, ${spiking} spiking now`;
  }
}
