// Brain panel: every FlyWire v783 neuron as a grey dot (front view); simulated neurons flash when they spike.
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
    this.rgb = n.role.map(r => (r === 'pam' ? rgb(colors.gfp) : r === 'ppl1' ? rgb(colors.magenta) : rgb(colors.ink)));
  }

  frame(net) {
    const g = this.ctx, t = net.t, ls = net.lastSpike;
    g.drawImage(this.bg, 0, 0);
    let spiking = 0;
    for (let i = 0; i < this.xy.length; i++) {
      const age = t - ls[i];
      if (age < 10) spiking++;
      if (age > 600 || this.xy[i][0] < 0) continue; // negative x: no matching cell type to draw at
      g.fillStyle = `rgba(${this.rgb[i]},${Math.exp(-age / 200)})`;
      g.fillRect(this.xy[i][0] - 1.5, this.xy[i][1] - 1.5, 3, 3);
    }
    const m = this.sub.meta;
    this.header.textContent = `${m.n_neurons.toLocaleString()} neurons simulated, ${m.n_synapses.toLocaleString()} synapses, ${spiking} spiking now`;
  }
}
