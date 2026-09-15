// Brain panel: WebGL point cloud of real FlyWire neuron positions + live spikes of the simulated subgraph.
import { rng } from './sim.js';

const VS = `attribute vec2 a_pos; attribute vec3 a_col; uniform vec4 u_rect; uniform float u_size; varying vec3 v_col;
void main() { gl_Position = vec4(u_rect.xy + a_pos * u_rect.zw, 0.0, 1.0); gl_PointSize = u_size; v_col = a_col; }`;
const FS = `precision mediump float; varying vec3 v_col; uniform float u_alpha;
void main() { float d = length(gl_PointCoord - 0.5); if (d > 0.5) discard; gl_FragColor = vec4(v_col, u_alpha * (1.0 - 2.0 * d)); }`;

const FLASH = { pam: [0.35, 1, 0.45], ppl1: [1, 0.3, 0.25], other: [1, 0.93, 0.62] };
const GLOW = { vpn: '255,140,40', kc: '255,140,40', mbon: '255,140,40', dn: '255,140,40', pam: '80,255,120', ppl1: '255,70,60' };
const LABELS = [
  ['vpn', 'eyes · visual projection neurons'], ['kc', 'memory · Kenyon cells (mushroom body)'],
  ['pam', 'dopamine · reward (PAM)'], ['ppl1', 'dopamine · punishment (PPL1)'], ['mbon', 'output · MBONs'],
  ['dn', 'output · descending neurons → nerve cord'],
];

function shader(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}

export class Brain {
  constructor(glCanvas, overlay, header, points, sub) {
    Object.assign(this, { glCanvas, overlay, header, sub });
    const gl = this.gl = glCanvas.getContext('webgl', { alpha: false, antialias: false });
    const prog = gl.createProgram();
    gl.attachShader(prog, shader(gl, gl.VERTEX_SHADER, VS));
    gl.attachShader(prog, shader(gl, gl.FRAGMENT_SHADER, FS));
    gl.bindAttribLocation(prog, 0, 'a_pos');
    gl.linkProgram(prog);
    gl.useProgram(prog);
    this.loc = {
      pos: 0, col: gl.getAttribLocation(prog, 'a_col'), rect: gl.getUniformLocation(prog, 'u_rect'),
      size: gl.getUniformLocation(prog, 'u_size'), alpha: gl.getUniformLocation(prog, 'u_alpha'),
    };
    this.nBase = points.length / 2;
    this.baseBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.baseBuf);
    gl.bufferData(gl.ARRAY_BUFFER, points, gl.STATIC_DRAW);

    const n = sub.neurons, N = this.N = n.role.length;
    this.simBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.simBuf);
    gl.bufferData(gl.ARRAY_BUFFER, Float32Array.from({ length: 2 * N }, (_, k) => (k % 2 ? n.y : n.x)[k >> 1]), gl.STATIC_DRAW);
    this.col = new Float32Array(3 * N);
    this.colBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.colBuf);
    gl.bufferData(gl.ARRAY_BUFFER, this.col, gl.DYNAMIC_DRAW);
    this.flashCol = n.role.map(r => FLASH[r] || FLASH.other);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);

    this.roles = {};
    for (const r of Object.keys(GLOW)) {
      const ids = n.role.flatMap((x, i) => (x === r ? [i] : []));
      const cx = ids.reduce((s, i) => s + n.x[i], 0) / ids.length, cy = ids.reduce((s, i) => s + n.y[i], 0) / ids.length;
      const vx = ids.reduce((s, i) => s + (n.x[i] - cx) ** 2, 0) / ids.length, vy = ids.reduce((s, i) => s + (n.y[i] - cy) ** 2, 0) / ids.length;
      this.roles[r] = { ids, cx, cy, vx, vy, glow: 0 };
    }
    this.pulses = [];
    this.lastT = 0;
    this.lastWall = performance.now();
    this.resize();
    addEventListener('resize', () => this.resize());
  }

  resize() {
    const dpr = this.dpr = devicePixelRatio || 1;
    const W = this.W = this.glCanvas.clientWidth, H = this.H = this.glCanvas.clientHeight;
    for (const c of [this.glCanvas, this.overlay]) { c.width = W * dpr; c.height = H * dpr; }
    const aspect = this.sub.meta.aspect;
    const bw = Math.min(W - 16, (H * 0.5) / aspect), bh = bw * aspect;
    this.rect = { bx: (W - bw) / 2, by: 24, bw, bh };
    this.ctx = this.overlay.getContext('2d');
    this.buildNerveCord();
  }

  px(nx, ny) { return [this.rect.bx + nx * this.rect.bw, this.rect.by + ny * this.rect.bh]; }

  // Schematic only: FlyWire v783 is brain-only, so there are no real nerve-cord positions to draw.
  buildNerveCord() {
    const { W, H, dpr, rect } = this;
    const c = this.vnc = document.createElement('canvas');
    c.width = W * dpr; c.height = H * dpr;
    const g = c.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const cx = rect.bx + rect.bw / 2, top = rect.by + rect.bh * 0.97;
    const s = this.vncScale = Math.max(0.4, Math.min(1.2, (H - top - 30) / 330));
    this.vncTop = top;
    this.vncLen = 320 * s;
    const blobs = [[0, 22, 6, 24], [0, 75, 34, 30], [0, 140, 42, 34], [0, 202, 36, 30], [0, 275, 20, 48]];
    const rand = rng(99);
    g.fillStyle = 'rgba(120,135,170,0.35)';
    for (const [ox, oy, rx, ry] of blobs) {
      const n = Math.round(rx * ry * s * s * 0.9);
      for (let k = 0; k < n; k++) {
        const a = rand() * 2 * Math.PI, r = Math.sqrt(rand());
        g.fillRect(cx + (ox + Math.cos(a) * r * rx) * s, top + (oy + Math.sin(a) * r * ry) * s, 1, 1);
      }
    }
    g.fillStyle = '#666';
    g.font = '11px ui-monospace, monospace';
    g.fillText('nerve cord · schematic', cx + 50 * s, top + 130 * s);
    g.fillText('(not in FlyWire brain data)', cx + 50 * s, top + 145 * s);
  }

  frame(net) {
    const now = performance.now(), wallDt = Math.min(100, now - this.lastWall);
    this.lastWall = now;
    const t = net.t, ls = net.lastSpike, { gl, col, N, dpr, W, H, rect } = this;
    if (t < this.lastT) this.lastT = 0; // a new episode reset the clock
    let K = 0;
    for (let i = 0; i < N; i++) {
      const age = t - ls[i];
      if (age < 10) K++;
      const b = age < 1200 ? Math.exp(-age / 200) : 0, f = this.flashCol[i];
      col[3 * i] = f[0] * b; col[3 * i + 1] = f[1] * b; col[3 * i + 2] = f[2] * b;
    }
    const k = 1 - Math.exp(-wallDt / 250);
    for (const [r, R] of Object.entries(this.roles)) {
      let recent = 0;
      for (const i of R.ids) if (t - ls[i] < 40) recent++;
      R.glow += (Math.min(1, recent / Math.max(3, 0.02 * R.ids.length)) - R.glow) * k;
    }
    let dnNew = 0;
    for (const i of this.roles.dn.ids) if (ls[i] >= this.lastT && ls[i] < t) dnNew++;
    if (dnNew > 0 && (!this.pulses.length || this.pulses.at(-1).age > 80)) this.pulses.push({ age: 0, a: Math.min(1, dnNew / 8) });
    this.lastT = t;
    this.header.textContent = `${this.sub.meta.n_neurons.toLocaleString()} neurons simulated · ${this.sub.meta.n_synapses.toLocaleString()} synapses · ${K} spiking now`;

    // WebGL points
    gl.viewport(0, 0, W * dpr, H * dpr);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.uniform4f(this.loc.rect, (rect.bx / W) * 2 - 1, 1 - (rect.by / H) * 2, (rect.bw / W) * 2, (-rect.bh / H) * 2);
    gl.enableVertexAttribArray(0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.baseBuf);
    gl.vertexAttribPointer(0, 2, gl.UNSIGNED_SHORT, true, 0, 0);
    gl.disableVertexAttribArray(this.loc.col);
    gl.vertexAttrib3f(this.loc.col, 0.22, 0.27, 0.4);
    gl.uniform1f(this.loc.size, 1.6 * dpr);
    gl.uniform1f(this.loc.alpha, 0.4);
    gl.drawArrays(gl.POINTS, 0, this.nBase);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.simBuf);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.colBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, col);
    gl.enableVertexAttribArray(this.loc.col);
    gl.vertexAttribPointer(this.loc.col, 3, gl.FLOAT, false, 0, 0);
    gl.uniform1f(this.loc.size, 4.5 * dpr);
    gl.uniform1f(this.loc.alpha, 1);
    gl.drawArrays(gl.POINTS, 0, N);

    // 2D overlay: activity glow, nerve cord schematic, labels
    const g = this.ctx;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    g.globalCompositeOperation = 'lighter';
    for (const [r, R] of Object.entries(this.roles)) {
      if (R.glow < 0.02) continue;
      const [x, y] = this.px(R.cx, R.cy);
      const rad = Math.max(26, Math.sqrt(R.vx * rect.bw ** 2 + R.vy * rect.bh ** 2) * 1.4);
      const grad = g.createRadialGradient(x, y, 0, x, y, rad);
      grad.addColorStop(0, `rgba(${GLOW[r]},${0.4 * R.glow})`);
      grad.addColorStop(1, `rgba(${GLOW[r]},0)`);
      g.fillStyle = grad;
      g.fillRect(x - rad, y - rad, 2 * rad, 2 * rad);
    }
    g.drawImage(this.vnc, 0, 0, W, H);
    const cx = rect.bx + rect.bw / 2;
    for (const p of this.pulses) {
      p.age += wallDt;
      const y = this.vncTop + (p.age / 600) * this.vncLen, rad = 14 * this.vncScale;
      const grad = g.createRadialGradient(cx, y, 0, cx, y, rad);
      grad.addColorStop(0, `rgba(255,236,160,${0.8 * p.a * (1 - p.age / 600)})`);
      grad.addColorStop(1, 'rgba(255,236,160,0)');
      g.fillStyle = grad;
      g.fillRect(cx - rad, y - rad, 2 * rad, 2 * rad);
    }
    this.pulses = this.pulses.filter(p => p.age < 600);
    g.globalCompositeOperation = 'source-over';
    this.drawLabels(g);
  }

  drawLabels(g) {
    g.font = '11px ui-monospace, monospace';
    const items = LABELS.map(([r, text]) => ({ text, xy: this.px(this.roles[r].cx, this.roles[r].cy) }))
      .sort((a, b) => a.xy[1] - b.xy[1]);
    let y = 18;
    for (const it of items) {
      y = Math.max(y + 16, it.xy[1]);
      const tx = this.W - 8 - g.measureText(it.text).width;
      g.strokeStyle = 'rgba(200,200,200,0.25)';
      g.beginPath(); g.moveTo(it.xy[0], it.xy[1]); g.lineTo(tx - 4, y - 4); g.stroke();
      g.fillStyle = 'rgba(0,0,0,0.6)';
      g.fillRect(tx - 2, y - 12, this.W - tx, 15);
      g.fillStyle = '#bbb';
      g.fillText(it.text, tx, y);
    }
  }
}
