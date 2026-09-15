"""Extract the Fly Hero network and brain drawing from the Janelia/Google male CNS connectome (v1.0).

One fly supplies everything: the wiring, the sign of every synapse, the drawing, and each neuron's position.

Inputs (download into data/raw/ from https://male-cns.janelia.org/download/):
  mcns_annotations.feather       body-annotations-male-cns-v1.0-minconf-0.5.feather
  mcns_weights.feather           connectome-weights-male-cns-v1.0-minconf-0.5.feather
  mcns_neurotransmitters.feather body-neurotransmitters-male-cns-v1.0.feather
Outputs:
  data/subgraph.json    simulated neurons (role, lane/group, position) + signed synapse counts
  data/brain_points.bin Uint16 (x,y) cell body positions of every neuron, seen from above
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW, OUT = Path("data/raw"), Path("data")
VPN_MIN_SYN, MBON_MIN_SYN, HOP_MIN_SYN = 10, 500, 10
VNC_MIN_SYN = 100  # nerve cord neurons need this many synapses from the simulated descending neurons
# Same rule as the Shiu et al. 2024 model: GABA and glutamate inhibit, everything else excites.
# Histamine (fly photoreceptors) is inhibitory too; it did not occur in the FlyWire data that model used.
INHIBITORY = {"gaba", "glutamate", "histamine"}

a = pd.read_feather(RAW / "mcns_annotations.feather").drop_duplicates("bodyId").set_index("bodyId")

w = pd.read_feather(RAW / "mcns_weights.feather")
def column(*names):
    found = [c for c in w.columns if c.lower() in names]
    assert len(found) == 1, f"expected one of {names} in {list(w.columns)}"
    return found[0]
w = w.rename(columns={column("body_pre", "pre", "bodyid_pre", "pre_id"): "pre_id",
                      column("body_post", "post", "bodyid_post", "post_id"): "post_id",
                      column("weight", "syn", "count", "synapses"): "syn"})[["pre_id", "post_id", "syn"]]

nt = pd.read_feather(RAW / "mcns_neurotransmitters.feather", columns=["body", "consensus_nt"]).set_index("body").consensus_nt

R = a[a.somaSide == "R"]
kc = list(R[R["class"] == "Kenyon_Cell"].index)
vk = (w[w.pre_id.isin(a[a.superclass == "visual_projection"].index) & w.post_id.isin(kc)]
      .groupby("pre_id").syn.sum())
vpn = vk[vk >= VPN_MIN_SYN].sort_values(ascending=False, kind="stable")
apl = list(R[R.type.astype(str) == "APL"].index)
kcout = w[w.pre_id.isin(kc)].groupby("post_id").syn.sum()
mbon = kcout[kcout.index.isin(a[a["class"] == "MBON"].index)]
mbon = mbon[mbon >= MBON_MIN_SYN].sort_values(ascending=False, kind="stable")
pam = list(R[R.type.astype(str).str.startswith("PAM")].index)
ppl1 = list(R[R.type.astype(str).str.startswith("PPL1")].index)

# MBON -> (<=1 intermediate) -> descending neurons, every hop >= HOP_MIN_SYN synapses
dn_all = set(a[a.superclass == "descending_neuron"].index)
strong = w[w.syn >= HOP_MIN_SYN]
taken = set(vpn.index) | set(kc) | set(apl) | set(mbon.index) | set(pam) | set(ppl1)
mid = set(strong[strong.pre_id.isin(mbon.index)].post_id) - dn_all - taken
mid = set(strong[strong.pre_id.isin(mid) & strong.post_id.isin(dn_all)].pre_id)
dns = set(strong[strong.pre_id.isin(mid | set(mbon.index)) & strong.post_id.isin(dn_all)].post_id) - taken

# Nerve cord: the cord neurons that get the most input from the simulated descending neurons.
# They only receive from the brain circuit (cord -> brain connections are dropped), so learning is unaffected.
brain = taken | mid | dns
vnc_all = set(a[a.superclass.astype(str).str.startswith("vnc")].index) - brain
to_cord = w[w.pre_id.isin(dns) & w.post_id.isin(vnc_all)].groupby("post_id").syn.sum()
vnc = sorted(to_cord[to_cord >= VNC_MIN_SYN].index)

roles = [("vpn", list(vpn.index)), ("kc", kc), ("apl", apl), ("mbon", list(mbon.index)),
         ("pam", pam), ("ppl1", ppl1), ("mid", sorted(mid)), ("dn", sorted(dns)), ("vnc", vnc)]  # vnc last: brain indices unchanged
ids, role, tag = [], [], []
for r, members in roles:
    for i, rid in enumerate(members):
        ids.append(rid)
        role.append(r)
        # deterministic round-robin: VPNs -> 3 lanes, MBONs -> 3 action groups (both sorted by KC synapses)
        tag.append(i % 3 if r in ("vpn", "mbon") else -1)
index = {rid: i for i, rid in enumerate(ids)}
assert len(index) == len(ids), "neuron assigned to two roles"

e = w[w.pre_id.isin(index) & w.post_id.isin(index)]
cord = set(vnc)
e = e[~(e.pre_id.isin(cord) & ~e.post_id.isin(cord))]  # drop cord -> brain connections
sign = np.where(e.pre_id.map(nt).isin(INHIBITORY), -1, 1)
e = e.assign(pre=e.pre_id.map(index), post=e.post_id.map(index), w=e.syn * sign).sort_values(["pre", "post"])

# Drawing: every neuron's cell body, seen from above with the head at the top (x across the body, z along it).
# The right side has small x, so x is mirrored to put the fly's right on the viewer's right.
soma = a[a.somaLocation.notna() & a.superclass.notna() & (a.statusLabel != "Glia")]
xyz = np.stack(soma.somaLocation.to_numpy()).astype(float)
x0, x1, z0, z1 = xyz[:, 0].min(), xyz[:, 0].max(), xyz[:, 2].min(), xyz[:, 2].max()
nx = lambda v: (x1 - v) / (x1 - x0)
nz = lambda v: (v - z0) / (z1 - z0)
pts = np.stack([nx(xyz[:, 0]) * 65535, nz(xyz[:, 2]) * 65535], axis=1).round().astype("<u2")
OUT.mkdir(exist_ok=True)
pts.tofile(OUT / "brain_points.bin")

# Each simulated neuron is drawn at its own cell body. The few without a recorded soma are not drawn.
sx, sy = [], []
for rid in ids:
    loc = a.at[rid, "somaLocation"]
    if loc is None or (isinstance(loc, float) and np.isnan(loc)):
        sx.append(-1.0)
        sy.append(-1.0)
    else:
        sx.append(nx(float(loc[0])))
        sy.append(nz(float(loc[2])))

out = {
    "meta": {
        "source": "Janelia/Google male CNS connectome v1.0 (Berg et al. 2025), CC-BY; neuron model after Shiu et al. 2024",
        "positions": "own cell body locations in the same fly, seen from above",
        "aspect": float((z1 - z0) / (x1 - x0)),
        "n_unplaced": int(sum(v < 0 for v in sx)),
        "n_points": int(len(pts)),
        "counts": {r: len(members) for r, members in roles},
        "n_neurons": len(ids), "n_brain_neurons": len(ids) - len(vnc), "n_edges": int(len(e)), "n_synapses": int(e.syn.sum()),
        "n_inhibitory_edges": int((e.w < 0).sum()),
        "thresholds": {"vpn_min_syn": VPN_MIN_SYN, "mbon_min_syn": MBON_MIN_SYN, "hop_min_syn": HOP_MIN_SYN, "vnc_min_syn": VNC_MIN_SYN},
    },
    "neurons": {
        "role": role, "tag": tag,
        "type": [str(t) if isinstance(t, str) else "" for t in a.loc[ids].type],
        "x": [round(float(v), 4) for v in sx], "y": [round(float(v), 4) for v in sy],
    },
    "edges": {"pre": e.pre.tolist(), "post": e.post.tolist(), "w": e.w.astype(int).tolist()},
}
(OUT / "subgraph.json").write_text(json.dumps(out, separators=(",", ":")))
print(json.dumps(out["meta"], indent=1))
