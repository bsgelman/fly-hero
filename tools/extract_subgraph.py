"""Extract the Fly Hero subgraph + brain point cloud from public FlyWire v783 data.

Inputs (download into data/raw/, see README):
  Connectivity_783.parquet  - github.com/philshiu/Drosophila_brain_model
  annotations.tsv           - github.com/flyconnectome/flywire_annotations
                              supplemental_files/Supplemental_file1_neuron_annotations.tsv
  mcns_annotations.feather  - storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/
                              body-annotations-male-cns-v1.0-minconf-0.5.feather (soma positions, CC-BY)
Outputs:
  data/subgraph.json    simulated neurons (role, lane/group, drawing position) + signed edges
  data/brain_points.bin Uint16 (x,y) soma positions of male CNS neurons, seen from above
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW, OUT = Path("data/raw"), Path("data")
VPN_MIN_SYN, MBON_MIN_SYN, HOP_MIN_SYN = 10, 500, 10

a = pd.read_csv(RAW / "annotations.tsv", sep="\t", low_memory=False).set_index("root_id")
c = pd.read_parquet(RAW / "Connectivity_783.parquet",
                    columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity", "Excitatory"])
R = a[a.side == "right"]

kc = list(R[R.cell_class == "Kenyon_Cell"].index)
vk = (c[c.Presynaptic_ID.isin(a[a.super_class == "visual_projection"].index) & c.Postsynaptic_ID.isin(kc)]
      .groupby("Presynaptic_ID").Connectivity.sum())
vpn = vk[vk >= VPN_MIN_SYN].sort_values(ascending=False, kind="stable")
apl = list(R[R.cell_type == "APL"].index)
kcout = c[c.Presynaptic_ID.isin(kc)].groupby("Postsynaptic_ID").Connectivity.sum()
mbon = kcout[kcout.index.isin(a[a.cell_class == "MBON"].index)]
mbon = mbon[mbon >= MBON_MIN_SYN].sort_values(ascending=False, kind="stable")
dan = R[R.cell_class == "DAN"]
pam = list(dan[dan.cell_type.str.startswith("PAM")].index)
ppl1 = list(dan[dan.cell_type.str.startswith("PPL1")].index)

# MBON -> (<=1 intermediate) -> descending neurons, every hop >= HOP_MIN_SYN synapses
dn_all = set(a[a.super_class == "descending"].index)
strong = c[c.Connectivity >= HOP_MIN_SYN]
taken = set(vpn.index) | set(kc) | set(apl) | set(mbon.index) | set(pam) | set(ppl1)
mid = set(strong[strong.Presynaptic_ID.isin(mbon.index)].Postsynaptic_ID) - dn_all - taken
mid = set(strong[strong.Presynaptic_ID.isin(mid) & strong.Postsynaptic_ID.isin(dn_all)].Presynaptic_ID)
dns = set(strong[strong.Presynaptic_ID.isin(mid | set(mbon.index)) & strong.Postsynaptic_ID.isin(dn_all)].Postsynaptic_ID) - taken

roles = [("vpn", list(vpn.index)), ("kc", kc), ("apl", apl), ("mbon", list(mbon.index)),
         ("pam", pam), ("ppl1", ppl1), ("mid", sorted(mid)), ("dn", sorted(dns))]
ids, role, tag = [], [], []
for r, members in roles:
    for i, rid in enumerate(members):
        ids.append(rid)
        role.append(r)
        # deterministic round-robin: VPNs -> 3 lanes, MBONs -> 3 action groups (both sorted by KC synapses)
        tag.append(i % 3 if r in ("vpn", "mbon") else -1)
index = {rid: i for i, rid in enumerate(ids)}
assert len(index) == len(ids), "neuron assigned to two roles"

e = c[c.Presynaptic_ID.isin(index) & c.Postsynaptic_ID.isin(index)]
e = e.assign(pre=e.Presynaptic_ID.map(index), post=e.Postsynaptic_ID.map(index)).sort_values(["pre", "post"])

# Drawing positions come from the Janelia male CNS (brain and nerve cord of one fly); FlyWire is brain-only.
# View from above with the head at the top: x across the body, z along it. The male CNS right side has small x,
# so x is mirrored to put the fly's right on the viewer's right.
m = pd.read_feather(RAW / "mcns_annotations.feather")
m = m[m.somaLocation.notna() & m.superclass.notna() & (m.statusLabel != "Glia")].copy()
soma = np.stack(m.somaLocation.to_numpy()).astype(float)
m["sx"], m["sz"] = soma[:, 0], soma[:, 2]
x0, x1, z0, z1 = m.sx.min(), m.sx.max(), m.sz.min(), m.sz.max()
nx = lambda v: (x1 - v) / (x1 - x0)
nz = lambda v: (v - z0) / (z1 - z0)
pts = np.stack([nx(m.sx) * 65535, nz(m.sz) * 65535], axis=1).round().astype("<u2")
OUT.mkdir(exist_ok=True)
pts.tofile(OUT / "brain_points.bin")

# Each simulated FlyWire neuron is drawn at the soma of a male CNS cell of the same type (same side when possible),
# dealing out that type's cells in bodyId order. Neurons whose type has no male CNS match are not drawn.
m["key"] = m.flywireType.fillna(m.type).astype(str)
by_type = {k: g.sort_values("bodyId") for k, g in m.groupby("key")}
side_code = {"right": "R", "left": "L"}
dealt, sx, sy = {}, [], []
for rid in ids:
    t, side = a.at[rid, "cell_type"], side_code.get(a.at[rid, "side"])
    g = by_type.get(t) if isinstance(t, str) else None
    if g is not None and (g.somaSide == side).any():
        g = g[g.somaSide == side]
    if g is None:
        sx.append(-1.0)
        sy.append(-1.0)
        continue
    i = dealt.get((t, side), 0)
    dealt[(t, side)] = i + 1
    row = g.iloc[i % len(g)]
    sx.append(nx(row.sx))
    sy.append(nz(row.sz))
out = {
    "meta": {
        "source": "FlyWire v783 (Dorkenwald et al. 2024; Schlegel et al. 2024); edges via Shiu et al. 2024",
        "positions": "Janelia male CNS v1.0 soma locations (CC-BY), view from above; simulated neurons matched by cell type",
        "aspect": float((z1 - z0) / (x1 - x0)),
        "n_unplaced": int(sum(v < 0 for v in sx)),
        "n_points": int(len(pts)),
        "counts": {r: len(m) for r, m in roles},
        "n_neurons": len(ids), "n_edges": int(len(e)), "n_synapses": int(e.Connectivity.sum()),
        "thresholds": {"vpn_min_syn": VPN_MIN_SYN, "mbon_min_syn": MBON_MIN_SYN, "hop_min_syn": HOP_MIN_SYN},
    },
    "neurons": {
        "role": role, "tag": tag,
        "type": [str(t) if isinstance(t, str) else "" for t in a.loc[ids].cell_type],
        "x": [round(float(v), 4) for v in sx], "y": [round(float(v), 4) for v in sy],
    },
    "edges": {"pre": e.pre.tolist(), "post": e.post.tolist(),
              "w": (e.Connectivity * e.Excitatory).astype(int).tolist()},
}
(OUT / "subgraph.json").write_text(json.dumps(out, separators=(",", ":")))
print(json.dumps(out["meta"], indent=1))
