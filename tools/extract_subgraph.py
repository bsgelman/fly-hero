"""Extract the Fly Hero subgraph + brain point cloud from public FlyWire v783 data.

Inputs (download into data/raw/, see README):
  Connectivity_783.parquet  - github.com/philshiu/Drosophila_brain_model
  annotations.tsv           - github.com/flyconnectome/flywire_annotations
                              supplemental_files/Supplemental_file1_neuron_annotations.tsv
Outputs:
  data/subgraph.json    simulated neurons (role, lane/group, position) + signed edges
  data/brain_points.bin Uint16 (x,y) front-view positions of all annotated neurons
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

# front view: x mirrored so the fly's right hemisphere appears on the viewer's left, y grows ventrally
x0, x1 = a.pos_x.min(), a.pos_x.max()
y0, y1 = a.pos_y.min(), a.pos_y.max()
def norm(df):
    return (x1 - df.pos_x) / (x1 - x0), (df.pos_y - y0) / (y1 - y0)
nx, ny = norm(a)
pts = np.stack([nx * 65535, ny * 65535], axis=1).round().astype("<u2")
OUT.mkdir(exist_ok=True)
pts.tofile(OUT / "brain_points.bin")

sx, sy = norm(a.loc[ids])
out = {
    "meta": {
        "source": "FlyWire v783 (Dorkenwald et al. 2024; Schlegel et al. 2024); edges via Shiu et al. 2024",
        "aspect": float((y1 - y0) / (x1 - x0)),
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
