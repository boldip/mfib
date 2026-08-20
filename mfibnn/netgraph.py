"""Chain <-> M-graph conversion for mfibnn (self-contained port of Experiments/nn_graph.py).

A *chain* is a list of layers {"kind": "conv"|"fc", "W", "b"}, with W of shape
(out, in, kh, kw) for conv and (out, in) for fc, plus, for each fc layer that
follows a conv layer, the spatial size S (H*W of the incoming feature map), so
that the fc weight columns are grouped per input channel (C-major, ONNX layout).
The M-graph has one node per channel/unit plus a "bias" node; labels are kernels
(R^{kh x kw}), spatial weight vectors (R^S) at the conv->fc boundary, and scalars.
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def chain_to_graph(chain, n_in, spatial, weight="weight"):
    """spatial: list, spatial[i] = S for layer i if it is an fc after a conv (else None)."""
    G = nx.MultiDiGraph()
    names = [[("in", c) for c in range(n_in)]]
    for l, layer in enumerate(chain):
        tag = "out" if l == len(chain) - 1 else f"h{l + 1}"
        names.append([(tag, k) for k in range(layer["W"].shape[0])])
    G.add_nodes_from(n for layer in names for n in layer)
    G.add_node("bias")
    prev_kind = "conv"
    for l, layer in enumerate(chain):
        W, b = layer["W"], layer["b"]
        src, dst = names[l], names[l + 1]
        if layer["kind"] == "conv":
            for k in range(W.shape[0]):
                for c in range(W.shape[1]):
                    ker = W[k, c]
                    if np.any(ker != 0):
                        G.add_edge(src[c], dst[k], **{weight: np.array(ker, dtype=float)})
                if b[k] != 0:
                    G.add_edge("bias", dst[k], **{weight: float(b[k])})
        elif spatial[l] is not None:                   # flatten boundary
            S = spatial[l]
            C = len(src)
            Wr = W.reshape(W.shape[0], C, S)
            for k in range(W.shape[0]):
                for c in range(C):
                    vec = Wr[k, c]
                    if np.any(vec != 0):
                        G.add_edge(src[c], dst[k], **{weight: np.array(vec, dtype=float)})
                if b[k] != 0:
                    G.add_edge("bias", dst[k], **{weight: float(b[k])})
        else:
            for k in range(W.shape[0]):
                for j in np.flatnonzero(W[k]):
                    G.add_edge(src[j], dst[k], **{weight: float(W[k, j])})
                if b[k] != 0:
                    G.add_edge("bias", dst[k], **{weight: float(b[k])})
        prev_kind = layer["kind"]
    initial = {}
    for i, n in enumerate(names[0]):
        initial[n] = ("in", i)
    for l in range(1, len(names) - 1):
        for n in names[l]:
            initial[n] = ("layer", l)
    for k, n in enumerate(names[-1]):
        initial[n] = ("out", k)
    initial["bias"] = ("bias",)
    return G, {"names": names, "initial": initial,
               "kinds": [l["kind"] for l in chain], "spatial": spatial}


def graph_to_chain(f, meta, weight="weight"):
    names, kinds, spatial = meta["names"], meta["kinds"], meta["spatial"]
    B = f.B
    layer_classes = []
    for l in range(len(names)):
        seen = []
        for n in names[l]:
            c = f.node_map[n]
            if c not in seen:
                seen.append(c)
        layer_classes.append(seen)
    bias_class = f.node_map["bias"]
    labels = {(u, v): d.get(weight, 0.0) for u, v, k, d in B.edges(keys=True, data=True)}
    chain = []
    prev_kind = "conv"
    for l, kind in enumerate(kinds):
        prev, cur = layer_classes[l], layer_classes[l + 1]
        if kind == "conv":
            kshape = next(np.shape(v) for (u, t), v in labels.items()
                          if t in cur and u in prev and np.ndim(v) == 2)
            W = np.zeros((len(cur), len(prev)) + tuple(kshape))
            for k, ck in enumerate(cur):
                for j, cj in enumerate(prev):
                    lab = labels.get((cj, ck))
                    if lab is not None:
                        W[k, j] = lab
        elif spatial[l] is not None:
            S = spatial[l]
            W = np.zeros((len(cur), len(prev) * S))
            for k, ck in enumerate(cur):
                for j, cj in enumerate(prev):
                    lab = labels.get((cj, ck))
                    if lab is not None:
                        W[k, j * S:(j + 1) * S] = np.asarray(lab).ravel()
        else:
            W = np.zeros((len(cur), len(prev)))
            for k, ck in enumerate(cur):
                for j, cj in enumerate(prev):
                    lab = labels.get((cj, ck))
                    if lab is not None:
                        W[k, j] = float(lab)
        b = np.array([float(labels.get((bias_class, ck), 0.0)) for ck in cur])
        chain.append({"kind": kind, "W": W, "b": b})
        prev_kind = kind
    return chain


def layer_scales(G, meta, weight="weight", sample=400, seed=0):
    """alpha_l = 1 / median pairwise l1 distance between the raw in-label vectors
    of the units of layer l (the weighted product metric of the paper)."""
    rng = np.random.default_rng(seed)
    names = meta["names"]
    alphas = {}
    for l in range(1, len(names)):
        nodes = names[l]
        rows, keys = [], {}
        for n in nodes:
            r = {}
            for u, _, d in G.in_edges(n, data=True):
                r[u] = np.asarray(d.get(weight, 1.0), dtype=float).ravel()
                keys.setdefault(u, r[u].size)
            rows.append(r)
        if not keys or len(nodes) < 2:
            alphas[l] = 1.0
            continue
        offs, tot = {}, 0
        for u in keys:
            offs[u] = tot; tot += keys[u]
        F = np.zeros((len(nodes), tot))
        for i, r in enumerate(rows):
            for u, v in r.items():
                F[i, offs[u]:offs[u] + v.size] = v
        if len(nodes) > sample:
            F = F[rng.choice(len(nodes), sample, replace=False)]
        d = np.concatenate([np.abs(F[i + 1:] - F[i]).sum(1) for i in range(len(F))]) if len(F) > 1 else np.array([0.0])
        med = float(np.median(d))
        alphas[l] = 1.0 / med if med > 0 else 1.0
    return alphas


def scale_graph(G, meta, alphas, weight="weight"):
    H = G.copy()
    layer_of = {n: l for l, layer in enumerate(meta["names"]) for n in layer}
    for u, v, k, d in H.edges(keys=True, data=True):
        l = layer_of.get(v)
        if l in alphas and l >= 1:
            d[weight] = d.get(weight, 1.0) * alphas[l]
    return H


def unscale_fibration_labels(f, meta, alphas, weight="weight"):
    layer_of_class = {}
    for l, layer in enumerate(meta["names"]):
        for n in layer:
            layer_of_class[f.node_map[n]] = l
    for u, v, k, d in f.B.edges(keys=True, data=True):
        l = layer_of_class.get(v)
        if l in alphas and l >= 1:
            d[weight] = d[weight] / alphas[l]
    return f
