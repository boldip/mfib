"""mfibnn: compress a chain-shaped ONNX network to a certified eps-approximate
M-fibration quotient, and evaluate it."""

from __future__ import annotations

import numpy as np
import onnx

import mfib

from . import netgraph, onnx_io


def analyze(model: onnx.ModelProto):
    pm = onnx_io.parse(model)
    chain, spatial = onnx_io.to_chain(pm)
    G, meta = netgraph.chain_to_graph(chain, pm.n_in, spatial)
    alphas = netgraph.layer_scales(G, meta)
    return pm, chain, G, meta, alphas


def compress(model: onnx.ModelProto, eps: float, metric: str = "layer-spread",
             center: str = "mean", verbose=False):
    """Returns (compressed ModelProto, report dict)."""
    pm, chain, G, meta, alphas = analyze(model)
    if metric == "layer-spread":
        Gs = netgraph.scale_graph(G, meta, alphas)
    elif metric == "plain":
        Gs, alphas = G, None
    else:
        raise ValueError(f"unknown metric {metric!r} (use 'layer-spread' or 'plain')")
    h = mfib.epsilon_partition(Gs, float(eps), monoid=mfib.VECTOR, metric=mfib.l1,
                               initial=meta["initial"], center=center, verbose=verbose)
    f = mfib.quotient_with_centers(Gs, h.partition, h.centers, monoid=mfib.VECTOR)
    defect = mfib.defect(f, metric=mfib.l1, monoid=mfib.VECTOR)   # in the scaled metric
    if alphas is not None:
        netgraph.unscale_fibration_labels(f, meta, alphas)
    comp_chain = netgraph.graph_to_chain(f, meta)
    new_model = onnx_io.rewrite(pm, comp_chain)
    onnx.checker.check_model(new_model)
    sizes_before = [l["W"].shape[0] for l in chain]
    sizes_after = [l["W"].shape[0] for l in comp_chain]
    params = lambda ch: int(sum(l["W"].size + l["b"].size for l in ch))
    report = {
        "eps": float(eps), "metric": metric, "center": center,
        "layers": [{"kind": k, "before": b, "after": a, "frozen": i >= len(chain) - 1}
                   for i, (k, b, a) in enumerate(zip(meta["kinds"], sizes_before, sizes_after))],
        "units_before": int(sum(sizes_before[:-1])), "units_after": int(sum(sizes_after[:-1])),
        "params_before": params(chain), "params_after": params(comp_chain),
        "certified_defect": float(defect), "certified": bool(defect <= eps + 1e-6),
        "rounds": h.rounds, "stopped_early": pm.stopped_early, "skipped_ops": pm.skipped,
        "layer_scales": None if alphas is None else {k: float(v) for k, v in alphas.items()},
    }
    return new_model, report
