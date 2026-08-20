import argparse
import json
import sys

import numpy as np
import onnx

from . import compress, evaluate
from .core import analyze


def _load_data(path):
    z = np.load(path)
    X = z["X"] if "X" in z else z["x"]
    y = z["y"]
    return X, y


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mfibnn",
                                 description="Certified epsilon-fibration compression of chain-shaped ONNX networks")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("info", help="show the detected chain, unit counts and layer spreads")
    p.add_argument("model")
    p = sub.add_parser("compress", help="compress to a certified eps-fibration quotient")
    p.add_argument("model")
    p.add_argument("--eps", type=float, required=True,
                   help="defect budget; with the default metric, a fraction of each layer's spread")
    p.add_argument("--out", default=None, help="output path (default: <model>_eps<eps>.onnx)")
    p.add_argument("--data", default=None, help=".npz with X and y: report accuracy before/after")
    p.add_argument("--metric", choices=["layer-spread", "plain"], default="layer-spread")
    p.add_argument("--center", choices=["mean", "iter", "seed"], default="mean")
    p.add_argument("--verbose", action="store_true", help="trace the refinement algorithm")
    p.add_argument("--json", action="store_true", help="print the report as JSON")
    p = sub.add_parser("eval", help="top-1 accuracy of a model on an .npz dataset")
    p.add_argument("model")
    p.add_argument("--data", required=True)
    a = ap.parse_args(argv)

    model = onnx.load(a.model)
    if a.cmd == "info":
        pm, chain, G, meta, alphas = analyze(model)
        print(f"input: {pm.input_name}  output: {pm.output_name}  "
              f"chain of {len(chain)} layers" + (" (stopped early: unsupported "
              + ",".join(pm.skipped) + ")" if pm.stopped_early else ""))
        for i, (l, sp) in enumerate(zip(chain, meta["spatial"])):
            frozen = " [frozen]" if i >= len(chain) - 1 else ""
            extra = f", spatial {sp}" if sp else ""
            print(f"  layer {i}: {l['kind']:4} {l['W'].shape[0]:>5} units, W {l['W'].shape}{extra}, "
                  f"spread {1/alphas[i+1]:.4g}{frozen}")
        print(f"graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} arcs")
        return 0
    if a.cmd == "eval":
        X, y = _load_data(a.data)
        print(f"accuracy: {evaluate(model, X, y):.4f}")
        return 0
    small, report = compress(model, a.eps, metric=a.metric, center=a.center, verbose=a.verbose)
    out = a.out or a.model.replace(".onnx", f"_eps{a.eps:g}.onnx")
    onnx.save(small, out)
    if a.data:
        X, y = _load_data(a.data)
        report["accuracy_before"] = round(evaluate(model, X, y), 4)
        report["accuracy_after"] = round(evaluate(small, X, y), 4)
    if a.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"wrote {out}")
        for l in report["layers"]:
            print(f"  {l['kind']:4} {l['before']:>5} -> {l['after']:>5}" + ("  [frozen]" if l["frozen"] else ""))
        print(f"units {report['units_before']} -> {report['units_after']} "
              f"({report['units_after']/max(report['units_before'],1):.0%}), "
              f"params {report['params_before']} -> {report['params_after']} "
              f"({report['params_after']/report['params_before']:.0%})")
        print(f"certified defect {report['certified_defect']:.4g} <= eps: {report['certified']}"
              + (f"; rounds {report['rounds']}"))
        if "accuracy_before" in report:
            print(f"accuracy: {report['accuracy_before']:.4f} -> {report['accuracy_after']:.4f}")
        if report["stopped_early"]:
            print(f"note: chain stopped early at unsupported op(s) {report['skipped_ops']}; "
                  "everything from the last parsed layer on was left untouched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
