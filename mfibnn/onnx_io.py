"""ONNX <-> chain: parse a chain-shaped ONNX model, rewrite it after compression,
evaluate with onnxruntime.

Supported (v1): a single-path graph made of
  Conv (group=1; bias as 3rd input or as a following Add with an initializer),
  Gemm / MatMul(+Add) (fully connected; weight possibly behind a constant Reshape),
  BatchNormalization (folded into the preceding Conv/Gemm: exact at inference),
  elementwise activations (Relu, LeakyRelu, Sigmoid, Tanh, Elu, Clip, Selu, ...),
  MaxPool / AveragePool / GlobalAveragePool / LpPool,
  Flatten / Reshape (flatten boundary), Dropout / Identity, Softmax (last).
Anything else ends the compressible region: the last parsed layer before it is
frozen (its units are never merged), and the rest of the graph is left verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import onnx
from onnx import numpy_helper

ACTIVATIONS = {"Relu", "LeakyRelu", "Sigmoid", "Tanh", "Elu", "Selu", "Clip",
               "HardSigmoid", "Softplus", "Softsign", "PRelu"}
POOLS = {"MaxPool", "AveragePool", "GlobalAveragePool", "LpPool"}
TRANSPARENT = {"Dropout", "Identity"}


@dataclass
class Layer:
    kind: str                 # "conv" | "fc"
    W: np.ndarray             # conv: (out, in, kh, kw); fc: (out, in)
    b: np.ndarray             # (out,)
    node: onnx.NodeProto      # the Conv/Gemm/MatMul node
    w_init: str               # name of the weight initializer to rewrite
    b_init: Optional[str]     # name of the bias initializer (None if no bias)
    w_reshape_init: Optional[str] = None   # shape initializer of a weight Reshape to fix
    w_pre_reshape: Optional[tuple] = None  # original (pre-reshape) weight shape
    gemm_transB: int = 1                   # for Gemm: whether W is stored (out, in)
    bn: Optional[onnx.NodeProto] = None    # folded BatchNormalization node (to remove)
    spatial: Optional[int] = None          # for fc after conv: H*W of the incoming map


@dataclass
class ParsedModel:
    model: onnx.ModelProto
    layers: List[Layer]
    n_in: int                 # input channels (conv first) or input features (fc first)
    input_name: str
    output_name: str
    batch_dim: Optional[int]  # None if symbolic/dynamic
    stopped_early: bool       # True if an unsupported op ended the chain
    skipped: List[str] = field(default_factory=list)


def _inits(graph):
    return {t.name: t for t in graph.initializer}


def _shapes(model):
    """tensor name -> list of dims (ints; 0 for symbolic) via shape inference."""
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception:
        inferred = model
    out = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        dims = [d.dim_value if d.HasField("dim_value") else 0
                for d in vi.type.tensor_type.shape.dim]
        out[vi.name] = dims
    return out


def parse(model: onnx.ModelProto) -> ParsedModel:
    graph = model.graph
    inits = _inits(graph)
    shapes = _shapes(model)
    consumers: Dict[str, list] = {}
    for n in graph.node:
        for x in n.input:
            consumers.setdefault(x, []).append(n)
    graph_inputs = [i for i in graph.input if i.name not in inits]
    if len(graph_inputs) != 1 or len(graph.output) != 1:
        raise ValueError("mfibnn v1 supports models with exactly one input and one output")
    input_name = graph_inputs[0].name
    output_name = graph.output[0].name
    in_dims = [d.dim_value if d.HasField("dim_value") else 0
               for d in graph_inputs[0].type.tensor_type.shape.dim]
    batch_dim = in_dims[0] if in_dims and in_dims[0] > 0 else None
    n_in = in_dims[1] if len(in_dims) >= 2 and in_dims[1] > 0 else 1

    layers: List[Layer] = []
    skipped: List[str] = []
    stopped = False
    cur = input_name                      # tensor we are following
    prev_kind = "conv" if len(in_dims) == 4 else "fc"
    last_conv_out: Optional[str] = None   # tensor whose spatial size feeds the next fc

    def single_consumer(t):
        cs = consumers.get(t, [])
        return cs[0] if len(cs) == 1 else None

    while cur != output_name:
        node = single_consumer(cur)
        if node is None:
            stopped = True
            break
        op = node.op_type
        if op in ACTIVATIONS or op in TRANSPARENT or (op == "Softmax"):
            cur = node.output[0]; continue
        if op in POOLS:
            last_conv_out = node.output[0]
            cur = node.output[0]; continue
        if op in ("Flatten", "Reshape"):
            # flatten boundary (data reshape); weight reshapes are handled at MatMul
            cur = node.output[0]; continue
        if op == "BatchNormalization":
            if layers and layers[-1].node.output[0] == _origin(node.input[0], layers):
                pass  # will be folded below when constructed after its conv; here fold into last layer
            if not layers:
                stopped = True; break
            lay = layers[-1]
            g = numpy_helper.to_array(inits[node.input[1]]).astype(np.float64)
            beta = numpy_helper.to_array(inits[node.input[2]]).astype(np.float64)
            mu = numpy_helper.to_array(inits[node.input[3]]).astype(np.float64)
            var = numpy_helper.to_array(inits[node.input[4]]).astype(np.float64)
            eps_attr = next((a.f for a in node.attribute if a.name == "epsilon"), 1e-5)
            s = g / np.sqrt(var + eps_attr)
            lay.W = lay.W * s.reshape([-1] + [1] * (lay.W.ndim - 1))
            lay.b = (lay.b - mu) * s + beta
            lay.bn = node
            if lay.kind == "conv":
                last_conv_out = node.output[0]
            cur = node.output[0]; continue
        if op == "Conv":
            group = next((a.i for a in node.attribute if a.name == "group"), 1)
            if group != 1 or node.input[1] not in inits:
                stopped = True; break
            W = numpy_helper.to_array(inits[node.input[1]]).astype(np.float64)
            b = (numpy_helper.to_array(inits[node.input[2]]).astype(np.float64)
                 if len(node.input) > 2 and node.input[2] in inits else np.zeros(W.shape[0]))
            b_init = node.input[2] if len(node.input) > 2 else None
            lay = Layer("conv", W, b, node, node.input[1], b_init)
            out = node.output[0]
            # a following Add with a per-channel initializer is the bias
            nxt = single_consumer(out)
            if b_init is None and nxt is not None and nxt.op_type == "Add":
                other = [x for x in nxt.input if x != out]
                if len(other) == 1 and other[0] in inits:
                    bias = numpy_helper.to_array(inits[other[0]]).astype(np.float64).ravel()
                    if bias.size == W.shape[0]:
                        lay.b = bias
                        lay.b_init = other[0]
                        out = nxt.output[0]
            layers.append(lay)
            last_conv_out = out
            prev_kind = "conv"
            cur = out; continue
        if op in ("Gemm", "MatMul"):
            # weight input: for Gemm it is input B (position 1, with A the data);
            # for MatMul, the non-data input; possibly behind a constant Reshape
            w_name, w_reshape_init, w_pre_shape = None, None, None
            if op == "Gemm":
                cands = [node.input[1]] if node.input[0] == cur else [node.input[0]]
            else:
                cands = [x for x in node.input if x != cur]
            for x in cands:
                if x in inits:
                    w_name = x
                else:
                    prod = [n2 for n2 in graph.node if x in n2.output]
                    if prod and prod[0].op_type == "Reshape" and prod[0].input[0] in inits:
                        w_name = prod[0].input[0]
                        w_reshape_init = prod[0].input[1]
                        w_pre_shape = tuple(numpy_helper.to_array(inits[w_name]).shape)
            if w_name is None:
                stopped = True; break
            W = numpy_helper.to_array(inits[w_name]).astype(np.float64)
            if w_pre_shape is not None:
                W = W.reshape(-1, W.shape[-1]) if W.ndim > 2 else W
            transB = next((a.i for a in node.attribute if a.name == "transB"), 0) if op == "Gemm" else 0
            if op == "MatMul" or transB == 0:
                W = W.T                              # normalize to (out, in)
            b = np.zeros(W.shape[0]); b_init = None
            out = node.output[0]
            if op == "Gemm" and len(node.input) > 2 and node.input[2] in inits:
                b = numpy_helper.to_array(inits[node.input[2]]).astype(np.float64).ravel()
                b_init = node.input[2]
            else:
                nxt = single_consumer(out)
                if nxt is not None and nxt.op_type == "Add":
                    other = [x for x in nxt.input if x != out]
                    if len(other) == 1 and other[0] in inits:
                        bias = numpy_helper.to_array(inits[other[0]]).astype(np.float64).ravel()
                        if bias.size == W.shape[0]:
                            b = bias; b_init = other[0]; out = nxt.output[0]
            lay = Layer("fc", W, b, node, w_name, b_init,
                        w_reshape_init=w_reshape_init, w_pre_reshape=w_pre_shape,
                        gemm_transB=(transB if op == "Gemm" else -1))
            if prev_kind == "conv":
                dims = shapes.get(last_conv_out or "", [])
                if len(dims) == 4 and dims[2] > 0 and dims[3] > 0:
                    lay.spatial = dims[2] * dims[3]
                elif len(dims) == 4:
                    stopped = True; break
                else:
                    lay.spatial = 1
            layers.append(lay)
            prev_kind = "fc"
            cur = out; continue
        skipped.append(op)
        stopped = True
        break
    return ParsedModel(model, layers, n_in, input_name, output_name, batch_dim, stopped, skipped)


def _origin(name, layers):
    return name


def to_chain(pm: ParsedModel):
    """The chain of the *compressible region* and its metadata. Layers taking part:
    all parsed layers; mergeable are 0..L-2 (the last parsed layer is frozen: it is
    the classifier, or it feeds an unsupported region)."""
    chain = [{"kind": l.kind, "W": l.W, "b": l.b} for l in pm.layers]
    spatial = [l.spatial for l in pm.layers]
    return chain, spatial


def rewrite(pm: ParsedModel, chain) -> onnx.ModelProto:
    """A copy of the model with the compressed weights in place: initializers of
    every parsed layer replaced (shapes updated), folded BatchNormalizations and
    absorbed biases handled, flatten Reshape shape constants relaxed to [0, -1]."""
    model = onnx.ModelProto(); model.CopyFrom(pm.model)
    graph = model.graph
    inits = _inits(graph)

    def set_init(name, arr, like=None):
        old = inits[name]
        dtype = numpy_helper.to_array(old).dtype
        new = numpy_helper.from_array(arr.astype(dtype), name)
        old.CopyFrom(new)

    remove_nodes = []
    for lay, comp in zip(pm.layers, chain):
        W, b = comp["W"], comp["b"]
        if lay.kind == "conv":
            set_init(lay.w_init, W)
            if lay.b_init is not None:
                old_b = numpy_helper.to_array(inits[lay.b_init])
                set_init(lay.b_init, b.reshape((-1,) + old_b.shape[1:]) if old_b.ndim > 1 else b)
            elif np.any(b != 0) and lay.bn is None:
                raise RuntimeError("conv without bias produced a nonzero compressed bias")
        else:
            Wst = W                                   # (out, in) -> storage layout
            if lay.gemm_transB in (0, -1):            # stored as (in, out)
                Wst = W.T
            if lay.w_reshape_init is not None:
                set_init(lay.w_init, Wst)             # store already-reshaped 2-D weight
                set_init(lay.w_reshape_init, np.array(Wst.shape, dtype=np.int64))
            else:
                set_init(lay.w_init, Wst)
            if lay.b_init is not None:
                set_init(lay.b_init, b)
            elif np.any(b != 0):
                raise RuntimeError("fc without bias produced a nonzero compressed bias")
        if lay.bn is not None:
            remove_nodes.append(lay.bn)
    # remove folded BN nodes, rewiring their outputs
    for bn in remove_nodes:
        src = bn.input[0]
        dst = bn.output[0]
        for n in graph.node:
            for i, x in enumerate(n.input):
                if x == dst:
                    n.input[i] = src
        for o in graph.output:
            if o.name == dst:
                o.name = src
        graph.node.remove(bn)
        for name in list(bn.input[1:]):
            t = next((t for t in graph.initializer if t.name == name), None)
            if t is not None and not any(name in n.input for n in graph.node):
                graph.initializer.remove(t)
    # relax data-flatten Reshape shape constants to [0, -1]
    for n in graph.node:
        if n.op_type == "Reshape" and n.input[1] in inits and n.input[0] not in inits:
            producer_is_weight = any(n.input[0] == l.w_init for l in pm.layers)
            if not producer_is_weight:
                shp = numpy_helper.to_array(inits[n.input[1]])
                if shp.ndim == 1 and shp.size == 2:
                    set_init(n.input[1], np.array([0, -1], dtype=np.int64))
    del graph.value_info[:]
    return model


def evaluate(model: onnx.ModelProto, X: np.ndarray, y: np.ndarray, batch=256) -> float:
    import onnxruntime as ort
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    ishape = sess.get_inputs()[0].shape
    fixed_batch = isinstance(ishape[0], int) and ishape[0] > 0
    step = ishape[0] if fixed_batch else batch
    correct = 0
    for i in range(0, len(X), step):
        xb = X[i:i + step].astype(np.float32)
        if fixed_batch and len(xb) != step:
            break
        out = sess.run(None, {iname: xb})[0]
        correct += int((out.reshape(len(xb), -1).argmax(1) == y[i:i + len(xb)]).sum())
    n = (len(X) // step) * step if fixed_batch else len(X)
    return correct / n
