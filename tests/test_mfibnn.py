import os

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

import mfibnn

HERE = os.path.dirname(os.path.abspath(__file__))


def tiny_mlp(w1, b1, w2, b2):
    """x (N,4) -> Gemm(w1,b1) -> Relu -> Gemm(w2,b2): weights given as (out, in)."""
    nodes = [
        helper.make_node("Gemm", ["x", "W1", "b1"], ["z1"], transB=1),
        helper.make_node("Relu", ["z1"], ["h1"]),
        helper.make_node("Gemm", ["h1", "W2", "b2"], ["out"], transB=1),
    ]
    graph = helper.make_graph(
        nodes, "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", w1.shape[1]])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, ["N", w2.shape[0]])],
        [numpy_helper.from_array(w1.astype(np.float32), "W1"),
         numpy_helper.from_array(b1.astype(np.float32), "b1"),
         numpy_helper.from_array(w2.astype(np.float32), "W2"),
         numpy_helper.from_array(b2.astype(np.float32), "b2")])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def test_exact_merge_and_roundtrip():
    rng = np.random.default_rng(0)
    w1 = rng.normal(size=(6, 4)); w1[3] = w1[0]           # units 0 and 3 identical
    b1 = rng.normal(size=6); b1[3] = b1[0]
    w2 = rng.normal(size=(3, 6)); b2 = rng.normal(size=3)
    model = tiny_mlp(w1, b1, w2, b2)
    X = rng.normal(size=(50, 4)).astype(np.float32)
    y = rng.integers(0, 3, 50)
    small, rep = mfibnn.compress(model, eps=0.0, metric="plain")
    assert rep["certified"] and rep["certified_defect"] == 0
    assert rep["units_after"] == 5                        # the twin units merged
    import onnxruntime as ort
    a = ort.InferenceSession(model.SerializeToString()).run(None, {"x": X})[0]
    b = ort.InferenceSession(small.SerializeToString()).run(None, {"x": X})[0]
    assert np.allclose(a, b, atol=1e-4)                   # exact fibration: same function
    assert mfibnn.evaluate(model, X, y) == mfibnn.evaluate(small, X, y)


def test_eps_certificate_and_monotonicity():
    rng = np.random.default_rng(1)
    model = tiny_mlp(rng.normal(size=(8, 4)), rng.normal(size=8),
                     rng.normal(size=(3, 8)), rng.normal(size=3))
    sizes = []
    for eps in (0.0, 0.5, 1.0, 2.0):
        _, rep = mfibnn.compress(model, eps=eps)
        assert rep["certified"], (eps, rep["certified_defect"])
        sizes.append(rep["units_after"])
    assert sizes == sorted(sizes, reverse=True)


@pytest.mark.skipif(not os.path.exists(os.path.join(HERE, "..", "mfibnn", "mnist-12.onnx")),
                    reason="zoo model not downloaded")
def test_zoo_mnist12_parses_and_roundtrips():
    model = onnx.load(os.path.join(HERE, "..", "mfibnn", "mnist-12.onnx"))
    pm = mfibnn.parse(model)
    assert [l.kind for l in pm.layers] == ["conv", "conv", "fc"] and not pm.stopped_early
    small, rep = mfibnn.compress(model, eps=0.0)
    assert rep["certified_defect"] == 0 and rep["units_after"] == rep["units_before"]
    onnx.checker.check_model(small)
