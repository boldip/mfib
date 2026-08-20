"""mfibnn: certified epsilon-fibration compression of chain-shaped ONNX networks.

    import onnx, mfibnn
    model = onnx.load("model.onnx")
    small, report = mfibnn.compress(model, eps=0.35)
    acc = mfibnn.evaluate(small, X, y)

CLI:  python3 -m mfibnn {info|compress|eval} ...   (see mfibnn/README.md)
"""

from .core import analyze, compress
from .onnx_io import evaluate, parse

__all__ = ["analyze", "compress", "evaluate", "parse"]
__version__ = "0.1.0"
