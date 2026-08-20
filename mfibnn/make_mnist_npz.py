"""Download the MNIST test set and write an .npz ready for mfibnn.

Usage: python3 -m mfibnn.make_mnist_npz out.npz [--flat]
  default: X of shape (10000, 1, 28, 28), float32 in [0, 1]  (for CNNs, e.g. mnist-12)
  --flat:  X of shape (10000, 784)                           (for MLPs)
"""

import gzip
import struct
import sys
import urllib.request

import numpy as np

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"


def _fetch(fname):
    with urllib.request.urlopen(MIRROR + fname) as r:
        raw = gzip.decompress(r.read())
    magic = struct.unpack(">I", raw[:4])[0]
    ndim = magic & 0xFF
    dims = struct.unpack(">" + "I" * ndim, raw[4:4 + 4 * ndim])
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def main(argv=None):
    args = argv if argv is not None else sys.argv[1:]
    out = args[0] if args else "mnist_test.npz"
    flat = "--flat" in args
    X = _fetch("t10k-images-idx3-ubyte.gz").astype(np.float32) / 255.0
    y = _fetch("t10k-labels-idx1-ubyte.gz").astype(np.int64)
    X = X.reshape(-1, 784) if flat else X.reshape(-1, 1, 28, 28)
    np.savez_compressed(out, X=X, y=y)
    print(f"wrote {out}: X {X.shape} float32 in [0,1], y {y.shape}")


if __name__ == "__main__":
    main()
