# mfibnn — certified ε-fibration compression of ONNX networks

`mfibnn` takes a neural network in **ONNX** format and an evaluation dataset, compresses
the network to a desired tolerance ε by merging equivalent-up-to-ε units/channels
(the ε-approximate M-fibration quotient — see the paper in the
[Superfibrations repository](https://github.com/boldip/Superfibrations) — computed
with the `mfib` library), writes a smaller
ONNX model of the same architecture, **certifies** the worst-case defect ≤ ε, and
reports the test accuracy before and after. No retraining is involved.

## 1. Install

From the repository root (Python ≥ 3.9):

```bash
pip install numpy scipy networkx onnx onnxruntime
```

Everything is run from the repository root, so that both `mfib/` and `mfibnn/` are
importable (`pip install -e .` also works).

## 2. Get a network (ONNX)

* **ONNX Model Zoo** — <https://github.com/onnx/models> — ready-made, validated models
  with documented accuracy. The example used below:

  ```bash
  curl -L -o mnist-12.onnx \
    https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12.onnx
  ```

* **Hugging Face Hub** — <https://huggingface.co/models?library=onnx> — filter by the
  ONNX library tag; download the `.onnx` file from the repo's *Files* tab (or with
  `huggingface_hub`).
* **Export your own** from PyTorch:

  ```python
  torch.onnx.export(model.eval(), torch.zeros(1, *input_shape), "model.onnx",
                    input_names=["x"], output_names=["y"],
                    dynamic_axes={"x": {0: "N"}, "y": {0: "N"}})
  ```

**Supported architectures (v1):** single-chain feedforward networks — `Gemm`/`MatMul(+Add)`
(fully connected), `Conv` (groups = 1; bias inline or as a following `Add`),
`BatchNormalization` (folded exactly), any elementwise activation, `MaxPool`/`AveragePool`/
`GlobalAveragePool`, `Flatten`/`Reshape`, `Dropout`, final `Softmax`. This covers MLPs and
LeNet/VGG-style CNNs. On anything else (residual `Add`, `Concat`, attention) the tool
compresses the maximal supported prefix and leaves the rest untouched — the certificate
still holds; `info` tells you what was detected.

## 3. Get the data (as ready tensors)

The dataset must be an `.npz` with `X` (inputs, first axis = samples, already
preprocessed exactly as the network expects) and `y` (integer labels). **Preprocessing
is your responsibility** — it is the one thing no exchange format standardizes.

* **MNIST** (for `mnist-12.onnx` and MNIST MLPs) — a helper downloads the test set from
  the public S3 mirror and writes the `.npz`:

  ```bash
  python3 -m mfibnn.make_mnist_npz mnist_test.npz          # (10000,1,28,28), for CNNs
  python3 -m mfibnn.make_mnist_npz mnist_flat.npz --flat   # (10000,784), for MLPs
  ```

  (Original distribution: <https://ossci-datasets.s3.amazonaws.com/mnist/>; the classic
  yann.lecun.com page is often unavailable.)
* **CIFAR-10** — <https://www.cs.toronto.edu/~kriz/cifar.html> (python pickle batches;
  remember the per-channel mean/std normalization used by the model you download).
* **Anything else** — Hugging Face `datasets` (<https://huggingface.co/datasets>) or
  OpenML (<https://www.openml.org>) and a few lines of numpy to dump the `.npz`.

## 4. Run

```bash
# what was detected: layers, unit counts, per-layer spreads
python3 -m mfibnn info mnist-12.onnx

# accuracy of the original model
python3 -m mfibnn eval mnist-12.onnx --data mnist_test.npz

# compress to eps and evaluate both models
python3 -m mfibnn compress mnist-12.onnx --eps 0.4 --data mnist_test.npz
```

Output of the last command (real run):

```
wrote mnist-12_eps0.4.onnx
  conv     8 ->     5
  conv    16 ->     9
  fc      10 ->    10  [frozen]
units 24 -> 14 (58%), params 5994 -> 2714 (45%)
certified defect 0.3982 <= eps: True; rounds 3
accuracy: 0.9890 -> 0.9106
```

Options: `--metric layer-spread` (default: ε is measured, on every layer, in units of
the layer's median pairwise weight distance — the weighted product metric; strongly
recommended) or `--metric plain` (raw ℓ¹); `--center mean|iter|seed` (center rule);
`--out FILE`; `--json` (machine-readable report); `--verbose` (trace of the refinement
algorithm, the variables of Algorithm 1 of the paper). Library use:

```python
import onnx, mfibnn
small, report = mfibnn.compress(onnx.load("model.onnx"), eps=0.4)
acc = mfibnn.evaluate(small, X, y)
```

## 5. How to read the numbers

* **certified defect ≤ ε** is a *worst-case guarantee on the weights*: for every unit of
  the compressed network, the aggregated incoming weights differ from those of each unit
  it replaces by at most ε (in the scaled ℓ¹ metric). It is *not* an accuracy guarantee.
* Accuracy typically stays flat for small ε and then drops sharply once ε crosses
  ≈ 0.35–0.5 of the layer spread (a concentration phenomenon: see the paper); the useful
  regime is the shoulder just below the drop. Sweep ε (e.g. 0.2 … 0.5) and pick.
* Inputs, outputs, and the last layer of the detected chain are never merged (frozen);
  ε = 0 reproduces the original network exactly (bit-for-bit function).

## 6. Choosing ε end to end (worked example, MLP)

```bash
python3 -m mfibnn.make_mnist_npz mnist_flat.npz --flat
for e in 0 0.2 0.3 0.4 0.5; do
  python3 -m mfibnn compress mlp.onnx --eps $e --data mnist_flat.npz --json \
    | python3 -c 'import json,sys; r=json.load(sys.stdin); print(r["eps"], r["units_after"], r["accuracy_after"])'
done
```

For the LeNet-300-100 MLP of the Superfibrations experiments this gives 400 → 305
units at −0.2 accuracy points (ε = 0.4) and the drop at ε ≈ 0.5 — identical, by
construction, to the `mfib` experiments reported in the paper.
