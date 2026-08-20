# mfib — M-fibrations of monoid-labelled graphs

Python library and tools for **M-fibrations**: fibrations of directed multigraphs whose
arcs are labelled in a commutative monoid, as developed in the paper *"M-Fibration
Theory with Applications to Neural Network Compression"* (P. Boldi; see the
[Superfibrations repository](https://github.com/boldip/Superfibrations) for the paper,
the working notes and the experiments). It generalizes graph fibrations
(Boldi–Vigna, *Discrete Math.* 243, 2002), equitable partitions, colour refinement /
1-WL, and exact lumpability of Markov chains.

Two packages:

* **`mfib`** — the core library on `networkx` multidigraphs: monoids as `(zero, add, key)`
  (with built-ins: additive reals with rounding, exact ℕ/ℤ, ℤ_k, max, min, bool,
  multisets, vector monoids for array labels), `coarsest_equitable_partition`,
  `minimum_base` / `quotient` (minimum bases and minimum M-fibrations),
  ε-approximate theory (`epsilon_partition` with certificates, `defect`,
  `quotient_with_centers`, exact `beta_exact` by MILP, `optimal_centers`,
  `unevenness`), behavioural checks (`is_fibration`, `is_equitable`, `is_prime`).
  See `mfib/README.md`.
* **`mfibnn`** — a self-contained tool that compresses a neural network in **ONNX**
  format to a certified ε-approximate M-fibration quotient and evaluates it:
  `python3 -m mfibnn {info|compress|eval}`. See `mfibnn/README.md` for an
  end-to-end guide (including where to download networks and data).

## Install

```bash
pip install networkx numpy scipy          # core
pip install onnx onnxruntime              # for mfibnn
pip install -e .                          # this repo
```

## Quick start

```python
import networkx as nx, mfib
G = nx.MultiDiGraph()
G.add_edge("u", "x", weight=1); G.add_edge("u", "x", weight=1); G.add_edge("u", "y", weight=2)
P = mfib.coarsest_equitable_partition(G)     # {u}, {x,y}: 1+1 = 2 splits/merges
f = mfib.minimum_base(G, P)                  # base: one arc labelled 2; f.node_map, f.arc_map
h = mfib.epsilon_partition(G, 0.5)           # certified eps-equitable partition
```

```bash
python3 -m mfibnn compress model.onnx --eps 0.4 --data test.npz
```

## Tests

```bash
python3 -m pytest tests/
```
