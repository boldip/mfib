# mfib — M-fibrations of monoid-labelled graphs

Python library and tools for **M-fibrations**: fibrations of directed multigraphs whose
arcs are labelled in a commutative monoid, as developed in the paper
[*"M-Fibration Theory with Applications to Weighted Graphs"*](https://arxiv.org/abs/2608.25598)
(P. Boldi, O. M. Velarde, H. A. Makse). It generalizes graph fibrations
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
* **`mfibnn`** — a self-contained tool that reduces a model in **ONNX** format to a
  certified ε-approximate M-fibration quotient and evaluates the result:
  `python3 -m mfibnn {info|compress|eval}`. See `mfibnn/README.md` for an
  end-to-end guide (including where to download networks and data).

## Install

```bash
pip install networkx numpy scipy          # core
pip install onnx onnxruntime              # for mfibnn (onnx>=1.20 writes IR 14: needs a recent onnxruntime, 1.32+ works)
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

## Background and credits

The use of fibration symmetries to reduce machine-learning models originates in the
work of the Makse group and coauthors:

* O. M. Velarde, L. C. Parra, P. Boldi, H. A. Makse.
  *The role of fibration symmetries in geometric deep learning.*
  Proc. Natl. Acad. Sci. USA 123(4), 2026.
* O. M. Velarde, L. C. Parra, A. Hashemi, H. A. Makse.
  *Emergence of fibrations, compression, and symmetry breaking in artificial neural
  networks.* [arXiv:2609.01768](https://arxiv.org/abs/2609.01768), 2026 — ε-fibers of
  weighted graphs, the mean-weight compression rule, and their application to model
  compression.

This repository implements the general algebraic theory (minimum bases of graphs
labelled over arbitrary commutative monoids, and the certified ε-approximate theory);
for the model-compression methodology please refer to — and cite — the papers above.

## Command line: minimum base of a weighted graph

`python3 -m mfib` reads a graph as `source target weight` lines (weights are
positive reals; `#` starts a comment) from a file or standard input, and prints
`node class` lines. With `--eps 0` (the default) `class` identifies the fibre
of the minimum fibration the node belongs to (the coarsest equitable
partition); with `--eps > 0` it identifies the fibre of a certified
epsilon-approximate fibration computed by tolerant refinement, which is not
guaranteed to have the minimum number of fibres (and a minimum
epsilon-approximate fibration need not even be unique):

```bash
python3 -m mfib graph.txt                 # exact minimum base, directed
python3 -m mfib --undirected graph.txt    # each edge read as two opposite arcs
python3 -m mfib --eps 0.5 graph.txt       # certified 0.5-approximate partition
cat graph.txt | python3 -m mfib -e 1 -c minimax   # with the l1 Chebyshev center rule
```

## Tests

```bash
python3 -m pytest tests/
```
