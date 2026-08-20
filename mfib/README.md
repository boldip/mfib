# mfib — M-fibrations of monoid-labelled graphs

Python companion to the theory in `LaTeX/paper.tex` / `Robaccia/superfibration.tex`.
Depends only on `networkx`.

```python
import networkx as nx, mfib

G = nx.MultiDiGraph()
G.add_edge("u", "x", weight=1); G.add_edge("u", "x", weight=1); G.add_edge("u", "y", weight=2)

P = mfib.coarsest_equitable_partition(G)     # Partition(2 blocks: [['u'], ['x','y']])
f = mfib.minimum_base(G, P)                   # or simply mfib.minimum_base(G)
f.B                                           # the minimum base: MultiDiGraph, node 0 -> node 1 labelled 2
f.node_map, f.arc_map                         # the minimum M-fibration G -> f.B
mfib.is_fibration(f)                          # True
mfib.is_prime(f.B)                            # True
```

## Conventions

* An M-graph is an `nx.MultiDiGraph` with the label in the edge attribute
  `weight` (change with `weight=...`); a **missing label defaults to 1**, so a
  plain multidigraph is a "normal" graph and `minimum_base(G, monoid="N")` is the
  classical minimum base with parallel arcs re-encoded as counts.
* Labels may be `0`. The quotient `G/Π` has an arc `C -> D` iff `G` has an arc from
  `C` to `D`, labelled by `W(C, D)`; the base is a *simple* `MultiDiGraph` (all keys 0),
  whose node `i` carries the attribute `members` (the `i`-th block).
* `initial=` freezes classes that must never be merged (dict node -> colour,
  iterable of node sets, callable, node-attribute name, or a `Partition`): use it
  for input/output nodes of a network, or for arc-coloured graphs encoded on nodes.

## Specifying the monoid

Pass `monoid=` a `Monoid`, a name, or nothing:

| spec | meaning |
|---|---|
| `None` (default) / `"R"` / `"additive"` | numbers under `+`, compared after rounding to 9 decimals (`mfib.ADDITIVE`) |
| `"N"` / `"exact"` | numbers under `+`, exact comparison (`mfib.NAT`; use for ints, `Fraction`s) |
| `"max"`, `"min"`, `"bool"`, `"free"` | `(x,max,0)`, tropical `(x,min,+inf)`, `({0,1},or,0)`, multisets of labels |
| `mfib.additive(ndigits)` | addition with a chosen rounding (`None` = exact) |
| `mfib.Monoid(zero, add, key=..., name=...)` | anything else |

A `Monoid` is `(zero, add, key)`: `key` maps an element to a hashable canonical
representative and **defines equality** (`a == b` iff `key(a) == key(b)`). Refinement
groups nodes by hashing their aggregated in-weights, so for floating point you want a
rounding `key` (rounding, not a tolerance: `key` must be a function). For exact
monoids `key` is the identity.

## Functions

* `coarsest_equitable_partition(G, monoid=None, weight="weight", default=1, initial=None) -> Partition`
  — refinement from the trivial (or `initial`) partition; O(rounds · |A|), rounds ≤ |N|−1.
* `minimum_base(G, partition=None, monoid=None, ...) -> Fibration` — the quotient by the
  coarsest partition and the canonical projection.
* `quotient(G, partition, ...)` — same for any equitable partition (raises if it isn't).
* `is_equitable(G, partition, ...)`, `is_fibration(f, ...)`, `is_prime(G, ...)` — checkers.
* `in_weights(G, partition, x, ...)` — `{block: W(block, x)}`.

`Partition`: `.blocks`, `.block_of`, `same_block(x,y)`, `is_finer_than(other)`, `len()`.
`Fibration`: `.G`, `.B`, `.node_map`, `.arc_map` (`(u,v,key) -> (u',v',key')`), `.fibre(b)`.

## Tests

`python3 -m pytest tests/` — includes a brute-force check that the computed partition is
the coarsest equitable one on random small graphs.

## Approximate fibrations (`mfib.approx`)

For a metric monoid $(M,d)$ (default: numbers with $d=|u-v|$), following §9 of the note:

* `defect(f, metric=abs_diff, ...)` — nodewise defect $D(\varphi)=\max_x\sum_{a\to\varphi(x)} d(\text{fibre sum},\lambda(a))$
  of any morphism given as a `Fibration` object (`per_node=True` also returns the per-node values).
* `epsilon_partition(G, eps, metric=..., center="mean"|"minimax"|"seed"|callable, initial=...) -> EpsPartition`
  — a **certified ε-equitable partition** (every class has a center of radius ≤ ε w.r.t. its own
  aggregated in-weights) by tolerant refinement; `.partition`, `.centers`, `.radius`. Polynomial;
  an **upper bound** on $\beta_G(\varepsilon)$, not the optimum. `center="minimax"` (exact ℓ¹ Chebyshev
  centers by small LPs) is slower and better than `"mean"`.
  Pass `verbose=True` (or a callable such as `logger.info`) to trace the algorithm: each
  round's partition and vectors $w_x$, and for every class the seed, candidates, center,
  shrinking/absorb steps and resulting group — the variables of Algorithm 1 of the paper.
* `quotient_with_centers(G, P, centers)` — the witness base $B_P$ (arc $C\to D$ labelled by the center
  coordinate) and the morphism $G\to B_P$; `defect(...)` of it is ≤ ε when the centers are certified.
* `optimal_centers(G, P)`, `unevenness(G, P)` — exact minimax centers / $u(P)$ for a given partition
  (numeric labels, ℓ¹; one LP per class).
* `beta_exact(G, eps, kmax=None, time_limit=None) -> BetaResult` — **exact** $\beta_G(\varepsilon)$ with an
  optimal witness (`.value`, `.partition`, `.centers`, `.fibration`, `.defect`), by MILP (scipy/HiGHS;
  numeric labels, $d=|u-v|$). Since $\beta_G(\varepsilon)=p_G(\varepsilon)$ and optimal bases are labelled
  quotients, the search is over partitions only; ~$n^2K$ constraints, fine up to a few dozen nodes.
  Complexity of $\beta_G$ in general: conjecturally NP-hard (unproved).

```python
r = mfib.beta_exact(G, 0.2)        # r.value = beta_G(0.2); r.fibration.B = a witness B
h = mfib.epsilon_partition(G, 0.2, center="minimax")   # certified upper bound, any size
f = mfib.quotient_with_centers(G, h.partition, h.centers); mfib.defect(f) <= 0.2
```

## Example: $(\mathbb Z_k,+)$

```python
Z5 = mfib.cyclic(5)                       # integers mod 5; labels reduced mod 5
d5 = mfib.circular_metric(5)              # Lee metric d(a,b) = min((a-b)%5, (b-a)%5) — a group norm
P  = mfib.coarsest_equitable_partition(G, monoid=Z5)
h  = mfib.epsilon_partition(G, 1, monoid=Z5, metric=d5, center="seed")   # certified w.r.t. d5
```
On a group every translation-nonexpansive metric is translation-invariant, i.e. a group
norm; `circular_metric(k)` is the natural one. Note the zero-sums: `2 + 3 = 0` in Z_5.
