"""Coarsest M-equitable partitions, minimum bases and minimum M-fibrations.

Conventions (following the paper):

* an M-graph is an ``nx.MultiDiGraph`` whose arcs carry a label in the monoid
  in the edge attribute ``weight`` (configurable); a missing label defaults to
  ``1``, so a plain multidigraph is a "normal" graph;
* labels may be ``0``; the quotient ``G/Pi`` has an arc ``C -> D`` iff ``G`` has
  at least one arc from ``C`` to ``D``, labelled by the total weight ``W(C, D)``;
* ``W(C, x)`` is the total label of the arcs from the set ``C`` into ``x``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Hashable, Iterable, List, Mapping, Optional, Tuple

import networkx as nx

from .monoid import Monoid, resolve

Node = Hashable
ArcKey = Tuple[Node, Node, Hashable]


# --------------------------------------------------------------------------- #
# partitions
# --------------------------------------------------------------------------- #

@dataclass
class Partition:
    """A partition of the node set: ``block_of[x]`` is the index of the block
    containing ``x``; ``blocks[i]`` is the ``i``-th block as a list of nodes."""
    block_of: Dict[Node, int]
    blocks: List[List[Node]] = field(default_factory=list)

    def __post_init__(self):
        if not self.blocks:
            n = 1 + max(self.block_of.values(), default=-1)
            self.blocks = [[] for _ in range(n)]
            for x, i in self.block_of.items():
                self.blocks[i].append(x)

    def __len__(self) -> int:
        return len(self.blocks)

    def __iter__(self):
        return iter(self.blocks)

    def __getitem__(self, i: int) -> List[Node]:
        return self.blocks[i]

    def same_block(self, x: Node, y: Node) -> bool:
        return self.block_of[x] == self.block_of[y]

    def is_finer_than(self, other: "Partition") -> bool:
        return all(other.same_block(b[0], x) for b in self.blocks for x in b)

    def as_sets(self) -> List[frozenset]:
        return [frozenset(b) for b in self.blocks]

    def __repr__(self) -> str:  # pragma: no cover
        return f"Partition({len(self)} blocks: {[sorted(map(str, b)) for b in self.blocks]})"


def _normalize_initial(G: nx.MultiDiGraph, initial) -> Dict[Node, int]:
    """Accept: None (trivial partition), a dict node -> colour, an iterable of
    node collections, a callable node -> colour, or a node-attribute name."""
    nodes = list(G.nodes)
    if initial is None:
        return {x: 0 for x in nodes}
    if isinstance(initial, Partition):
        return dict(initial.block_of)
    if isinstance(initial, str):
        colours = {x: G.nodes[x].get(initial) for x in nodes}
    elif callable(initial):
        colours = {x: initial(x) for x in nodes}
    elif isinstance(initial, Mapping):
        colours = {x: initial.get(x) for x in nodes}
    else:  # iterable of collections
        colours = {}
        for i, block in enumerate(initial):
            for x in block:
                colours[x] = i
        for x in nodes:
            colours.setdefault(x, None)
    ids: Dict[Any, int] = {}
    return {x: ids.setdefault(colours[x], len(ids)) for x in nodes}


def _in_arcs(G: nx.MultiDiGraph, weight: str, default) -> Dict[Node, List[Tuple[Node, Any]]]:
    ins: Dict[Node, List[Tuple[Node, Any]]] = {x: [] for x in G.nodes}
    for u, v, data in G.edges(data=True):
        ins[v].append((u, data.get(weight, default)))
    return ins


def in_weights(G: nx.MultiDiGraph, partition: Partition, x: Node, monoid=None,
               weight: str = "weight", default=1) -> Dict[int, Any]:
    """``{block index C: W(C, x)}`` for the blocks with at least one arc into x."""
    M = resolve(monoid)
    acc: Dict[int, Any] = {}
    for u, v, data in G.in_edges(x, data=True):
        c = partition.block_of[u]
        acc[c] = M.add(acc.get(c, M.zero), data.get(weight, default))
    return acc


# --------------------------------------------------------------------------- #
# coarsest equitable partition
# --------------------------------------------------------------------------- #

def coarsest_equitable_partition(G: nx.MultiDiGraph, monoid: Monoid | str | None = None,
                                 weight: str = "weight", default=1, initial=None,
                                 max_rounds: Optional[int] = None) -> Partition:
    """The coarsest M-equitable partition of ``G`` (refining ``initial`` if given).

    Refinement from the top: at each round two nodes stay together iff they
    were together and receive the same total weight from every current block.
    Stabilizes in at most ``|N_G| - 1`` rounds; naive ``O(rounds * |A_G|)``.

    ``initial`` may be a dict node -> colour, an iterable of node sets, a
    callable, a node-attribute name, or a ``Partition``: its blocks are never
    merged (use it to freeze inputs/outputs of a network, or for coloured graphs).
    """
    M = resolve(monoid)
    nodes = list(G.nodes)
    block = _normalize_initial(G, initial)
    ins = _in_arcs(G, weight, default)
    zero_key = M.key(M.zero)
    rounds = 0
    while True:
        ids: Dict[Any, int] = {}
        new_block: Dict[Node, int] = {}
        for x in nodes:
            acc: Dict[int, Any] = {}
            for u, lab in ins[x]:
                c = block[u]
                acc[c] = M.add(acc.get(c, M.zero), lab)
            sig = (block[x], tuple((c, M.key(acc[c])) for c in sorted(acc)
                                   if M.key(acc[c]) != zero_key))
            new_block[x] = ids.setdefault(sig, len(ids))
        stable = len(ids) == len(set(block.values()))
        block = new_block
        rounds += 1
        if stable or (max_rounds is not None and rounds >= max_rounds):
            break
    # renumber blocks in order of first appearance for readability
    order: Dict[int, int] = {}
    block = {x: order.setdefault(block[x], len(order)) for x in nodes}
    return Partition(block)


def is_equitable(G: nx.MultiDiGraph, partition: Partition, monoid=None,
                 weight: str = "weight", default=1) -> bool:
    """Check that ``W(C, x) = W(C, y)`` whenever ``x, y`` share a block."""
    M = resolve(monoid)
    ref: Dict[int, Dict[int, Hashable]] = {}
    for x in G.nodes:
        w = in_weights(G, partition, x, M, weight, default)
        sig = {c: M.key(v) for c, v in w.items() if not M.is_zero(v)}
        b = partition.block_of[x]
        if b in ref:
            if ref[b] != sig:
                return False
        else:
            ref[b] = sig
    return True


# --------------------------------------------------------------------------- #
# quotients, minimum base, fibrations
# --------------------------------------------------------------------------- #

@dataclass
class Fibration:
    """An M-fibration ``G -> B`` given by its node and arc components.

    ``node_map[x]`` is the node of ``B`` under ``x``; ``arc_map[(u, v, key)]``
    is the arc ``(u', v', key')`` of ``B`` under the arc ``(u, v, key)`` of ``G``.
    """
    G: nx.MultiDiGraph
    B: nx.MultiDiGraph
    node_map: Dict[Node, Node]
    arc_map: Dict[ArcKey, ArcKey]

    def fibre(self, b: Node) -> List[Node]:
        return [x for x, y in self.node_map.items() if y == b]

    def __call__(self, x: Node) -> Node:
        return self.node_map[x]


def quotient(G: nx.MultiDiGraph, partition: Partition, monoid=None,
             weight: str = "weight", default=1, check: bool = True) -> Fibration:
    """The quotient ``G/Pi`` and the canonical projection.

    ``G/Pi`` is a simple ``MultiDiGraph`` (all keys ``0``): node ``i`` for the
    ``i``-th block, with attribute ``members``; an arc ``i -> j`` labelled
    ``W(C_i, C_j)`` iff ``G`` has an arc from ``C_i`` to ``C_j``.  If ``check``
    is true, raises ``ValueError`` unless the partition is M-equitable (in
    which case the projection is an epimorphic M-fibration).
    """
    M = resolve(monoid)
    if check and not is_equitable(G, partition, M, weight, default):
        raise ValueError("partition is not M-equitable; the quotient map would not be an M-fibration")
    B = nx.MultiDiGraph()
    for i, members in enumerate(partition.blocks):
        B.add_node(i, members=tuple(members))
    # W(C, D) is the common value of W(C, x) for x in D: take a representative x0
    # of each block D; an arc C -> D exists iff *some* arc of G goes from C to D.
    rep_in: Dict[Tuple[int, int], Any] = {}   # (C, D) -> W(C, x0_D)
    pairs = set()
    arc_map: Dict[ArcKey, ArcKey] = {}
    rep = {i: members[0] for i, members in enumerate(partition.blocks) if members}
    is_rep = {x: i for i, x in rep.items()}
    for u, v, k, data in G.edges(keys=True, data=True):
        c, d = partition.block_of[u], partition.block_of[v]
        pairs.add((c, d))
        arc_map[(u, v, k)] = (c, d, 0)
        if is_rep.get(v) == d:
            rep_in[(c, d)] = M.add(rep_in.get((c, d), M.zero), data.get(weight, default))
    for c, d in sorted(pairs, key=repr):
        B.add_edge(c, d, key=0, **{weight: rep_in.get((c, d), M.zero)})
    return Fibration(G, B, dict(partition.block_of), arc_map)


def minimum_base(G: nx.MultiDiGraph, partition: Optional[Partition] = None, monoid=None,
                 weight: str = "weight", default=1, initial=None) -> Fibration:
    """The minimum base of ``G`` and the minimum M-fibration onto it.

    If ``partition`` is omitted it is computed by
    :func:`coarsest_equitable_partition` (with ``initial`` if given).  The
    returned :class:`Fibration` ``f`` has ``f.B`` = minimum base (simple, prime),
    ``f.node_map`` = the coarsest partition, ``f.arc_map`` = arc component.
    """
    M = resolve(monoid)
    if partition is None:
        partition = coarsest_equitable_partition(G, M, weight, default, initial)
    return quotient(G, partition, M, weight, default, check=False)


def is_fibration(f: Fibration, monoid=None, weight: str = "weight", default=1,
                 require_morphism: bool = True) -> bool:
    """Check the M-fibration condition: for every arc ``a`` of ``B`` and node
    ``x`` of ``G`` over ``t(a)``, the labels of the arcs of ``G`` over ``a`` into
    ``x`` sum to ``lambda(a)``.  With ``require_morphism`` also checks that the
    maps commute with sources and targets and are total."""
    M = resolve(monoid)
    G, B = f.G, f.B
    if require_morphism:
        if set(f.node_map) != set(G.nodes):
            return False
        for (u, v, k), (bu, bv, bk) in f.arc_map.items():
            if not B.has_edge(bu, bv, key=bk):
                return False
            if f.node_map[u] != bu or f.node_map[v] != bv:
                return False
        if len(f.arc_map) != G.number_of_edges():
            return False
    # accumulate fibre sums per (base arc, target node)
    sums: Dict[Tuple[ArcKey, Node], Any] = {}
    for (u, v, k), a in f.arc_map.items():
        lab = G.edges[u, v, k].get(weight, default)
        sums[(a, v)] = M.add(sums.get((a, v), M.zero), lab)
    fibres: Dict[Node, List[Node]] = {}
    for x, b in f.node_map.items():
        fibres.setdefault(b, []).append(x)
    for bu, bv, bk, data in B.edges(keys=True, data=True):
        lam = data.get(weight, default)
        for x in fibres.get(bv, ()):
            if not M.eq(sums.get(((bu, bv, bk), x), M.zero), lam):
                return False
    return True


def is_prime(G: nx.MultiDiGraph, monoid=None, weight: str = "weight", default=1) -> bool:
    """``G`` is M-fibration prime iff its coarsest equitable partition is discrete
    and it has no parallel arcs."""
    if any(G.number_of_edges(u, v) > 1 for u, v in set(G.edges())):
        return False
    return len(coarsest_equitable_partition(G, monoid, weight, default)) == G.number_of_nodes()
