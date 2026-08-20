import itertools
import random
from fractions import Fraction

import networkx as nx
import pytest

import mfib
from mfib import (BOOL, FREE, MAX, NAT, Monoid, Partition, coarsest_equitable_partition,
                  is_equitable, is_fibration, is_prime, minimum_base, quotient)


def mdg(arcs, weight="weight"):
    G = nx.MultiDiGraph()
    for a in arcs:
        if len(a) == 2:
            G.add_edge(*a)
        else:
            u, v, w = a
            G.add_edge(u, v, **{weight: w})
    return G


# ---------------------------------------------------------------- basic examples

def test_split_merge_example():
    # u -> x twice with 1, u -> y once with 2: x and y merge; base has one arc labelled 2
    G = mdg([("u", "x", 1), ("u", "x", 1), ("u", "y", 2)])
    P = coarsest_equitable_partition(G)
    assert P.same_block("x", "y") and not P.same_block("u", "x")
    f = minimum_base(G, P)
    assert f.B.number_of_nodes() == 2 and f.B.number_of_edges() == 1
    (bu, bv, k, d), = f.B.edges(keys=True, data=True)
    assert d["weight"] == 2
    assert is_fibration(f)
    assert is_prime(f.B)


def test_normal_graph_matches_classical_minimum_base():
    # bidirectional 4-cycle: classical minimum base is one node with two loops;
    # in the M-world (N,+): one node with a single loop labelled 2
    G = nx.MultiDiGraph()
    for i in range(4):
        G.add_edge(i, (i + 1) % 4)
        G.add_edge((i + 1) % 4, i)
    f = minimum_base(G, monoid=NAT)
    assert f.B.number_of_nodes() == 1
    (u, v, k, d), = f.B.edges(keys=True, data=True)
    assert u == v and d["weight"] == 2
    assert is_fibration(f, NAT)


def test_generic_real_weights_are_prime():
    rng = random.Random(1)
    G = nx.MultiDiGraph()
    for u in range(5):
        for v in range(5):
            if u != v:
                G.add_edge(u, v, weight=rng.random())
    P = coarsest_equitable_partition(G)
    assert len(P) == 5
    assert is_prime(G)


def test_float_rounding_default_monoid():
    # 0.1 + 0.2 vs 0.3: with the default (rounded) monoid these nodes merge
    G = mdg([("u", "x", 0.1), ("u", "x", 0.2), ("u", "y", 0.3)])
    assert coarsest_equitable_partition(G).same_block("x", "y")
    assert not coarsest_equitable_partition(G, monoid=mfib.additive(None)).same_block("x", "y")


def test_zero_labels_and_zero_sums():
    # (Z,+): arcs 3 and -3 into x sum to 0; y has an arc labelled 0 from the same block
    G = mdg([("u", "x", 3), ("u", "x", -3), ("u", "y", 0)])
    P = coarsest_equitable_partition(G, monoid=NAT)
    assert P.same_block("x", "y")
    f = minimum_base(G, P, monoid=NAT)
    assert f.B.number_of_edges() == 1 and list(f.B.edges(data=True))[0][2]["weight"] == 0
    assert is_fibration(f, NAT)


def test_initial_partition_is_never_merged():
    G = mdg([("u", "x", 1), ("u", "y", 1)])
    assert coarsest_equitable_partition(G).same_block("x", "y")
    P = coarsest_equitable_partition(G, initial={"x": "out1", "y": "out2", "u": "in"})
    assert not P.same_block("x", "y")
    assert is_equitable(G, P)


def test_other_monoids():
    G = mdg([("u", "x", 2), ("u", "x", 5), ("u", "y", 5), ("u", "y", 1)])
    assert coarsest_equitable_partition(G, monoid=MAX).same_block("x", "y")   # max = 5 both
    assert not coarsest_equitable_partition(G, monoid=NAT).same_block("x", "y")  # 7 vs 6
    assert coarsest_equitable_partition(G, monoid=BOOL).same_block("x", "y")
    G2 = mdg([("u", "x", "a"), ("u", "x", "b"), ("u", "y", "b"), ("u", "y", "a")])
    G3 = mdg([("u", "x", "a"), ("u", "x", "b"), ("u", "y", "a"), ("u", "y", "a")])
    free = Monoid((), lambda a, b: tuple(sorted(a + b)), key=lambda a: tuple(sorted(a)))
    G2f = mdg([(u, v, (w,)) for u, v, w in [e[:2] + (e[2]["weight"],) for e in G2.edges(data=True)]])
    G3f = mdg([(u, v, (w,)) for u, v, w in [e[:2] + (e[2]["weight"],) for e in G3.edges(data=True)]])
    assert coarsest_equitable_partition(G2f, monoid=free).same_block("x", "y")
    assert not coarsest_equitable_partition(G3f, monoid=free).same_block("x", "y")


def test_string_monoid_specs():
    G = mdg([("u", "x", 1), ("u", "y", 1)])
    for spec in ["N", "R", "additive", "max", "bool"]:
        assert coarsest_equitable_partition(G, monoid=spec).same_block("x", "y")
    with pytest.raises(ValueError):
        coarsest_equitable_partition(G, monoid="nonsense")


def test_quotient_rejects_non_equitable():
    G = mdg([("u", "x", 1), ("u", "y", 2)])
    P = Partition({"u": 0, "x": 1, "y": 1})
    with pytest.raises(ValueError):
        quotient(G, P)


def test_missing_weight_defaults_to_one_and_custom_attribute():
    G = nx.MultiDiGraph([("u", "x"), ("u", "x"), ("u", "y"), ("u", "y")])
    assert coarsest_equitable_partition(G).same_block("x", "y")
    G2 = mdg([("u", "x", 2), ("u", "y", 2)], weight="w")
    assert coarsest_equitable_partition(G2, weight="w").same_block("x", "y")


def test_base_of_base_is_base_and_prime():
    G = mdg([("a", "b", 1), ("a", "c", 1), ("b", "d", 2), ("c", "d", 2), ("d", "a", 1), ("d", "a", 1)])
    f = minimum_base(G)
    assert is_fibration(f)
    assert is_prime(f.B)
    g = minimum_base(f.B)
    assert g.B.number_of_nodes() == f.B.number_of_nodes()
    assert g.B.number_of_edges() == f.B.number_of_edges()


# ---------------------------------------------------------------- brute force

def _all_partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for p in _all_partitions(rest):
        for i in range(len(p)):
            yield p[:i] + [[first] + p[i]] + p[i + 1:]
        yield [[first]] + p


def _to_partition(blocks):
    return Partition({x: i for i, b in enumerate(blocks) for x in b})


@pytest.mark.parametrize("seed", range(40))
def test_coarsest_against_brute_force(seed):
    rng = random.Random(seed)
    n = rng.randint(2, 6)
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(n))
    for _ in range(rng.randint(0, 3 * n)):
        G.add_edge(rng.randrange(n), rng.randrange(n), weight=rng.choice([1, 1, 1, 2, 3]))
    P = coarsest_equitable_partition(G, monoid=NAT)
    assert is_equitable(G, P, NAT)
    equitable = [_to_partition(b) for b in _all_partitions(list(range(n))) if is_equitable(G, _to_partition(b), NAT)]
    for Q in equitable:                      # every equitable partition refines P
        assert Q.is_finer_than(P)
    assert any(len(Q) == len(P) for Q in equitable)  # and P is one of them
    f = minimum_base(G, P, monoid=NAT)
    assert is_fibration(f, NAT) and is_prime(f.B, NAT)


@pytest.mark.parametrize("seed", range(20))
def test_fractions_exact(seed):
    rng = random.Random(seed)
    G = nx.MultiDiGraph()
    for _ in range(12):
        G.add_edge(rng.randrange(4), rng.randrange(4), weight=Fraction(rng.randint(1, 4), rng.randint(1, 4)))
    P = coarsest_equitable_partition(G, monoid=NAT)
    f = minimum_base(G, P, monoid=NAT)
    assert is_fibration(f, NAT)


def test_cyclic_monoid():
    Z5 = mfib.cyclic(5)
    # u -> x with labels 2, 4 (sum 6 = 1 mod 5); u -> y with label 1; u -> z with label 6 (= 1)
    G = mdg([("u", "x", 2), ("u", "x", 4), ("u", "y", 1), ("u", "z", 6)])
    P = coarsest_equitable_partition(G, monoid=Z5)
    assert P.same_block("x", "y") and P.same_block("y", "z")
    f = minimum_base(G, P, monoid=Z5)
    assert is_fibration(f, Z5)
    (bu, bv, k, d), = f.B.edges(keys=True, data=True)
    assert d["weight"] % 5 == 1
    # zero-sums: 2 + 3 = 0 mod 5 merges with a node receiving nothing... only if arcs exist
    G2 = mdg([("u", "x", 2), ("u", "x", 3), ("u", "y", 0)])
    assert coarsest_equitable_partition(G2, monoid=Z5).same_block("x", "y")
    # circular metric: 0 and 4 are at distance 1 in Z_5, 1 and 4 at distance 2
    d5 = mfib.circular_metric(5)
    assert d5(0, 4) == 1 and d5(1, 4) == 2 and d5(0, 2) == 2 and d5(3, 3) == 0
    G3 = mdg([("u", "x", 0), ("u", "y", 4)])
    assert not coarsest_equitable_partition(G3, monoid=Z5).same_block("x", "y")
    h = mfib.epsilon_partition(G3, 1, monoid=Z5, metric=d5, center="seed")
    assert h.partition.same_block("x", "y")
    fq = mfib.quotient_with_centers(G3, h.partition, h.centers, monoid=Z5)
    assert mfib.defect(fq, metric=d5, monoid=Z5) <= 1
