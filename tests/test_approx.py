import random

import networkx as nx
import pytest

import mfib
from mfib.approx import (beta_exact, defect, epsilon_partition, optimal_centers,
                         quotient_with_centers, unevenness)
from mfib import Partition


def chain():
    G = nx.MultiDiGraph()
    G.add_edge("u", "u", weight=10); G.add_edge("u", "x", weight=1)
    G.add_edge("u", "y", weight=1.2); G.add_edge("u", "z", weight=1.4)
    return G


def test_defect_of_exact_fibration_is_zero_and_of_bad_map_positive():
    G = nx.MultiDiGraph()
    G.add_edge("u", "x", weight=1); G.add_edge("u", "x", weight=1); G.add_edge("u", "y", weight=2)
    f = mfib.minimum_base(G)
    assert defect(f) == 0
    P = Partition({"u": 0, "x": 1, "y": 1})
    f2 = quotient_with_centers(G, P, {1: {0: 1.5}})     # label 1.5 instead of 2
    assert abs(defect(f2) - 0.5) < 1e-12
    D, dx = defect(f2, per_node=True)
    assert dx["x"] == pytest.approx(0.5) and dx["y"] == pytest.approx(0.5) and dx["u"] == 0


def test_chain_frontier_matches_note():
    G = chain()
    expected = {0.0: 4, 0.05: 4, 0.1: 3, 0.15: 3, 0.2: 2, 1.0: 2, 4.4: 2, 4.5: 1, 5.0: 1}
    for eps, k in expected.items():
        r = beta_exact(G, eps)
        assert r.value == k, (eps, r.value)
        assert r.defect <= eps + 1e-6
        assert len(epsilon_partition(G, eps, center="minimax")) == k


def test_heuristic_is_certified_and_upper_bound():
    rng = random.Random(3)
    for trial in range(15):
        n = rng.randint(3, 8)
        G = nx.MultiDiGraph()
        G.add_nodes_from(range(n))
        for _ in range(rng.randint(2, 3 * n)):
            G.add_edge(rng.randrange(n), rng.randrange(n), weight=round(rng.uniform(0.5, 3), 1))
        for eps in (0.0, 0.3, 1.0, 3.0):
            for center in ("mean", "minimax", "seed"):
                h = epsilon_partition(G, eps, center=center)
                f = quotient_with_centers(G, h.partition, h.centers)
                assert defect(f) <= eps + 1e-6              # certified
                assert max(h.radius.values()) <= eps + 1e-6
            r = beta_exact(G, eps)
            assert r.value <= len(epsilon_partition(G, eps, center="minimax"))
            assert r.defect <= eps + 1e-6
            assert unevenness(G, r.partition) <= eps + 1e-6


def test_beta_zero_is_minimum_base_size():
    rng = random.Random(5)
    for trial in range(10):
        n = rng.randint(3, 7)
        G = nx.MultiDiGraph(); G.add_nodes_from(range(n))
        for _ in range(rng.randint(2, 3 * n)):
            G.add_edge(rng.randrange(n), rng.randrange(n), weight=rng.choice([1, 1, 2]))
        r = beta_exact(G, 0.0)
        assert r.value == len(mfib.coarsest_equitable_partition(G))
        assert mfib.is_fibration(r.fibration)


def _all_partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for p in _all_partitions(rest):
        for i in range(len(p)):
            yield p[:i] + [[first] + p[i]] + p[i + 1:]
        yield [[first]] + p


@pytest.mark.parametrize("seed", range(12))
def test_beta_exact_against_brute_force(seed):
    rng = random.Random(seed)
    n = rng.randint(2, 5)
    G = nx.MultiDiGraph(); G.add_nodes_from(range(n))
    for _ in range(rng.randint(1, 2 * n)):
        G.add_edge(rng.randrange(n), rng.randrange(n), weight=rng.choice([1, 2, 3]))
    for eps in (0.0, 0.5, 1.0, 2.0):
        best = min(len(b) for b in _all_partitions(list(range(n)))
                   if unevenness(G, Partition({x: i for i, blk in enumerate(b) for x in blk})) <= eps + 1e-9)
        assert beta_exact(G, eps).value == best


def test_beta_monotone_in_eps():
    G = chain()
    vals = [beta_exact(G, e).value for e in (0, 0.1, 0.2, 1, 4.5)]
    assert vals == sorted(vals, reverse=True)
