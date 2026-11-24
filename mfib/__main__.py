"""Command-line interface: minimum base of a generic weighted graph.

Reads a graph as whitespace-separated lines

    source target weight

(weights are positive reals; blank lines and lines starting with ``#`` are
ignored; parallel edges are allowed and node names are arbitrary tokens),
computes an (epsilon-approximate) M-equitable partition over the monoid
(R, +, 0), and prints one line

    node class

per node.  With eps = 0 (the default) ``class`` is the index of the fibre of
the minimum fibration the node belongs to (the coarsest equitable partition,
which is unique).  With eps > 0 it is the index of the fibre of a certified
epsilon-approximate fibration computed by tolerant refinement with the chosen
center rule; this is an upper bound on the minimum number of fibres, not
necessarily the minimum (and a minimum epsilon-approximate fibration need not
be unique).  Classes are numbered consecutively in order of first appearance
of their nodes in the input.

With ``--sweep`` the script prints, instead of a partition, one line

    eps classes

per value of a geometric grid of tolerances (starting from the exact count at
eps = 0 and stopping as soon as a single class is reached), so that one can
pick a tolerance by looking for wide plateaus of the class count.

Examples::

    python3 -m mfib graph.txt                  # exact minimum base, directed
    python3 -m mfib --eps 0.5 graph.txt        # certified 0.5-approximate
    python3 -m mfib --sweep graph.txt          # class count over a grid of eps
    cat graph.txt | python3 -m mfib --undirected
"""

import argparse
import sys

import networkx as nx

from .approx import epsilon_partition
from .core import coarsest_equitable_partition


def read_graph(stream, undirected: bool) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for lineno, line in enumerate(stream, 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 3:
            raise SystemExit(f"line {lineno}: expected 'source target weight', got {line!r}")
        u, v, w = parts
        try:
            weight = float(w)
        except ValueError:
            raise SystemExit(f"line {lineno}: weight {w!r} is not a number") from None
        if weight <= 0:
            raise SystemExit(f"line {lineno}: weight must be a positive real, got {w}")
        G.add_edge(u, v, weight=weight)
        if undirected and u != v:
            G.add_edge(v, u, weight=weight)
    return G


def sweep(G, center: str) -> None:
    """Print 'eps classes' lines over a geometric grid of tolerances.

    Starts with the exact count at eps = 0, then walks a geometric grid
    (10 points per decade) from a thousandth of the inequity of the trivial
    partition, stopping as soon as the count reaches 1.  Any center rule
    places centers within the range of the observed values, so the count is
    guaranteed to reach 1 by twice that inequity.
    """
    exact = coarsest_equitable_partition(G, "R")
    print(0, len(exact.blocks))
    if len(exact.blocks) == 1:
        return
    tot = dict(G.in_degree(weight="weight"))
    hi = (max(tot.values()) - min(tot.values())) / 2
    eps = hi / 1000
    while eps <= 4 * hi:  # safety cap, never reached: 2*hi always collapses
        n = len(epsilon_partition(G, eps, "R", center=center).partition.blocks)
        print(f"{eps:.6g}", n)
        if n == 1:
            return
        eps *= 10 ** 0.1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python3 -m mfib",
        description="Reads 'source target weight' lines and prints 'node class' lines: "
                    "with --eps 0 (default) class is the fibre of the minimum fibration; "
                    "with --eps > 0, of a certified eps-approximate fibration "
                    "(not necessarily minimum).")
    ap.add_argument("input", nargs="?", default=None,
                    help="input file (default: standard input)")
    ap.add_argument("-u", "--undirected", action="store_true",
                    help="treat the graph as undirected (each edge becomes two opposite arcs)")
    ap.add_argument("-e", "--eps", type=float, default=0.0, metavar="EPS",
                    help="tolerance: compute a certified EPS-approximate M-equitable "
                         "partition instead of the exact one (default: 0, exact)")
    ap.add_argument("-c", "--center", default="mean",
                    choices=["mean", "iter", "minimax", "seed"],
                    help="center rule used when --eps > 0 (default: mean)")
    ap.add_argument("-s", "--sweep", action="store_true",
                    help="print 'eps classes' lines over a geometric grid of "
                         "tolerances instead of a partition (incompatible with --eps)")
    args = ap.parse_args(argv)

    if args.eps < 0:
        ap.error("--eps must be nonnegative")
    if args.sweep and args.eps > 0:
        ap.error("--sweep and --eps are mutually exclusive")

    if args.input:
        try:
            stream = open(args.input)
        except OSError as e:
            raise SystemExit(f"cannot open {args.input!r}: {e.strerror}")
    else:
        stream = sys.stdin
    try:
        G = read_graph(stream, args.undirected)
    finally:
        if args.input:
            stream.close()

    if G.number_of_nodes() == 0:
        return 0

    if args.sweep:
        sweep(G, args.center)
        return 0

    if args.eps > 0:
        partition = epsilon_partition(G, args.eps, "R", center=args.center).partition
    else:
        partition = coarsest_equitable_partition(G, "R")

    # number classes by first appearance of their nodes in input order
    renumber = {}
    for x in G.nodes:  # insertion order
        b = partition.block_of[x]
        if b not in renumber:
            renumber[b] = len(renumber)
    for x in G.nodes:
        print(x, renumber[partition.block_of[x]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
