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

Examples::

    python3 -m mfib graph.txt                  # exact minimum base, directed
    python3 -m mfib --eps 0.5 graph.txt        # certified 0.5-approximate
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
    args = ap.parse_args(argv)

    if args.eps < 0:
        ap.error("--eps must be nonnegative")

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
