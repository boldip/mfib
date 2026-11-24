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

With ``--sweep`` the script prints, instead of a partition, a summary of what
the computed partitions certify about the minimum number of classes as a
function of the tolerance.  An eps-equitable partition is eps'-equitable for
every eps' >= eps, so each computed partition bounds the minimum from its
tolerance on; the summary reports, per interval of tolerances, the best such
bound (``<= n``), and marks as ``exact`` the two provably exact ends: eps = 0
(the coarsest equitable partition is a true minimum) and the final collapse
to one class.  Breakpoints are tolerances where the output of tolerant
refinement was observed to drop, localized by bisection to 1% relative
precision; the true minimum may drop earlier than a reported breakpoint, and
partitions at different tolerances are in general not nested.  ``--raw``
prints the sampled ``eps classes`` pairs instead of the summary.

Examples::

    python3 -m mfib graph.txt                  # exact minimum base, directed
    python3 -m mfib --eps 0.5 graph.txt        # certified 0.5-approximate
    python3 -m mfib --sweep graph.txt          # certified class-count intervals
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


def sweep(G, center: str, raw: bool = False) -> None:
    """Print certified intervals of the class count over the tolerance axis.

    Samples tolerant refinement on a geometric grid (10 points per decade,
    from a thousandth of the inequity of the trivial partition, which always
    collapses everything by twice that value since any center rule places
    centers within the range of the observed values), then localizes each
    drop of the running-minimum count by bisection to 1% relative precision.
    Each sample at tolerance g with n classes certifies that at most n
    classes suffice for every tolerance >= g; only eps = 0 and the final
    collapse to one class are exact.
    """
    REL = 1.01  # relative precision of the breakpoints

    n0 = len(coarsest_equitable_partition(G, "R").blocks)
    if n0 == 1:
        print("[0, inf)  1  exact")
        return
    tot = dict(G.in_degree(weight="weight"))
    hi = (max(tot.values()) - min(tot.values())) / 2

    samples = {}

    def count(e: float) -> int:
        if e not in samples:
            samples[e] = len(epsilon_partition(G, e, "R", center=center).partition.blocks)
        return samples[e]

    eps = hi / 1000
    while eps <= 4 * hi:  # safety cap, never reached: 2*hi always collapses
        if count(eps) == 1:
            break
        eps *= 10 ** 0.1

    # localize each drop of the running minimum by geometric bisection
    while True:
        probe = None
        level, prev = n0, 0.0
        for e in sorted(samples):
            if samples[e] < level:
                if prev > 0 and e / prev > REL:
                    probe = (prev * e) ** 0.5
                    break
                level = samples[e]
            prev = e
        if probe is None:
            break
        count(probe)

    if raw:
        print(0, n0)
        for e in sorted(samples):
            print(f"{e:.6g}", samples[e])
        return

    drops, level = [], n0
    for e in sorted(samples):
        if samples[e] < level:
            level = samples[e]
            drops.append((e, level))

    print(f"[0, 0]  {n0}  exact")
    prev, level = 0.0, n0
    for e, n in drops:
        left = "(" if prev == 0.0 else "["
        print(f"{left}{prev:.4g}, {e:.4g})  <= {level}")
        prev, level = e, n
    if level == 1:
        print(f"[{prev:.4g}, inf)  1  exact")
    else:  # safety cap was hit, cannot happen
        print(f"[{prev:.4g}, inf)  <= {level}")


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
                    help="print certified intervals of the class count over the "
                         "tolerance axis instead of a partition (incompatible with --eps)")
    ap.add_argument("--raw", action="store_true",
                    help="with --sweep, print the sampled 'eps classes' pairs "
                         "instead of the interval summary")
    args = ap.parse_args(argv)

    if args.eps < 0:
        ap.error("--eps must be nonnegative")
    if args.sweep and args.eps > 0:
        ap.error("--sweep and --eps are mutually exclusive")
    if args.raw and not args.sweep:
        ap.error("--raw requires --sweep")

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
        sweep(G, args.center, raw=args.raw)
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
