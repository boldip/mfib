"""mfib -- M-fibrations of monoid-labelled multidigraphs.

>>> import networkx as nx, mfib
>>> G = nx.MultiDiGraph()
>>> _ = G.add_edge("u", "x", weight=1), G.add_edge("u", "x", weight=1), G.add_edge("u", "y", weight=2)
>>> P = mfib.coarsest_equitable_partition(G)          # {u}, {x, y}
>>> f = mfib.minimum_base(G, P)                        # base: 0 -> 1 labelled 2
>>> mfib.is_fibration(f)
True
"""

from .monoid import (ADDITIVE, BOOL, FREE, MAX, MAX_EXT, MIN, NAT, REAL, VECTOR, Monoid,
                     additive, cyclic, resolve)
from .core import (Fibration, Partition, coarsest_equitable_partition, in_weights,
                   is_equitable, is_fibration, is_prime, minimum_base, quotient)
from .approx import (BetaResult, EpsPartition, abs_diff, beta_exact, circular_metric, defect,
                     epsilon_partition, l1, optimal_centers, quotient_with_centers, unevenness)

__all__ = [
    "Monoid", "additive", "cyclic", "resolve", "ADDITIVE", "VECTOR", "l1", "circular_metric", "NAT", "REAL", "MAX", "MAX_EXT", "MIN", "BOOL", "FREE",
    "Partition", "Fibration", "coarsest_equitable_partition", "is_equitable", "in_weights",
    "quotient", "minimum_base", "is_fibration", "is_prime",
    "defect", "epsilon_partition", "EpsPartition", "quotient_with_centers", "optimal_centers",
    "unevenness", "beta_exact", "BetaResult", "abs_diff",
]
__version__ = "0.1.0"
