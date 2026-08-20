"""Commutative monoids as label domains for M-graphs.

A monoid is specified by three things:

* ``zero``  -- the neutral element;
* ``add``   -- a binary associative, commutative operation;
* ``key``   -- a function sending an element to a *hashable canonical
  representative*; two elements are considered equal iff their keys are equal.
  For exact monoids ``key`` is the identity.  For floating-point sums it should
  round (e.g. ``lambda a: round(a, 9)``): partition refinement groups nodes by
  hashing aggregated weights, and without rounding ``0.1 + 0.2 != 0.3`` would
  split nodes spuriously.

Convenience: anywhere a monoid is expected you may pass a ``Monoid``, one of the
strings ``"N"``/``"additive"``, ``"R"``, ``"max"``, ``"min"``, ``"bool"``,
``"free"``, or ``None`` (= :data:`ADDITIVE`, addition with 9-digit rounding,
which handles ints, floats and Fractions sensibly).
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Iterable


@dataclass(frozen=True)
class Monoid:
    zero: Any
    add: Callable[[Any, Any], Any]
    key: Callable[[Any], Hashable] = field(default=lambda a: a)
    name: str = ""

    def sum(self, xs: Iterable[Any]) -> Any:
        acc = self.zero
        for x in xs:
            acc = self.add(acc, x)
        return acc

    def eq(self, a: Any, b: Any) -> bool:
        return self.key(a) == self.key(b)

    def is_zero(self, a: Any) -> bool:
        return self.key(a) == self.key(self.zero)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Monoid({self.name or 'anonymous'})"


def additive(ndigits: int | None = 9) -> Monoid:
    """(numbers, +, 0). ``ndigits=None`` means exact comparison (use for ints,
    Fractions, sympy expressions); otherwise sums are compared after rounding
    to ``ndigits`` decimals, so ``0.1 + 0.2`` and ``0.3`` coincide."""
    if ndigits is None:
        return Monoid(0, operator.add, name="(numbers,+,0) exact")
    return Monoid(0, operator.add, key=lambda a: round(a, ndigits),
                  name=f"(numbers,+,0) rounded to {ndigits} digits")


#: default label domain: addition of numbers, 9-digit rounding.
ADDITIVE = additive(9)
#: exact addition (natural numbers, integers, Fractions).
NAT = additive(None)
#: (R, +) with 9-digit rounding (alias of ADDITIVE).
REAL = ADDITIVE
#: (numbers, max, 0) -- for non-negative labels; use MAX_EXT for arbitrary reals.
MAX = Monoid(0, max, name="(numbers>=0,max,0)")
MAX_EXT = Monoid(float("-inf"), max, name="(R∪{-inf},max,-inf)")
#: (numbers, min, +inf) -- tropical.
MIN = Monoid(float("inf"), min, name="(R∪{+inf},min,+inf)")
#: (bool, or, False) -- reachability / unweighted "is there an arc".
BOOL = Monoid(False, lambda a, b: bool(a) or bool(b), key=bool, name="({0,1},or,0)")
#: (R^k, +) with l1-compatible rounding key: labels are numpy arrays (any shape,
#: consistent per arc type); the zero 0 broadcasts.  Use with ``mfib.l1``.
def _vec_key(a, ndigits=9):
    import numpy as np
    if np.ndim(a) == 0:
        return round(float(a), ndigits)
    return np.round(np.asarray(a, dtype=float), ndigits).tobytes()


VECTOR = Monoid(0, lambda a, b: a + b, key=_vec_key, name="(R^k,+,0) arrays, rounded")

#: free commutative monoid on labels: multisets, i.e. "count each label".
FREE = Monoid((), lambda a, b: tuple(sorted(a + b)),
              key=lambda a: tuple(sorted(a)), name="free commutative monoid (multisets)")

def cyclic(k: int) -> Monoid:
    """(Z_k, +, 0): integers modulo k. Labels may be any ints (reduced mod k).
    A metric monoid with the circular (Lee) metric ``mfib.circular_metric(k)``,
    d(a, b) = min((a-b) mod k, (b-a) mod k): on a group, translation-nonexpansive
    metrics are exactly the translation-invariant ones, i.e. the group norms."""
    if k < 1:
        raise ValueError("k must be >= 1")
    return Monoid(0, lambda a, b: (a + b) % k, key=lambda a: a % k, name=f"(Z_{k},+,0)")


_BY_NAME = {
    "n": NAT, "nat": NAT, "exact": NAT,
    "additive": ADDITIVE, "r": ADDITIVE, "real": ADDITIVE, "sum": ADDITIVE,
    "max": MAX, "min": MIN, "bool": BOOL, "or": BOOL, "free": FREE, "multiset": FREE,
    "vector": VECTOR, "vec": VECTOR, "kernel": VECTOR,
}


def resolve(spec: Monoid | str | None) -> Monoid:
    """Turn a monoid specification (Monoid, name, or None) into a Monoid."""
    if spec is None:
        return ADDITIVE
    if isinstance(spec, Monoid):
        return spec
    if isinstance(spec, str):
        try:
            return _BY_NAME[spec.strip().lower()]
        except KeyError:
            raise ValueError(f"unknown monoid name {spec!r}; known: {sorted(_BY_NAME)}") from None
    raise TypeError(f"cannot interpret {spec!r} as a monoid")
