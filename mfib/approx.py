"""Approximate fibrations: defects, epsilon-equitable partitions, beta_G(eps).

Theory (Robaccia/superfibration.tex, Sect. 9): for a metric monoid (M, d),
the nodewise defect of a morphism phi: G -> B is

    D(phi) = max_x  sum_{a into phi(x)}  d( sum of labels of the fibre of a at x , lambda(a) ),

an epsilon-fibration is a morphism with D <= eps, and

    beta_G(eps) = min{ |N_B| : G -> B epimorphic eps-fibration }
                = min{ |P| : P eps-equitable },

where P is eps-equitable iff every class D has a *center* l^D in M^P with
sum_C d(W(C,x), l^D_C) <= eps for all x in D; an optimal base is always the
quotient B_P labelled by the centers.  This module provides

* ``defect(f, ...)``                    -- D(phi) of any (Fibration-like) morphism;
* ``epsilon_partition(G, eps, ...)``    -- a certified eps-equitable partition with
                                           centers (tolerant refinement; upper bound on beta);
* ``quotient_with_centers(G, P, C)``    -- the witness B_P and the morphism G -> B_P;
* ``beta_exact(G, eps, ...)``           -- beta_G(eps) and an optimal witness by MILP
                                           (numeric labels, d = |u - v|; small graphs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence, Tuple

import networkx as nx

from .core import Fibration, Partition, _in_arcs, _normalize_initial
from .monoid import Monoid, resolve

Node = Hashable
Metric = Callable[[Any, Any], float]


def abs_diff(u, v) -> float:
    return abs(u - v)


def circular_metric(k: int):
    """The Lee (circular) metric on Z_k: d(a, b) = min((a-b) mod k, (b-a) mod k)."""
    def d(a, b) -> float:
        x = (a - b) % k
        return float(min(x, k - x))
    return d


def l1(u, v) -> float:
    """l1 distance; works for scalars and numpy arrays (and 0 broadcast against arrays)."""
    import numpy as np
    return float(np.abs(np.asarray(u, dtype=float) - np.asarray(v, dtype=float)).sum())


# --------------------------------------------------------------------------- #
# defect
# --------------------------------------------------------------------------- #

def defect(f: Fibration, metric: Metric = abs_diff, monoid: Monoid | str | None = None,
           weight: str = "weight", default=1, per_node: bool = False):
    """Nodewise defect D(phi) of the morphism ``f`` (a :class:`Fibration` object,
    which need not actually be a fibration).  With ``per_node=True`` also returns
    the dict ``{x: D_x}``."""
    M = resolve(monoid)
    G, B = f.G, f.B
    sums: Dict[Tuple[Tuple, Node], Any] = {}
    for (u, v, k), a in f.arc_map.items():
        sums[(a, v)] = M.add(sums.get((a, v), M.zero), G.edges[u, v, k].get(weight, default))
    arcs_into: Dict[Node, List[Tuple[Tuple, Any]]] = {}
    for bu, bv, bk, data in B.edges(keys=True, data=True):
        arcs_into.setdefault(bv, []).append(((bu, bv, bk), data.get(weight, default)))
    dx: Dict[Node, float] = {}
    for x in G.nodes:
        total = 0.0
        for a, lam in arcs_into.get(f.node_map[x], ()):
            total += metric(sums.get((a, x), M.zero), lam)
        dx[x] = total
    D = max(dx.values(), default=0.0)
    return (D, dx) if per_node else D


# --------------------------------------------------------------------------- #
# epsilon-equitable partitions (heuristic, certified)
# --------------------------------------------------------------------------- #

def _d1(M: Monoid, metric: Metric, u: Dict[int, Any], v: Dict[int, Any]) -> float:
    return sum(metric(u.get(c, M.zero), v.get(c, M.zero)) for c in set(u) | set(v))


def _mean_center(M: Monoid, feats: List[Dict[int, Any]]) -> Dict[int, Any]:
    keys = set().union(*feats) if feats else set()
    n = len(feats)
    return {c: sum(f.get(c, M.zero) for f in feats) / n for c in keys}


def _minimax_center(M: Monoid, feats: List[Dict[int, Any]]) -> Dict[int, float]:
    """l1 Chebyshev center of sparse numeric feature dicts (small LP)."""
    import numpy as np
    from scipy.optimize import linprog
    keys = sorted(set().union(*feats)) if feats else []
    if not keys or len(feats) == 1:
        return dict(feats[0]) if feats else {}
    m, J = len(feats), len(keys)
    nv = J + m * J + 1
    c = np.zeros(nv); c[-1] = 1.0
    A, b = [], []
    for i, f in enumerate(feats):
        row = np.zeros(nv); row[J + i * J: J + (i + 1) * J] = 1.0; row[-1] = -1.0
        A.append(row); b.append(0.0)
        for jj, C in enumerate(keys):
            w = float(f.get(C, 0.0))
            row = np.zeros(nv); row[jj] = -1.0; row[J + i * J + jj] = -1.0; A.append(row); b.append(-w)
            row = np.zeros(nv); row[jj] = 1.0; row[J + i * J + jj] = -1.0; A.append(row); b.append(w)
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b),
                  bounds=[(None, None)] * J + [(0, None)] * (m * J + 1), method="highs")
    return {C: float(res.x[jj]) for jj, C in enumerate(keys)}


def _is_numeric_additive(M: Monoid, metric: Metric) -> bool:
    """Scalar or array labels under elementwise addition with the l1 metric."""
    if metric is not abs_diff and metric is not l1:
        return False
    try:
        import numpy as np
        return (np.ndim(M.zero) == 0 and float(M.zero) == 0.0 and M.add(1.5, 2) == 3.5
                and bool(np.all(M.add(np.ones(2), np.ones(2)) == 2)))
    except Exception:
        return False


def _minimax_center_dense(F):
    """l1 Chebyshev center of the rows of the dense matrix F, by a sparse LP:
    variables l (J), t (m*J), r; minimize r s.t. sum_j t_ij <= r, |F_ij - l_j| <= t_ij."""
    import numpy as np
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    F = np.asarray(F, dtype=float)
    m, J = F.shape
    if m == 1:
        return F[0].copy()
    nv = J + m * J + 1
    c = np.zeros(nv); c[-1] = 1.0
    ii, jj, vv, b = [], [], [], []
    r = 0
    ar = np.arange(J)
    for i in range(m):
        # sum_j t_ij - r <= 0
        ii += [r] * (J + 1); jj += list(J + i * J + ar) + [nv - 1]; vv += [1.0] * J + [-1.0]; b.append(0.0); r += 1
        # -l_j - t_ij <= -F_ij
        ii += list(np.repeat(np.arange(r, r + J), 2)); jj += [x for k in range(J) for x in (k, J + i * J + k)]
        vv += [-1.0, -1.0] * J; b += list(-F[i]); r += J
        #  l_j - t_ij <= F_ij
        ii += list(np.repeat(np.arange(r, r + J), 2)); jj += [x for k in range(J) for x in (k, J + i * J + k)]
        vv += [1.0, -1.0] * J; b += list(F[i]); r += J
    A = coo_matrix((vv, (ii, jj)), shape=(r, nv)).tocsr()
    res = linprog(c, A_ub=A, b_ub=np.array(b), bounds=[(None, None)] * J + [(0, None)] * (m * J + 1),
                  method="highs")
    if res.x is None:
        return F.mean(0)
    return res.x[:J].copy()


def _iter_center_dense(F, iters=60):
    """Cheap approximate l1 minimax center: start at the mean, move a fraction
    1/(t+2) toward the currently farthest row, keep the best iterate."""
    import numpy as np
    l = F.mean(0)
    best, best_r = l.copy(), float(np.abs(F - l).sum(1).max())
    for t in range(iters):
        d = np.abs(F - l).sum(1)
        i = int(np.argmax(d))
        l = l + (F[i] - l) / (t + 2)
        r = float(np.abs(F - l).sum(1).max())
        if r < best_r:
            best, best_r = l.copy(), r
    return best


def _fmt_vec(v, nd=4):
    """Pretty-print a sparse feature vector {class: value}."""
    import numpy as np
    items = []
    for c in sorted(v, key=repr):
        x = v[c]
        s = (f"{x:.{nd}g}" if isinstance(x, (int, float)) and not isinstance(x, bool)
             else (np.array2string(np.asarray(x), precision=3, separator=",").replace("\n", "")
                   if hasattr(x, "shape") else repr(x)))
        items.append(f"{c}:{s}")
    return "{" + ", ".join(items) + "}"


def _fmt_nodes(nodes, maxn=12):
    nodes = list(nodes)
    s = ", ".join(map(str, nodes[:maxn]))
    return "{" + s + (", ..." if len(nodes) > maxn else "") + "}" + f" (n={len(nodes)})"


def _split_class_dense(members, feats, eps, tol, center_mode, log=None, class_id=None):
    """Greedy radius-eps grouping of one class, vectorized (numeric labels, l1).
    Returns a list of (group_nodes, center_dict, radius)."""
    import numpy as np
    keys = sorted(set().union(*(feats[y].keys() for y in members)), key=repr)
    if not keys:
        return [(list(members), {}, 0.0)]
    # label shape per key (array labels): width of the flattened block
    width = {}
    for y in members:
        for c, v in feats[y].items():
            if c not in width:
                width[c] = int(np.size(v))
    offs = {}
    tot = 0
    for c in keys:
        offs[c] = tot; tot += width.get(c, 1)
    F = np.zeros((len(members), tot))
    for i, y in enumerate(members):
        for c, v in feats[y].items():
            F[i, offs[c]: offs[c] + width[c]] = np.asarray(v, dtype=float).ravel()
    _, first, inverse = np.unique(np.round(F, 12), axis=0, return_index=True, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    twins = {int(r): [] for r in first}
    for i in range(len(members)):
        twins[int(first[inverse[i]])].append(i)
    remaining = [int(r) for r in first]
    groups = []
    last = None
    while remaining:
        R = np.array(remaining)
        if last is None:
            seed = remaining[0]
        else:
            seed = int(R[int(np.argmax(np.abs(F[R] - F[last]).sum(1)))])
        last = seed
        dseed = np.abs(F[R] - F[seed]).sum(1)
        name = lambda i: members[int(i)]
        if log:
            log(f"    seed s = {name(seed)}")
        if center_mode == "seed":
            grp = R[dseed <= eps + tol]
            ctr = F[seed].copy()
            if log:
                log(f"    group (nodes within eps of the seed's vector) = {_fmt_nodes(name(i) for i in grp)}")
        else:
            grp = R[dseed <= 2 * eps + tol]
            if log:
                log(f"    candidates A (d <= 2eps from seed) = {_fmt_nodes(name(i) for i in grp)}")
            while True:
                ctr = (F[grp].mean(0) if center_mode == "mean" else
                       _iter_center_dense(F[grp]) if center_mode == "iter" else
                       _minimax_center_dense(F[grp]))
                dd = np.abs(F[grp] - ctr).sum(1)
                if log:
                    cvec = {keys[k]: (float(ctr[offs[keys[k]]]) if width[keys[k]] == 1
                                      else ctr[offs[keys[k]]: offs[keys[k]] + width[keys[k]]])
                            for k in range(len(keys)) if np.any(ctr[offs[keys[k]]: offs[keys[k]] + width[keys[k]]] != 0)}
                    log(f"      center c = {_fmt_vec(cvec)}; max_x d(w_x, c) = {dd.max():.4g}" +
                        ("  <= eps: ok" if dd.max() <= eps + tol else "  > eps: shrink"))
                if dd.max() <= eps + tol:
                    others = R[~np.isin(R, grp)]
                    if len(others):
                        dothers = np.abs(F[others] - ctr).sum(1)
                        absorbed = others[dothers <= eps + tol]
                        if log and len(absorbed):
                            log(f"      absorb (within eps of c): {_fmt_nodes(name(i) for i in absorbed)}")
                        grp = np.concatenate([grp, absorbed])
                    break
                if len(grp) == 1:
                    ctr = F[seed].copy(); break
                mask = grp != seed
                far = grp[mask][int(np.argmax(dd[mask]))]
                if log:
                    log(f"      remove farthest x = {name(far)} (d = {dd[mask].max():.4g})")
                grp = grp[grp != far]
        radius = float(np.abs(F[grp] - ctr).sum(1).max())
        nodes = [members[j] for i in grp for j in twins[int(i)]]
        if log:
            log(f"    -> group {_fmt_nodes(nodes)}, radius {radius:.4g}")
        cdict = {}
        for c in keys:
            blk = ctr[offs[c]: offs[c] + width[c]]
            if np.any(blk != 0.0):
                shape = None
                for y in members:          # recover the array shape from any label of key c
                    if c in feats[y]:
                        shape = np.shape(feats[y][c]); break
                cdict[c] = float(blk[0]) if not shape else blk.reshape(shape).copy()
        groups.append((nodes, cdict, radius))
        gs = set(int(i) for i in grp)
        remaining = [i for i in remaining if i not in gs]
    return groups


@dataclass
class EpsPartition:
    """An eps-equitable partition with certified centers.

    ``partition``: the :class:`Partition`; ``centers[D]``: dict ``{C: l^D_C}`` (sparse,
    missing = zero); ``radius[D]``: max_x d_1(w_x, l^D) over x in D; ``eps``: the
    requested tolerance; ``rounds``: refinement rounds used."""
    partition: Partition
    centers: Dict[int, Dict[int, Any]]
    radius: Dict[int, float]
    eps: float
    rounds: int

    def __len__(self):
        return len(self.partition)


def epsilon_partition(G: nx.MultiDiGraph, eps: float, monoid: Monoid | str | None = None,
                      metric: Metric = abs_diff, center: str | Callable = "mean",
                      weight: str = "weight", default=1, initial=None,
                      max_rounds: Optional[int] = None, tol: float = 1e-9,
                      verbose=False) -> EpsPartition:
    """A certified eps-equitable partition (upper bound on beta_G(eps)) by
    *tolerant refinement*: start from the trivial (or ``initial``) partition; at
    each round compute the in-weight vectors w_x over the current classes and
    split each class greedily into groups admitting a center of radius <= eps
    (farthest-first seeds; ``center="mean"`` uses the coordinatewise mean,
    ``center="iter"`` a cheap iterative approximation of the l1 minimax center,
    ``center="minimax"`` the exact l1 Chebyshev center by an LP (expensive: only for
    small classes / low dimension),
    ``center="seed"`` always a member's vector, or pass a callable
    ``center(list_of_feature_dicts) -> dict``); stop at the first round in
    which nothing splits, which certifies the partition (its classes have radius <=
    eps with respect to their own aggregated features).  On acyclic graphs the
    result is reached in a single forward pass; in general it is a fixed point of
    the refinement, not a global optimum.

    ``verbose``: ``True`` prints a trace of the algorithm (rounds, current
    partition, vectors w_x, seeds, candidates, centers, shrinking, absorption,
    resulting groups) to stdout; a callable (e.g. ``logger.info``) receives the
    lines instead.
    """
    log = (print if verbose is True else verbose) if verbose else None
    M = resolve(monoid)
    nodes = list(G.nodes)
    block = _normalize_initial(G, initial)
    ins = _in_arcs(G, weight, default)
    fast = _is_numeric_additive(M, metric) and center in ("mean", "iter", "minimax", "seed")
    if center == "mean":
        center_fn = lambda feats: _mean_center(M, feats)
    elif center == "minimax":
        center_fn = lambda feats: _minimax_center(M, feats)
    elif center == "iter":
        center_fn = lambda feats: _mean_center(M, feats)   # generic path: mean
    elif center == "seed":
        center_fn = None
    else:
        center_fn = center
    rounds = 0
    if log:
        log(f"epsilon_partition: eps = {eps}, center rule = {center}, {len(nodes)} nodes, "
            f"{'dense (numpy)' if fast else 'generic'} path")
    while True:
        rounds += 1
        feats: Dict[Node, Dict[int, Any]] = {}
        for x in nodes:
            acc: Dict[int, Any] = {}
            for u, lab in ins[x]:
                c = block[u]
                acc[c] = M.add(acc.get(c, M.zero), lab)
            feats[x] = {c: v for c, v in acc.items() if not M.is_zero(v)}
        classes: Dict[int, List[Node]] = {}
        for x in nodes:
            classes.setdefault(block[x], []).append(x)
        if log:
            log(f"=== round {rounds}: current partition Pi has {len(classes)} classes")
            for cid, members in sorted(classes.items()):
                log(f"  class {cid}: {_fmt_nodes(members)}")
            log("  vectors w_x = (W(C,x))_C over the current classes:")
            for x in nodes:
                log(f"    w_{x} = {_fmt_vec(feats[x])}")
        new_block: Dict[Node, int] = {}
        centers: Dict[int, Dict[int, Any]] = {}
        radius: Dict[int, float] = {}
        nid = 0
        if fast:
            for cid, members in sorted(classes.items()):
                if log:
                    log(f"  splitting class {cid} = {_fmt_nodes(members)}")
                for nodes_g, ctr, rad in _split_class_dense(members, feats, eps, tol, center, log, cid):
                    for y in nodes_g:
                        new_block[y] = nid
                    centers[nid] = ctr; radius[nid] = rad; nid += 1
            classes_items = []
        else:
            classes_items = sorted(classes.items())
        for cid, members in classes_items:
            if log:
                log(f"  splitting class {cid} = {_fmt_nodes(members)}")
            # nodes with identical feature vectors always stay together: cluster
            # one representative per signature, then copy the assignment
            sig_of = {y: tuple((c, M.key(v)) for c, v in sorted(feats[y].items(), key=lambda kv: repr(kv[0])))
                      for y in members}
            reps: Dict[Any, Node] = {}
            twins: Dict[Node, List[Node]] = {}
            for y in members:
                r0 = reps.setdefault(sig_of[y], y)
                twins.setdefault(r0, []).append(y)
            remaining = list(reps.values())
            last_seed = None
            while remaining:
                if last_seed is None:
                    seed = remaining[0]
                else:  # farthest-first
                    seed = max(remaining, key=lambda y: _d1(M, metric, feats[y], feats[last_seed]))
                last_seed = seed
                if log:
                    log(f"    seed s = {seed}")
                # candidates: sharing a center of radius eps with the seed requires d <= 2 eps
                group = [y for y in remaining if _d1(M, metric, feats[y], feats[seed]) <= 2 * eps + tol]
                if log:
                    log(f"    candidates A (d <= 2eps from seed) = {_fmt_nodes(group)}")
                ctr = feats[seed]
                if center_fn is not None:
                    while True:
                        try:
                            cand = center_fn([feats[y] for y in group])
                        except TypeError:   # non-numeric labels: seed ball only
                            group = [y for y in group if _d1(M, metric, feats[y], ctr) <= eps + tol]
                            if log:
                                log(f"      center rule not applicable: using the seed's vector; group = {_fmt_nodes(group)}")
                            break
                        dists = {y: _d1(M, metric, feats[y], cand) for y in group}
                        if log:
                            log(f"      center c = {_fmt_vec(cand)}; max_x d(w_x, c) = {max(dists.values()):.4g}" +
                                ("  <= eps: ok" if max(dists.values()) <= eps + tol else "  > eps: shrink"))
                        if max(dists.values()) <= eps + tol:
                            ctr = cand
                            extra = [y for y in remaining if y not in dists
                                     and _d1(M, metric, feats[y], cand) <= eps + tol]
                            if log and extra:
                                log(f"      absorb (within eps of c): {_fmt_nodes(extra)}")
                            group += extra
                            break
                        far = max((y for y in group if y != seed), key=dists.get, default=None)
                        if far is None:
                            group = [seed]
                            break
                        if log:
                            log(f"      remove farthest x = {far} (d = {dists[far]:.4g})")
                        group.remove(far)
                else:
                    group = [y for y in group if _d1(M, metric, feats[y], ctr) <= eps + tol]
                    if log:
                        log(f"      center = seed's vector {_fmt_vec(ctr)}; group = {_fmt_nodes(group)}")
                for y in group:
                    for yy in twins[y]:
                        new_block[yy] = nid
                centers[nid] = dict(ctr)
                radius[nid] = max(_d1(M, metric, feats[y], ctr) for y in group)
                if log:
                    log(f"    -> group {_fmt_nodes(group)}, center {_fmt_vec(ctr)}, radius {radius[nid]:.4g}")
                nid += 1
                gs = set(group)
                remaining = [y for y in remaining if y not in gs]
        stable = nid == len(classes)
        if log:
            log(f"  round {rounds} produced {nid} classes " +
                ("== |Pi|: stable, stop (certificate holds)" if stable else f"> |Pi| = {len(classes)}: refine and repeat"))
        if stable:
            # classes unchanged: the centers/radii computed above refer to features over
            # the *same* partition, so the certificate is valid; keep the old ids order
            order = {}
            final_block = {x: order.setdefault(block[x], len(order)) for x in nodes}
            remap = {}
            for x in nodes:
                remap[final_block[x]] = new_block[x]
            centers = {k: {order[c]: v for c, v in centers[remap[k]].items()} for k in order.values()}
            radius = {k: radius[remap[k]] for k in order.values()}
            return EpsPartition(Partition(final_block), centers, radius, eps, rounds)
        block = new_block
        if max_rounds is not None and rounds >= max_rounds:
            break
    # not stabilized: recompute a certified answer for the final partition by
    # one more (splitting) pass -- guaranteed to terminate since it only refines
    return epsilon_partition(G, eps, M, metric, center, weight, default, Partition(block), None, tol, verbose)


def optimal_centers(G: nx.MultiDiGraph, partition: Partition, weight: str = "weight",
                    default=1) -> Tuple[Dict[int, Dict[int, float]], Dict[int, float]]:
    """For numeric labels and d = |u - v|: the minimax (Chebyshev, l1) center of
    every class, by one small LP per class.  Returns ``(centers, radius)``; the
    unevenness of the partition is ``max(radius.values())``."""
    import numpy as np

    M = resolve("R")
    ins = _in_arcs(G, weight, default)
    feats: Dict[Node, Dict[int, float]] = {}
    for x in G.nodes:
        acc: Dict[int, float] = {}
        for u, lab in ins[x]:
            c = partition.block_of[u]
            acc[c] = acc.get(c, 0.0) + float(lab)
        feats[x] = acc
    centers: Dict[int, Dict[int, float]] = {}
    radius: Dict[int, float] = {}
    for D, members in enumerate(partition.blocks):
        keys = sorted(set().union(*(feats[x].keys() for x in members)))
        if not keys:
            centers[D], radius[D] = {}, 0.0
            continue
        F = np.array([[feats[x].get(C, 0.0) for C in keys] for x in members], dtype=float)
        ctr = _minimax_center_dense(F)
        centers[D] = {C: float(ctr[jj]) for jj, C in enumerate(keys)}
        radius[D] = float(np.abs(F - ctr).sum(1).max())
    return centers, radius


def unevenness(G: nx.MultiDiGraph, partition: Partition, weight: str = "weight", default=1) -> float:
    """u(P) for numeric labels and d = |u - v| (exact, via :func:`optimal_centers`)."""
    _, radius = optimal_centers(G, partition, weight, default)
    return max(radius.values(), default=0.0)


def quotient_with_centers(G: nx.MultiDiGraph, partition: Partition,
                          centers: Dict[int, Dict[int, Any]], monoid=None,
                          weight: str = "weight", default=1) -> Fibration:
    """The quotient B_P with arc C -> D (iff G(C, D) nonempty) labelled by the
    center coordinate ``centers[D][C]`` (zero if absent), and the canonical
    morphism G -> B_P (an eps-fibration iff the centers have radius <= eps)."""
    M = resolve(monoid)
    B = nx.MultiDiGraph()
    for i, members in enumerate(partition.blocks):
        B.add_node(i, members=tuple(members))
    pairs = set()
    arc_map = {}
    for u, v, k in G.edges(keys=True):
        c, d = partition.block_of[u], partition.block_of[v]
        pairs.add((c, d))
        arc_map[(u, v, k)] = (c, d, 0)
    for c, d in sorted(pairs, key=repr):
        B.add_edge(c, d, key=0, **{weight: centers.get(d, {}).get(c, M.zero)})
    return Fibration(G, B, dict(partition.block_of), arc_map)


# --------------------------------------------------------------------------- #
# exact beta_G(eps) by MILP (numeric labels, d = |u - v|)
# --------------------------------------------------------------------------- #

@dataclass
class BetaResult:
    value: int                       # beta_G(eps)
    partition: Partition
    centers: Dict[int, Dict[int, float]]
    fibration: Fibration             # witness G -> B_P
    defect: float                    # verified nodewise defect of the witness
    status: str


def beta_exact(G: nx.MultiDiGraph, eps: float, kmax: Optional[int] = None,
               weight: str = "weight", default=1, time_limit: Optional[float] = None,
               tol: float = 1e-7, verbose: bool = False) -> BetaResult:
    """beta_G(eps) for numeric labels and d(u, v) = |u - v|, with an optimal
    witness, by mixed-integer linear programming (scipy / HiGHS).

    Variables: z[x,k] (node x in class k), y[k] (class k used), l[k,j] (label of
    the arc j -> k of the base), t[x,j] >= |W(C_j, x) - l[k(x), j]|.  Since
    W(C_j, x) = sum_u w(u -> x) z[u,j] is linear in z, all constraints are
    linear (big-M linking of l to the class of x).  Minimizes sum_k y[k].
    Size ~ n^2 K constraints: fine up to a few dozen nodes.  ``kmax`` bounds
    the number of classes (default: the size of the heuristic solution).
    """
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    nodes = list(G.nodes)
    n = len(nodes)
    idx = {x: i for i, x in enumerate(nodes)}
    heur = epsilon_partition(G, eps, "R", abs_diff, "mean", weight, default)
    K = min(len(heur), kmax) if kmax is not None else len(heur)
    K = max(K, 1)
    win: List[List[Tuple[int, float]]] = [[] for _ in range(n)]   # x -> [(u, w)]
    for u, v, data in G.edges(data=True):
        win[idx[v]].append((idx[u], float(data.get(weight, default))))
    wmax = max((sum(abs(w) for _, w in win[x]) for x in range(n)), default=0.0)
    signed = any(w < 0 for x in range(n) for _, w in win[x])
    lo, hi = (-wmax, wmax) if signed else (0.0, wmax)
    Mbig = 2 * wmax + 1.0

    # variable layout
    def Z(x, k): return x * K + k
    def Y(k): return n * K + k
    def L(k, j): return n * K + K + k * K + j
    def T(x, j): return n * K + K + K * K + x * K + j
    nvar = n * K + K + K * K + n * K

    rows, cols, vals, lb, ub = [], [], [], [], []
    r = 0

    def add(coefs, low, high):
        nonlocal r
        for c, v in coefs:
            rows.append(r); cols.append(c); vals.append(v)
        lb.append(low); ub.append(high); r += 1

    for x in range(n):                                   # each node in one class
        add([(Z(x, k), 1.0) for k in range(K)], 1.0, 1.0)
    for x in range(n):                                   # z <= y
        for k in range(min(K, x + 1)):
            add([(Z(x, k), 1.0), (Y(k), -1.0)], -np.inf, 0.0)
    for k in range(K - 1):                               # y non-increasing (symmetry)
        add([(Y(k + 1), 1.0), (Y(k), -1.0)], -np.inf, 0.0)
    for x in range(n):                                   # |W_j(x) - l[k,j]| <= t[x,j] if z[x,k]
        for j in range(K):
            wcoefs = [(Z(u, j), w) for u, w in win[x] if j <= u]
            for k in range(min(K, x + 1)):
                add(wcoefs + [(L(k, j), -1.0), (T(x, j), -1.0), (Z(x, k), Mbig)], -np.inf, Mbig)
                add([(c, -v) for c, v in wcoefs] + [(L(k, j), 1.0), (T(x, j), -1.0), (Z(x, k), Mbig)],
                    -np.inf, Mbig)
    for x in range(n):                                   # sum_j t[x,j] <= eps
        add([(T(x, j), 1.0) for j in range(K)], -np.inf, eps + tol)

    A = coo_matrix((vals, (rows, cols)), shape=(r, nvar)).tocsr()
    c = np.zeros(nvar)
    for k in range(K):
        c[Y(k)] = 1.0
    vlb = np.zeros(nvar); vub = np.full(nvar, np.inf)
    integrality = np.zeros(nvar)
    for x in range(n):
        for k in range(K):
            vub[Z(x, k)] = 1.0 if k <= x else 0.0
            integrality[Z(x, k)] = 1
    for k in range(K):
        vub[Y(k)] = 1.0; integrality[Y(k)] = 1
        for j in range(K):
            vlb[L(k, j)] = lo; vub[L(k, j)] = hi
    options = {"disp": verbose}
    if time_limit is not None:
        options["time_limit"] = time_limit
    res = milp(c, constraints=LinearConstraint(A, lb, ub), integrality=integrality,
               bounds=Bounds(vlb, vub), options=options)
    if res.x is None:
        raise RuntimeError(f"MILP failed: {res.message}")
    xsol = res.x
    block = {}
    for x in range(n):
        k = int(np.argmax([xsol[Z(x, kk)] for kk in range(K)]))
        block[nodes[x]] = k
    order: Dict[int, int] = {}
    part = Partition({x: order.setdefault(block[x], len(order)) for x in nodes})
    centers, _ = optimal_centers(G, part, weight, default)   # polish: minimax centers
    f = quotient_with_centers(G, part, centers, "R", weight, default)
    D = defect(f, abs_diff, "R", weight, default)
    return BetaResult(int(round(res.fun)), part, centers, f, D,
                      "optimal" if res.status == 0 else res.message)
