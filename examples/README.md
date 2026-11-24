# Example graphs

Two classic undirected weighted networks, in the `source target weight`
format read by `python3 -m mfib` (pass `-u`, both graphs are undirected):

- `karate.txt`: Zachary's karate club (34 nodes, 78 edges); weights are
  interaction strengths between club members.
  W. W. Zachary, *An information flow model for conflict and fission in
  small groups*, Journal of Anthropological Research 33 (1977), 452-473.
- `lesmis.txt`: co-appearance network of the characters of Victor Hugo's
  *Les Misérables* (77 nodes, 254 edges); the weight of an edge is the
  number of chapters in which the two characters appear together.
  D. E. Knuth, *The Stanford GraphBase: A Platform for Combinatorial
  Computing*, Addison-Wesley, 1993.
- `southern.txt`: co-attendance projection of the Davis Southern Women
  two-mode network (18 women, 139 edges); the weight of an edge is the
  number of social events the two women attended together, out of 14
  events in Natchez, Mississippi, in the 1930s.
  A. Davis, B. B. Gardner and M. R. Gardner, *Deep South*, University of
  Chicago Press, 1941.

All files were exported from the copies bundled with
[networkx](https://networkx.org/):

```python
import networkx as nx
G = nx.les_miserables_graph()        # or nx.karate_club_graph()
for u, v, d in G.edges(data=True):
    print(u, v, d["weight"])
```

For the Southern Women graph, which networkx ships as an unweighted
bipartite women-events graph, the weighted projection was taken first
(and spaces removed from the names):

```python
from networkx.algorithms import bipartite
G = nx.davis_southern_women_graph()
women = [n for n, d in G.nodes(data=True) if d["bipartite"] == 0]
P = bipartite.weighted_projected_graph(G, women)
for u, v, d in P.edges(data=True):
    print(u.replace(" ", ""), v.replace(" ", ""), d["weight"])
```

## Try them

```bash
python3 -m mfib -u examples/lesmis.txt           # exact minimum fibration
python3 -m mfib -u --sweep examples/lesmis.txt   # certified class-count intervals
python3 -m mfib -u -e 10 examples/lesmis.txt     # a stable intermediate scale
```

At `eps = 0` the exact minimum fibration of the Les Misérables graph has 63
classes, and the non-singleton fibres are the structurally equivalent bit
players (for instance Napoleon, the Countess de Lô, Geborand, Champtercier,
Cravatte and the old man, who each appear in exactly one chapter with bishop
Myriel).  The sweep shows a wide plateau of at most 25 classes for
tolerances in roughly [7.2, 13.2), and at `eps = 10` the fibres become
thematic: Tholomyès with the students and the grisettes, the Champmathieu
trial (the three convicts, Champmathieu and the judge), and so on.

The Southern Women projection illustrates that fibrations capture *roles*,
not communities: its certified 2-class partition (`-u -e 17`) separates the
two social hubs, Evelyn Jefferson and Theresa Anderson, from the other
sixteen women, rather than reproducing the two social circles of the
classic analyses.  It also shows why the sweep reports running minima: the
raw class count of tolerant refinement is not monotone in the tolerance on
this graph (with mean centers it is 10 at `eps = 6`, 14 at `eps = 8`, 12 at
`eps = 10`), so if the count at your chosen tolerance is larger than the
certified bound, rerun at the breakpoint tolerance printed by the sweep:
that partition is valid at every larger tolerance too.
