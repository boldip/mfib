"""Tests for the command-line interface (python3 -m mfib)."""

import io

import pytest

from mfib.__main__ import main, read_graph

# graph H from the paper: exact classes {0},{1},{2},{3,4};
# at eps=1 the classes become {0,1},{2},{3,4}; at eps>=13.5 everything merges
H = """\
# graph H
0 1 5
1 0 6
0 2 15
1 2 23
2 3 30
2 4 30
3 0 8
3 0 2
4 1 9
"""


def classes(out: str):
    m = {}
    for line in out.strip().splitlines():
        node, cls = line.split()
        m.setdefault(cls, set()).add(node)
    return sorted(frozenset(s) for s in m.values())


def run(argv, stdin_text, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(stdin_text))
    assert main(argv) == 0
    return capsys.readouterr().out


def test_exact(capsys, monkeypatch):
    out = run([], H, capsys, monkeypatch)
    assert classes(out) == sorted([frozenset("0"), frozenset("1"), frozenset("2"),
                                   frozenset({"3", "4"})])


def test_eps(capsys, monkeypatch):
    out = run(["--eps", "1"], H, capsys, monkeypatch)
    assert classes(out) == sorted([frozenset({"0", "1"}), frozenset("2"),
                                   frozenset({"3", "4"})])
    out = run(["--eps", "13.5"], H, capsys, monkeypatch)
    assert classes(out) == [frozenset({"0", "1", "2", "3", "4"})]


def test_file_input(tmp_path, capsys):
    f = tmp_path / "H.txt"
    f.write_text(H)
    assert main([str(f), "-e", "1"]) == 0
    assert len(classes(capsys.readouterr().out)) == 3


def test_undirected(capsys, monkeypatch):
    # triangle with equal weights: a single class
    out = run(["--undirected"], "a b 1\nb c 1\nc a 1\n", capsys, monkeypatch)
    assert classes(out) == [frozenset({"a", "b", "c"})]
    # path: endpoints together, middle alone
    out = run(["-u"], "a b 2\nb c 2\n", capsys, monkeypatch)
    assert classes(out) == sorted([frozenset({"a", "c"}), frozenset("b")])
    # directed, the same path splits the endpoints (a has no in-arcs)
    out = run([], "a b 2\nb c 2\n", capsys, monkeypatch)
    assert len(classes(out)) == 3


def test_read_graph_parallel_and_loop():
    G = read_graph(io.StringIO("a a 1\na b 2\na b 3\n"), undirected=True)
    assert G.number_of_edges() == 5  # loop once, each a-b edge twice


def test_errors():
    with pytest.raises(SystemExit):
        read_graph(io.StringIO("a b\n"), undirected=False)
    with pytest.raises(SystemExit):
        read_graph(io.StringIO("a b zero\n"), undirected=False)
    with pytest.raises(SystemExit):
        read_graph(io.StringIO("a b 0\n"), undirected=False)
    with pytest.raises(SystemExit):
        main(["/nonexistent-file"])
