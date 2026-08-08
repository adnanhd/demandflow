"""Bottom-up dynamic programming as pull-mode dataflow.

Bottom-up DP fills a table from base cases upward. With DemandGraph, each
subproblem is a node whose inputs are smaller subproblems. Demanding the
answer pulls only the subproblems actually needed, and the MemoStore
provides automatic memoization (the DP table).

Example: edit distance between two strings.
"""

from demandflow.demand import DemandGraph


def edit_distance(s: str, t: str) -> int:
    g = DemandGraph()
    n, m = len(s), len(t)

    # Base cases are passed as kwargs to demand().
    # Each cell node's param names match the topic names of its dependencies.
    base_kwargs = {}
    for i in range(n + 1):
        base_kwargs["d_%d_0" % i] = i
    for j in range(1, m + 1):
        base_kwargs["d_0_%d" % j] = j

    # Register each non-base cell as a node
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            cell = "d_%d_%d" % (i, j)
            above = "d_%d_%d" % (i - 1, j)
            left = "d_%d_%d" % (i, j - 1)
            diag = "d_%d_%d" % (i - 1, j - 1)

            def make_fn(a_key, l_key, dg_key, c):
                def fn(**kw):
                    return min(kw[a_key] + 1, kw[l_key] + 1, kw[dg_key] + c)
                return fn

            g.node(
                cell,
                inputs={above: above, left: left, diag: diag},
                fn=make_fn(above, left, diag, cost),
                outs=[cell],
            )

    target = "d_%d_%d" % (n, m)
    if target in base_kwargs:
        return base_kwargs[target]
    return g.demand(target, **base_kwargs)


def main():
    cases = [
        ("kitten", "sitting", 3),
        ("saturday", "sunday", 3),
        ("", "abc", 3),
        ("abc", "abc", 0),
        ("intention", "execution", 5),
    ]
    for s, t, expected in cases:
        dist = edit_distance(s, t)
        print("edit_distance(%r, %r) = %d (expected %d)" % (s, t, dist, expected))
        assert dist == expected, "mismatch: got %d" % dist
    print("Bottom-up DP (pull-mode) verified.")


if __name__ == "__main__":
    main()
