"""Automatic differentiation via operator overloading on DemandGraph.

Each elementary operation is a class with forward() and backward() methods.
Var overloads +, *, sin, exp to build an expression tree of Op instances.
Calling f.grad(x, y) walks the tree, registers adjoint nodes, and returns
gradient Vars -- no manual backward wiring needed.

Same function as 01_vjp: f(x, y) = (x*y + sin(x)) * exp(y).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Set, Tuple

import numpy as np

from demandflow.demand import DemandGraph


# ---------------------------------------------------------------------------
# Op classes: each knows how to register forward and backward nodes
# ---------------------------------------------------------------------------

class Op:
    """Base class for differentiable operations."""

    def forward(self, g: DemandGraph, *inputs: str) -> str:
        raise NotImplementedError

    def backward(self, g: DemandGraph, grad_out: str, *input_topics: str) -> List[str]:
        """Register backward nodes. Return one grad contribution topic per input."""
        raise NotImplementedError


class Add(Op):
    def forward(self, g: DemandGraph, a: str, b: str) -> str:
        out = "add_%s_%s" % (a, b)
        g.node(out, inputs={a: a, b: b},
               fn=lambda **kw: kw[a] + kw[b], outs=[out])
        return out

    def backward(self, g: DemandGraph, grad_out: str, a: str, b: str) -> List[str]:
        ga = "d_%s_d_%s" % (grad_out, a)
        gb = "d_%s_d_%s" % (grad_out, b)
        g.node(ga, inputs={grad_out: grad_out},
               fn=lambda **kw: kw[grad_out], outs=[ga])
        g.node(gb, inputs={grad_out: grad_out},
               fn=lambda **kw: kw[grad_out], outs=[gb])
        return [ga, gb]


class Mul(Op):
    def forward(self, g: DemandGraph, a: str, b: str) -> str:
        out = "mul_%s_%s" % (a, b)
        g.node(out, inputs={a: a, b: b},
               fn=lambda **kw: kw[a] * kw[b], outs=[out])
        return out

    def backward(self, g: DemandGraph, grad_out: str, a: str, b: str) -> List[str]:
        ga = "d_%s_d_%s" % (grad_out, a)
        gb = "d_%s_d_%s" % (grad_out, b)
        g.node(ga, inputs={grad_out: grad_out, b: b},
               fn=lambda **kw: kw[grad_out] * kw[b], outs=[ga])
        g.node(gb, inputs={grad_out: grad_out, a: a},
               fn=lambda **kw: kw[grad_out] * kw[a], outs=[gb])
        return [ga, gb]


class Sin(Op):
    def forward(self, g: DemandGraph, a: str) -> str:
        out = "sin_%s" % a
        g.node(out, inputs={a: a},
               fn=lambda **kw: np.sin(kw[a]), outs=[out])
        return out

    def backward(self, g: DemandGraph, grad_out: str, a: str) -> List[str]:
        ga = "d_%s_d_%s" % (grad_out, a)
        g.node(ga, inputs={grad_out: grad_out, a: a},
               fn=lambda **kw: kw[grad_out] * np.cos(kw[a]), outs=[ga])
        return [ga]


class Exp(Op):
    def forward(self, g: DemandGraph, a: str) -> str:
        out = "exp_%s" % a
        g.node(out, inputs={a: a},
               fn=lambda **kw: np.exp(kw[a]), outs=[out])
        return out

    def backward(self, g: DemandGraph, grad_out: str, a: str) -> List[str]:
        ga = "d_%s_d_%s" % (grad_out, a)
        g.node(ga, inputs={grad_out: grad_out, a: a},
               fn=lambda **kw: kw[grad_out] * np.exp(kw[a]), outs=[ga])
        return [ga]


# ---------------------------------------------------------------------------
# Var: thin wrapper that builds an expression tree of Ops
# ---------------------------------------------------------------------------

class Var:
    """Differentiable variable backed by a DemandGraph topic."""

    def __init__(
        self,
        graph: DemandGraph,
        topic: str,
        op: Op | None = None,
        parents: List[Var] | None = None,
    ) -> None:
        self.g: DemandGraph = graph
        self.topic: str = topic
        self.op: Op | None = op
        self.parents: List[Var] = parents or []

    def _apply(self, op_cls: type[Op], *others: Var) -> Var:
        op = op_cls()
        input_topics: List[str] = [self.topic] + [o.topic for o in others]
        out_topic: str = op.forward(self.g, *input_topics)
        return Var(self.g, out_topic, op=op, parents=[self] + list(others))

    def __add__(self, other: Var) -> Var:
        return self._apply(Add, other)

    def __mul__(self, other: Var) -> Var:
        return self._apply(Mul, other)

    def sin(self) -> Var:
        return self._apply(Sin)

    def exp(self) -> Var:
        return self._apply(Exp)

    def grad(self, *wrt: Var) -> Var | Tuple[Var, ...]:
        """Build backward pass and return gradient Var(s) for the given inputs.

        >>> df_dx, df_dy = f.grad(x, y)
        >>> g.demand(df_dx.topic, x=1.5, y=2.0, seed=1.0)
        """
        grad_topics: Dict[str, str] = _backward(self)
        results: List[Var] = []
        for v in wrt:
            topic = grad_topics.get(v.topic)
            if topic is None:
                raise ValueError(
                    "no gradient path from %s to %s" % (self.topic, v.topic))
            results.append(Var(self.g, topic))
        return results[0] if len(results) == 1 else tuple(results)


def _pass_fn(k: str) -> Callable[..., float]:
    return lambda **kw: kw[k]


def _sum_fn(keys: List[str]) -> Callable[..., float]:
    return lambda **kw: sum(kw[k] for k in keys)


def _backward(result: Var) -> Dict[str, str]:
    """Walk the expression tree, call each Op.backward(), accumulate, return grad map."""
    g: DemandGraph = result.g

    # reverse topo order
    order: List[Var] = []
    visited: Set[str] = set()

    def visit(v: Var) -> None:
        if v.topic in visited:
            return
        visited.add(v.topic)
        for p in v.parents:
            visit(p)
        order.append(v)

    visit(result)
    order.reverse()

    # seed node
    seed_topic: str = "grad_%s" % result.topic
    g.node(seed_topic, inputs={"seed": "seed"},
           fn=lambda **kw: kw["seed"], outs=[seed_topic])

    contribs: Dict[str, List[str]] = {}

    for var in order:
        if var.op is None:
            continue
        grad_out: str = "grad_%s" % var.topic
        input_topics: List[str] = [p.topic for p in var.parents]
        grad_contribs: List[str] = var.op.backward(g, grad_out, *input_topics)
        for parent, contrib in zip(var.parents, grad_contribs):
            contribs.setdefault(parent.topic, []).append(contrib)

    # accumulate and register grad_<topic> for each var
    registered: Set[str] = set()
    grad_map: Dict[str, str] = {}

    for var in order:
        parts: List[str] | None = contribs.get(var.topic)
        if parts is None:
            continue
        grad_topic: str = "grad_%s" % var.topic
        if grad_topic in registered:
            grad_map[var.topic] = grad_topic
            continue
        registered.add(grad_topic)
        if len(parts) == 1:
            p = parts[0]
            g.node(grad_topic, inputs={p: p},
                   fn=_pass_fn(p), outs=[grad_topic])
        else:
            g.node(grad_topic, inputs={t: t for t in parts},
                   fn=_sum_fn(list(parts)), outs=[grad_topic])
        grad_map[var.topic] = grad_topic

    return grad_map


def main() -> None:
    g = DemandGraph()
    x = Var(g, "x")
    y = Var(g, "y")

    # f(x, y) = (x*y + sin(x)) * exp(y)
    f = (x * y + x.sin()) * y.exp()

    # one call builds the backward graph, returns gradient Vars
    df_dx, df_dy = f.grad(x, y)

    x_val, y_val = 1.5, 2.0
    inputs: Dict[str, float] = dict(x=x_val, y=y_val, seed=1.0)

    gx: float = g.demand(df_dx.topic, **inputs)
    gy: float = g.demand(df_dy.topic, **inputs)
    fv: float = g.demand(f.topic, **inputs)

    eps = 1e-7
    f_fn: Callable[[float, float], float] = lambda x, y: (x * y + np.sin(x)) * np.exp(y)
    num_dx: float = (f_fn(x_val + eps, y_val) - f_fn(x_val, y_val)) / eps
    num_dy: float = (f_fn(x_val, y_val + eps) - f_fn(x_val, y_val)) / eps

    print("f(%.1f, %.1f) = %.6f" % (x_val, y_val, fv))
    print("df/dx: analytic=%.6f  numerical=%.6f" % (gx, num_dx))
    print("df/dy: analytic=%.6f  numerical=%.6f" % (gy, num_dy))
    assert abs(gx - num_dx) < 1e-5
    assert abs(gy - num_dy) < 1e-5
    print("Operator-overloading AD verified.")


if __name__ == "__main__":
    main()
