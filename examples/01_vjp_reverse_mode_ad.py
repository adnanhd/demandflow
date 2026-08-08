"""VJP (reverse-mode AD) as pull-mode dataflow.

Reverse-mode AD propagates adjoints BACKWARD from the loss to the inputs,
just like DemandGraph propagates demand backward from subscribers to sources.
Each node pulls its upstream value only when demanded.

Computes gradients of f(x, y) = (x*y + sin(x)) * exp(y) using numpy.
Multiple demands on the same graph share memoized intermediate results.
"""

import numpy as np

from demandflow.demand import DemandGraph


def vjp_example():
    g = DemandGraph()

    # Convention: each node's input param names match output topic names
    # from upstream nodes (or kwargs for leaf inputs).

    # --- Forward pass nodes ---

    # a = x*y
    g.node("a", inputs={"x": "x", "y": "y"}, fn=lambda x, y: x * y, outs=["a"])
    g.node("da_dx", inputs={"y": "y"}, fn=lambda y: y, outs=["da_dx"])
    g.node("da_dy", inputs={"x": "x"}, fn=lambda x: x, outs=["da_dy"])

    # b = sin(x)
    g.node("b", inputs={"x": "x"}, fn=lambda x: np.sin(x), outs=["b"])
    g.node("db_dx", inputs={"x": "x"}, fn=lambda x: np.cos(x), outs=["db_dx"])

    # c = a + b
    g.node("c", inputs={"a": "a", "b": "b"}, fn=lambda a, b: a + b, outs=["c"])

    # d = exp(y)
    g.node("d", inputs={"y": "y"}, fn=lambda y: np.exp(y), outs=["d"])
    g.node("dd_dy", inputs={"y": "y"}, fn=lambda y: np.exp(y), outs=["dd_dy"])

    # f = c * d
    g.node("f", inputs={"c": "c", "d": "d"}, fn=lambda c, d: c * d, outs=["f"])
    g.node("df_dc", inputs={"d": "d"}, fn=lambda d: d, outs=["df_dc"])
    g.node("df_dd", inputs={"c": "c"}, fn=lambda c: c, outs=["df_dd"])

    # --- Backward pass: VJP pulls adjoints ---

    # grad_c = seed * df_dc, grad_d = seed * df_dd
    g.node("grad_c", inputs={"seed": "seed", "df_dc": "df_dc"},
           fn=lambda seed, df_dc: seed * df_dc, outs=["grad_c"])
    g.node("grad_d", inputs={"seed": "seed", "df_dd": "df_dd"},
           fn=lambda seed, df_dd: seed * df_dd, outs=["grad_d"])

    # addition passthrough
    g.node("grad_a", inputs={"grad_c": "grad_c"},
           fn=lambda grad_c: grad_c, outs=["grad_a"])
    g.node("grad_b", inputs={"grad_c": "grad_c"},
           fn=lambda grad_c: grad_c, outs=["grad_b"])

    # grad_x = grad_a * da_dx + grad_b * db_dx
    g.node("grad_x_mul", inputs={"grad_a": "grad_a", "da_dx": "da_dx"},
           fn=lambda grad_a, da_dx: grad_a * da_dx, outs=["grad_x_mul"])
    g.node("grad_x_sin", inputs={"grad_b": "grad_b", "db_dx": "db_dx"},
           fn=lambda grad_b, db_dx: grad_b * db_dx, outs=["grad_x_sin"])
    g.node("grad_x", inputs={"grad_x_mul": "grad_x_mul", "grad_x_sin": "grad_x_sin"},
           fn=lambda grad_x_mul, grad_x_sin: grad_x_mul + grad_x_sin, outs=["grad_x"])

    # grad_y = grad_a * da_dy + grad_d * dd_dy
    g.node("grad_y_mul", inputs={"grad_a": "grad_a", "da_dy": "da_dy"},
           fn=lambda grad_a, da_dy: grad_a * da_dy, outs=["grad_y_mul"])
    g.node("grad_y_exp", inputs={"grad_d": "grad_d", "dd_dy": "dd_dy"},
           fn=lambda grad_d, dd_dy: grad_d * dd_dy, outs=["grad_y_exp"])
    g.node("grad_y", inputs={"grad_y_mul": "grad_y_mul", "grad_y_exp": "grad_y_exp"},
           fn=lambda grad_y_mul, grad_y_exp: grad_y_mul + grad_y_exp, outs=["grad_y"])

    # --- Demand both gradients (shared memoization on second call) ---
    x_val, y_val = 1.5, 2.0

    grad_x = g.demand("grad_x", x=x_val, y=y_val, seed=1.0)
    grad_y = g.demand("grad_y", x=x_val, y=y_val, seed=1.0)
    f_val = g.demand("f", x=x_val, y=y_val, seed=1.0)

    # Numerical verification
    eps = 1e-7
    f_fn = lambda x, y: (x * y + np.sin(x)) * np.exp(y)
    numerical_dx = (f_fn(x_val + eps, y_val) - f_fn(x_val, y_val)) / eps
    numerical_dy = (f_fn(x_val, y_val + eps) - f_fn(x_val, y_val)) / eps

    print("f(%.1f, %.1f) = %.6f" % (x_val, y_val, f_val))
    print("df/dx: analytic=%.6f  numerical=%.6f" % (grad_x, numerical_dx))
    print("df/dy: analytic=%.6f  numerical=%.6f" % (grad_y, numerical_dy))
    assert abs(grad_x - numerical_dx) < 1e-5
    assert abs(grad_y - numerical_dy) < 1e-5
    print("VJP (pull-mode AD) verified.")


if __name__ == "__main__":
    vjp_example()
