"""Multithreaded demand on a shared DemandGraph.

Multiple threads demand different targets from the same graph concurrently.
The MemoStore is shared across calls, so a node computed by one thread
provides a cache hit for subsequent threads demanding the same subgraph.

Example: two threads demand df/dx and df/dy simultaneously from the VJP
graph. The shared forward-pass nodes (a, b, c, d, f) are computed once;
the second demand gets a MemoStore hit.
"""

import threading
import time

import numpy as np

from demandflow.demand import DemandGraph


def main():
    g = DemandGraph()
    call_count = {}
    count_lock = threading.Lock()

    def tracked(name, fn):
        def wrapper(**kwargs):
            with count_lock:
                call_count[name] = call_count.get(name, 0) + 1
            time.sleep(0.01)  # simulate work
            return fn(**kwargs)
        return wrapper

    # Forward pass
    g.node("a", inputs={"x": "x", "y": "y"},
           fn=tracked("a", lambda x, y: x * y), outs=["a"])
    g.node("da_dx", inputs={"y": "y"},
           fn=tracked("da_dx", lambda y: y), outs=["da_dx"])
    g.node("da_dy", inputs={"x": "x"},
           fn=tracked("da_dy", lambda x: x), outs=["da_dy"])
    g.node("b", inputs={"x": "x"},
           fn=tracked("b", lambda x: np.sin(x)), outs=["b"])
    g.node("db_dx", inputs={"x": "x"},
           fn=tracked("db_dx", lambda x: np.cos(x)), outs=["db_dx"])
    g.node("c", inputs={"a": "a", "b": "b"},
           fn=tracked("c", lambda a, b: a + b), outs=["c"])
    g.node("d", inputs={"y": "y"},
           fn=tracked("d", lambda y: np.exp(y)), outs=["d"])
    g.node("dd_dy", inputs={"y": "y"},
           fn=tracked("dd_dy", lambda y: np.exp(y)), outs=["dd_dy"])
    g.node("f", inputs={"c": "c", "d": "d"},
           fn=tracked("f", lambda c, d: c * d), outs=["f"])
    g.node("df_dc", inputs={"d": "d"},
           fn=tracked("df_dc", lambda d: d), outs=["df_dc"])
    g.node("df_dd", inputs={"c": "c"},
           fn=tracked("df_dd", lambda c: c), outs=["df_dd"])

    # Backward pass
    g.node("grad_c", inputs={"seed": "seed", "df_dc": "df_dc"},
           fn=tracked("grad_c", lambda seed, df_dc: seed * df_dc), outs=["grad_c"])
    g.node("grad_d", inputs={"seed": "seed", "df_dd": "df_dd"},
           fn=tracked("grad_d", lambda seed, df_dd: seed * df_dd), outs=["grad_d"])
    g.node("grad_a", inputs={"grad_c": "grad_c"},
           fn=tracked("grad_a", lambda grad_c: grad_c), outs=["grad_a"])
    g.node("grad_b", inputs={"grad_c": "grad_c"},
           fn=tracked("grad_b", lambda grad_c: grad_c), outs=["grad_b"])
    g.node("grad_x_mul", inputs={"grad_a": "grad_a", "da_dx": "da_dx"},
           fn=tracked("grad_x_mul", lambda grad_a, da_dx: grad_a * da_dx),
           outs=["grad_x_mul"])
    g.node("grad_x_sin", inputs={"grad_b": "grad_b", "db_dx": "db_dx"},
           fn=tracked("grad_x_sin", lambda grad_b, db_dx: grad_b * db_dx),
           outs=["grad_x_sin"])
    g.node("grad_x", inputs={"grad_x_mul": "grad_x_mul", "grad_x_sin": "grad_x_sin"},
           fn=tracked("grad_x", lambda grad_x_mul, grad_x_sin: grad_x_mul + grad_x_sin),
           outs=["grad_x"])
    g.node("grad_y_mul", inputs={"grad_a": "grad_a", "da_dy": "da_dy"},
           fn=tracked("grad_y_mul", lambda grad_a, da_dy: grad_a * da_dy),
           outs=["grad_y_mul"])
    g.node("grad_y_exp", inputs={"grad_d": "grad_d", "dd_dy": "dd_dy"},
           fn=tracked("grad_y_exp", lambda grad_d, dd_dy: grad_d * dd_dy),
           outs=["grad_y_exp"])
    g.node("grad_y", inputs={"grad_y_mul": "grad_y_mul", "grad_y_exp": "grad_y_exp"},
           fn=tracked("grad_y", lambda grad_y_mul, grad_y_exp: grad_y_mul + grad_y_exp),
           outs=["grad_y"])

    x_val, y_val = 1.5, 2.0
    inputs = dict(x=x_val, y=y_val, seed=1.0)
    results = {}

    def worker(target):
        results[target] = g.demand(target, **inputs)

    # Thread 1 demands grad_x, Thread 2 demands grad_y simultaneously.
    # They share many forward-pass nodes (a, b, c, d, f) and backward
    # nodes (grad_c, grad_d, grad_a, grad_b).
    t1 = threading.Thread(target=worker, args=("grad_x",), name="T-grad_x")
    t2 = threading.Thread(target=worker, args=("grad_y",), name="T-grad_y")

    start = time.time()
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    elapsed = time.time() - start

    print("grad_x = %.6f" % results["grad_x"])
    print("grad_y = %.6f" % results["grad_y"])

    # Node execution counts: shared nodes should be computed only once
    # thanks to MemoStore (second demand gets cache hit).
    shared = ["a", "b", "c", "d", "f", "df_dc", "df_dd", "grad_c", "grad_d",
              "grad_a", "grad_b"]
    print("\nNode execution counts:")
    total_saved = 0
    for node_name in sorted(call_count):
        count = call_count[node_name]
        marker = " (shared, memo hit)" if node_name in shared and count == 1 else ""
        if node_name in shared and count == 1:
            total_saved += 1
        print("  %-15s: %d%s" % (node_name, count, marker))

    print("\n%d/%d shared nodes computed only once (memo cache hit)" % (
        total_saved, len(shared)))
    print("Total time: %.3fs (vs %.3fs sequential)" % (
        elapsed, len(call_count) * 0.01))

    # Verify correctness
    f_fn = lambda x, y: (x * y + np.sin(x)) * np.exp(y)
    eps = 1e-7
    numerical_dx = (f_fn(x_val + eps, y_val) - f_fn(x_val, y_val)) / eps
    numerical_dy = (f_fn(x_val, y_val + eps) - f_fn(x_val, y_val)) / eps
    assert abs(results["grad_x"] - numerical_dx) < 1e-5
    assert abs(results["grad_y"] - numerical_dy) < 1e-5
    print("\nMultithreaded demand verified.")


if __name__ == "__main__":
    main()
