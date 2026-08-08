# Flow On-Demand

Pull-mode dataflow engine with content-addressed memoization.

Nothing runs unless something demands it. A `DemandGraph` wires
functions as nodes connected by named topics. Two execution modes:

- **`demand(target, **kwargs)`** -- one-shot topological resolve with
  MemoStore caching. Subsequent demands sharing subgraphs get cache hits.
- **`flow()`** -- streaming source-driven mode. Sources are pulled, nodes
  cascade, subscribers fire. Used for per-batch training loops.

## Features

- **Content-addressed memoization** via `MemoStore` + Merkle-style `node_key`
- **CachePolicy** (accept/check/fresh/force) with tolerance and custom comparators
- **ResourcePolicy** (exclusive/shared with capacity) for concurrency control
- **Recurrence** nodes for iterative computations (epoch loops) with per-iteration memo skip
- **Triggers** (Lingua Franca pattern) for mixed data/event-driven nodes
- **ROS2-style** nodes, topics, services, actions over zenoh transport

## Examples

```bash
python examples/01_vjp_reverse_mode_ad.py   # reverse-mode AD as pull-mode
python examples/02_dp_bottom_up.py          # edit distance via demand graph
python examples/03_multithread_demand.py    # concurrent demands, shared memo
```

See also [eventforge](https://github.com/adnanhd/observer-pattern) for the
push-mode counterpart (JVP forward-mode AD, top-down DP).
