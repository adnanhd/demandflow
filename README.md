# demandflow (Flow On-Demand)

Pull-mode dataflow engine with content-addressed memoization.

Nothing runs unless something demands it. A `DemandGraph` wires
functions as nodes connected by named topics. Two execution modes:

- **`demand(target, **kwargs)`** -- one-shot topological resolve with
  MemoStore caching. Subsequent demands sharing subgraphs get cache hits.
- **`flow()`** -- streaming source-driven mode. Sources are pulled only
  when a subscriber demands their output, nodes fire once all their input
  topics are fresh, subscribers fire. Used for per-batch training loops.

## Features

- **Content-addressed memoization** via `MemoStore` + Merkle-style `node_key`
- **CachePolicy** (accept/check/fresh/force) with tolerance and custom comparators
- **ResourcePolicy** (exclusive/shared with capacity) for concurrency control
- **Recurrence** nodes for iterative computations (epoch loops) with per-iteration memo skip
- **Triggers** (Lingua Franca pattern) for mixed data/event-driven nodes
- **Relations**: `@graph.relation` declares inputs with `Annotated[T, Demand(...)]`
  and the output with `Annotated[T, Flow("name")]`; `demand_relation` resolves them
- **Topics** with schema and QoS (`graph.topic(name, schema=..., qos=...)`;
  `topic_schema` is an alias)
- **`NodeInterceptor`** hooks before and after each node and source compute
- **ROS2-style** nodes, lifecycle nodes, publishers and subscribers,
  services, actions, timers and executors over zenoh transport

zenoh is imported lazily, so importing demandflow works without the zenoh
C extension; the ROS2-style parts need it.

## Install

```bash
pip install -e ".[dev]"      # pytest; ".[docs]" adds sphinx, furo, sphinx-autodoc-typehints
```

Python 3.8 or later; the one dependency is `eclipse-zenoh>=1.0`.

## Examples

```bash
python examples/01_vjp_reverse_mode_ad.py       # reverse-mode AD as pull-mode
python examples/02_dp_bottom_up.py              # edit distance via demand graph
python examples/03_multithread_demand.py        # concurrent demands, shared memo
python examples/04_ad_operator_overloading.py   # AD through overloaded operators (numpy)
```

See also [eventforge](https://github.com/adnanhd/observer-pattern) for the
push-mode counterpart (JVP forward-mode AD, top-down DP).

## Tests and docs

`pytest` runs `tests/` (63 tests). `docs/` is a Sphinx project
(`index.rst`, `quickstart.rst`, `concepts.rst`, one page per module under
`docs/api/`): `sphinx-build docs docs/_build`.

## On this machine

The checkout is `~/src/personal/demandflow`; it is not installed in the
system Python. torchestrator depends on it
(`demandflow @ git+https://github.com/adnanhd/demandflow.git`) and wires it in
through `torchestrator/plugin/demandflow.py`.
