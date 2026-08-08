Concepts
========

Nodes and Topics
----------------

A **node** is a named function with typed inputs and outputs. Inputs and
outputs are **topics** -- named data channels. A node fires only when all
its input topics have values.

.. code-block:: python

   g.node("preprocess", inputs={"raw": "raw_data"},
          fn=lambda raw: clean(raw), outs=["cleaned"])

Two Execution Modes
-------------------

**demand(target, \*\*kwargs)**
   One-shot topological resolve. Walks backward from the target,
   computes each node once, caches results in MemoStore. Subsequent
   demands sharing subgraphs get cache hits.

**flow()**
   Streaming source-driven mode. Pulls every demanded source, cascades
   through barrier nodes, delivers to subscribers. Returns ``True`` if
   any source emitted; ``False`` when all sources are exhausted.

Content-Addressed Memoization
-----------------------------

Each node result is keyed by ``node_key(name, input_hashes)`` -- a
Merkle-style hash of the node name and its input content hashes.
Identical inputs produce identical keys, so repeated demands skip
recomputation.

CachePolicy controls behavior:

- ``ACCEPT`` -- use cached result if available (default)
- ``FRESH`` -- always recompute, update cache
- ``FORCE`` -- always recompute, update cache
- ``CHECK`` -- recompute and compare against cache (tolerance-based)

Demand-Driven Laziness
----------------------

A source is pulled only if its output is **demanded** -- i.e., a
subscriber exists on a downstream topic, directly or transitively.
Subscribing a topic activates its upstream chain; unsubscribing
deactivates it.

Triggers
--------

A node can declare **trigger** topics in addition to data inputs. The
node fires only when all data inputs are fresh AND at least one trigger
has fired. Nodes without triggers fire on data readiness alone.

Recurrence
----------

For iterative computations (e.g., epoch loops), ``recurrence()``
registers a node that feeds its output back as input for a fixed number
of iterations, with optional per-iteration memo skip.

ResourcePolicy
--------------

Controls concurrency in the ``flow()`` path:

- ``EXCLUSIVE`` -- only one consumer at a time
- ``SHARED(n)`` -- up to *n* concurrent consumers
