Quickstart
==========

Install
-------

.. code-block:: bash

   pip install -e .

One-shot demand
---------------

Register functions as nodes, then ``demand()`` a target.
Only the subgraph needed for that target runs.

.. code-block:: python

   from demandflow.demand import DemandGraph

   g = DemandGraph()

   @g.compute
   def add(x, y):
       return x + y

   @g.compute
   def double(add):
       return add * 2

   print(g.demand("double", x=3, y=4))  # 14

Streaming flow
--------------

For streaming workloads, use sources, subscribers, and ``flow()``.

.. code-block:: python

   g = DemandGraph()
   data = iter([1, 2, 3])
   g.source("src", out="val", pull=lambda: next(data))

   @g.compute
   def squared(val):
       return val ** 2

   g.subscribe("squared", print)
   while g.flow():
       pass
   # prints 1, 4, 9
